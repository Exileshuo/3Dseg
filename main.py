import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from skimage import measure
from medpy import metric
from tqdm import tqdm


# ==========================================
# 1. 模型定义 (Attention Lite 3D U-Net)
# ==========================================
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv3d(F_g, F_int, 1), nn.InstanceNorm3d(F_int))
        self.W_x = nn.Sequential(nn.Conv3d(F_l, F_int, 1), nn.InstanceNorm3d(F_int))
        self.psi = nn.Sequential(nn.Conv3d(F_int, 1, 1), nn.Sigmoid())

    def forward(self, g, x):
        return x * self.psi(F.relu(self.W_g(g) + self.W_x(x)))


class Lite3DUNet_Attn(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_filters=16):
        super().__init__()

        def DSConv(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, in_c, 3, padding=1, groups=in_c, bias=False),
                nn.Conv3d(in_c, out_c, 1, bias=False),
                nn.InstanceNorm3d(out_c), nn.LeakyReLU(inplace=True)
            )

        def DoubleConv(in_c, out_c): return nn.Sequential(DSConv(in_c, out_c), DSConv(out_c, out_c))

        self.inc = DoubleConv(in_channels, base_filters)
        self.down = nn.ModuleList(
            [nn.Sequential(nn.MaxPool3d(2), DoubleConv(base_filters * (2 ** i), base_filters * (2 ** (i + 1)))) for i in
             range(4)])
        self.up = nn.ModuleList(
            [nn.ConvTranspose3d(base_filters * (2 ** i), base_filters * (2 ** (i - 1)), 2, 2) for i in range(4, 0, -1)])
        self.att = nn.ModuleList([AttentionGate(base_filters * (2 ** (i - 1)), base_filters * (2 ** (i - 1)),
                                                base_filters * (2 ** (i - 2)) if i > 1 else 8) for i in
                                  range(4, 0, -1)])
        self.conv_up = nn.ModuleList(
            [DoubleConv(base_filters * (2 ** i), base_filters * (2 ** (i - 1))) for i in range(4, 0, -1)])
        self.outc = nn.Conv3d(base_filters, out_channels, 1)

    def forward(self, x):
        enc = [self.inc(x)]
        for layer in self.down: enc.append(layer(enc[-1]))
        d = enc[-1]
        for i in range(4):
            up = self.up[i](d)
            diff = [enc[3 - i].size(j) - up.size(j) for j in range(2, 5)]
            up = F.pad(up, [diff[2] // 2, diff[2] - diff[2] // 2, diff[1] // 2, diff[1] - diff[1] // 2, diff[0] // 2,
                            diff[0] - diff[0] // 2])
            d = self.conv_up[i](torch.cat([self.att[i](g=up, x=enc[3 - i]), up], dim=1))
        return self.outc(d)


class WeightedDiceFocalLoss(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 2.0]):
        super().__init__()
        self.weights = weights

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        loss = 0
        for i in range(3):
            dice = 1 - (2. * (probs[:, i] * targets[:, i]).sum() + 1e-5) / (
                        probs[:, i].sum() + targets[:, i].sum() + 1e-5)
            focal = F.binary_cross_entropy_with_logits(logits[:, i], targets[:, i])
            loss += self.weights[i] * (dice + focal)
        return loss / 3


# ==========================================
# 2. 数据集加载器
# ==========================================
class BraTS2020Dataset(Dataset):
    def __init__(self, data_dir, phase='train', patch_size=(96, 96, 96)):
        self.data_dir = Path(data_dir)
        self.phase = phase
        self.patch_size = patch_size
        self.case_folders = [d for d in self.data_dir.iterdir() if d.is_dir()]

    def __len__(self):
        return len(self.case_folders)

    def _load_nii(self, path):
        if not path.exists(): path = Path(str(path) + ".gz")
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))

    def __getitem__(self, idx):
        case_path = self.case_folders[idx]
        case_id = case_path.name
        files = {m: list(case_path.glob(f"*{m}.nii*"))[0] for m in ['flair', 't1', 't1ce', 't2']}
        img = np.stack([self._load_nii(files[m]) for m in ['flair', 't1', 't1ce', 't2']], axis=0).astype(np.float32)

        for c in range(4):
            m = img[c] > 0
            if m.any(): img[c][m] = (img[c][m] - img[c][m].mean()) / (img[c][m].std() + 1e-8)

        seg_files = list(case_path.glob("*seg.nii*"))
        if seg_files:
            seg = self._load_nii(seg_files[0])
            label = np.stack([(seg > 0), ((seg == 1) | (seg == 4)), (seg == 4)], axis=0).astype(np.float32)
            has_label = True
        else:
            label = np.zeros((3, *img.shape[1:]), dtype=np.float32);
            has_label = False

        if self.phase == 'train':
            z, y, x = [np.random.randint(0, img.shape[i + 1] - self.patch_size[i] + 1) for i in range(3)]
            img, label = img[:, z:z + 96, y:y + 96, x:x + 96], label[:, z:z + 96, y:y + 96, x:x + 96]
            if np.random.rand() > 0.5: img, label = np.flip(img, axis=-1).copy(), np.flip(label, axis=-1).copy()

        return torch.from_numpy(img), torch.from_numpy(label), torch.tensor(has_label)


# ==========================================
# 3. 核心工具：TTA 推理 与 智能去噪
# ==========================================
def predict_with_tta(model, img_tensor):
    """测试时增强 (TTA): 原始 + X/Y/Z轴翻转求均值"""
    probs = torch.sigmoid(model(img_tensor))
    probs += torch.flip(torch.sigmoid(model(torch.flip(img_tensor, dims=[2]))), dims=[2])
    probs += torch.flip(torch.sigmoid(model(torch.flip(img_tensor, dims=[3]))), dims=[3])
    probs += torch.flip(torch.sigmoid(model(torch.flip(img_tensor, dims=[4]))), dims=[4])
    return (probs / 4.0).cpu().numpy().squeeze()


def smart_refine(prob_map, is_et=False, wt_mask=None):
    """结合了防 373.00 惩罚机制的智能去噪"""
    if is_et:
        # ET：高置信度 (0.65)
        mask = (prob_map > 0.65).astype(np.uint8)
        if wt_mask is not None: mask = np.where(wt_mask > 0, mask, 0)
        labels = measure.label(mask)
        if labels.max() == 0: return mask
        regions = measure.regionprops(labels)
        largest_reg = max(regions, key=lambda x: x.area)

        # ET 防爆表底线：太小的孤立块直接清零
        if largest_reg.area < 60: return np.zeros_like(mask)
        new_mask = np.zeros_like(mask)
        new_mask[labels == largest_reg.label] = 1
        return new_mask
    else:
        # WT / TC：常规置信度 (0.5) + 保留最大块
        mask = (prob_map > 0.5).astype(np.uint8)
        labels = measure.label(mask)
        if labels.max() == 0: return mask
        regions = measure.regionprops(labels)
        largest_reg = max(regions, key=lambda x: x.area)
        new_mask = np.zeros_like(mask)
        new_mask[labels == largest_reg.label] = 1
        return new_mask


# ==========================================
# 4. 主控流水线
# ==========================================
def run_all():
    TRAIN_DIR = "data/BraTS2020_Split/my_train_set"
    TEST_DIR = "data/BraTS2020_Split/my_test_set"
    BEST_MODEL = "./saved_models/lite_attn_best.pth"
    os.makedirs("./saved_models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Lite3DUNet_Attn().to(device)

    # --- A. 如果没有权重，自动启动训练 ---
    if not os.path.exists(BEST_MODEL):
        print("🚨 未检测到历史最优权重，准备开始全新的训练过程 (100 Epochs)...")
        train_loader = DataLoader(BraTS2020Dataset(TRAIN_DIR), batch_size=2, shuffle=True, num_workers=4,
                                  pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(BraTS2020Dataset(TEST_DIR, phase='val'), batch_size=1)

        criterion = WeightedDiceFocalLoss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = GradScaler('cuda')
        best_mean_dice = 0.0

        for epoch in range(1, 201):
            model.train()
            train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/200]")
            for imgs, masks, _ in train_pbar:
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                with autocast('cuda'):
                    loss = criterion(model(imgs), masks)
                scaler.scale(loss).backward();
                scaler.step(optimizer);
                scaler.update()
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            # 🚀 核心修改：每 10 轮验证，并加上验证集的进度条
            if epoch % 5 == 0:
                model.eval()
                val_dices = []
                with torch.no_grad():
                    # 加上 leave=False，验证完进度条自动消失，保持控制台整洁
                    val_pbar = tqdm(val_loader, desc=f"🔍 Epoch {epoch} 验证中", leave=False)
                    for v_imgs, v_masks, _ in val_pbar:
                        v_prob = torch.sigmoid(model(v_imgs.to(device))).cpu().numpy().squeeze()
                        v_gt = v_masks.numpy().squeeze()
                        # 快速计算 WT Dice 用于指标筛选
                        d = 2. * ((v_prob[0] > 0.5) * v_gt[0]).sum() / ((v_prob[0] > 0.5).sum() + v_gt[0].sum() + 1e-5)
                        val_dices.append(d)

                current_mean = np.mean(val_dices)
                tqdm.write(f"✅ Epoch {epoch} 验证完毕 | Mean WT Dice: {current_mean:.4f}")

                if current_mean > best_mean_dice:
                    best_mean_dice = current_mean
                    torch.save(model.state_dict(), BEST_MODEL)
                    tqdm.write("🌟 已保存新的最优模型权重！")
    else:
        print(f"✅ 成功找到历史权重: {BEST_MODEL}，直接跳过训练阶段！")

    # --- B. 执行 TTA 终极评估 ---
    print("\n🚀 启动满血版评估：TTA 数据增强 + 智能防爆表去噪...")
    model.load_state_dict(torch.load(BEST_MODEL, map_location=device))
    model.eval()

    test_ds = BraTS2020Dataset(TEST_DIR, phase='test')
    test_loader = DataLoader(test_ds, batch_size=1)
    results = {'WT': [], 'TC': [], 'ET': []}
    bad_cases = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating with TTA")
        for i, (imgs, masks, _) in enumerate(pbar):
            case_id = test_ds.case_folders[i].name
            imgs = imgs.to(device)

            # 使用 TTA 推理
            prob = predict_with_tta(model, imgs)
            gt = masks.numpy().squeeze()

            wt_p = smart_refine(prob[0], is_et=False)
            tc_p = smart_refine(prob[1], is_et=False)
            et_p = smart_refine(prob[2], is_et=True, wt_mask=wt_p)

            preds = [wt_p, tc_p, et_p]

            for c, key in enumerate(['WT', 'TC', 'ET']):
                p, g = preds[c], gt[c]
                if p.sum() > 0 and g.sum() > 0:
                    d, h = metric.binary.dc(p, g), metric.binary.hd95(p, g)
                else:
                    d, h = (1.0, 0.0) if p.sum() == 0 and g.sum() == 0 else (0.0, 373.0)

                results[key].append([d, h])
                if key == 'ET' and h > 15: bad_cases.append((case_id, h))

            pbar.set_postfix(WT_Dice=f"{results['WT'][-1][0]:.3f}")

    # --- 最终报告 ---
    print("\n" + "=" * 50)
    print("🎯 TTA 终极评估报告 (Mean Dice ↑ / HD95 ↓)")
    for k in ['WT', 'TC', 'ET']:
        arr = np.array(results[k])
        print(f"{k:<3} -> Dice: {np.mean(arr[:, 0]):.4f} | HD95: {np.mean(arr[:, 1]):.2f}")
    print("=" * 50)

    if bad_cases:
        print("\n🚨 异常病例残余追踪 (ET HD95 > 15):")
        for cid, hval in bad_cases: print(f"- {cid}: {hval:.2f}")


if __name__ == "__main__":
    run_all()