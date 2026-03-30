import os
import csv
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
from datetime import datetime
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from skimage import measure
from medpy import metric
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage import binary_closing, generate_binary_structure


# ==========================================
# 1. 核心架构：Res-SE Block + 稳健型全局门控
# ==========================================
class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation 通道注意力机制：动态捕捉模态权重"""

    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


def DSConv(in_c, out_c):
    return nn.Sequential(
        nn.Conv3d(in_c, in_c, 3, padding=1, groups=in_c, bias=False),
        nn.Conv3d(in_c, out_c, 1, bias=False),
        nn.InstanceNorm3d(out_c),
        nn.LeakyReLU(inplace=True)
    )


class ResSEBlock(nn.Module):
    """残差连接 + SE 通道注意力"""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = DSConv(in_c, out_c)
        self.conv2 = DSConv(out_c, out_c)
        self.se = SEBlock3D(out_c)

        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_c, out_c, 1, bias=False),
                nn.InstanceNorm3d(out_c)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        return F.leaky_relu(out + res, inplace=True)


class GlobalGatedTail3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.large_kernel = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim),
            nn.InstanceNorm3d(dim)
        )
        self.proj_gate = nn.Sequential(nn.Conv3d(dim, dim, 1), nn.InstanceNorm3d(dim))
        self.proj_value = nn.Sequential(nn.Conv3d(dim, dim, 1), nn.InstanceNorm3d(dim))
        self.act = nn.SiLU()
        self.proj_out = nn.Sequential(nn.Conv3d(dim, dim, 1), nn.InstanceNorm3d(dim))

    def forward(self, x):
        residual = x
        x_spatial = self.large_kernel(x)
        gate = self.act(self.proj_gate(x_spatial))
        value = self.proj_value(x)
        out = self.proj_out(gate * value)
        return out + residual


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv3d(F_g, F_int, 1), nn.InstanceNorm3d(F_int))
        self.W_x = nn.Sequential(nn.Conv3d(F_l, F_int, 1), nn.InstanceNorm3d(F_int))
        self.psi = nn.Sequential(nn.Conv3d(F_int, 1, 1), nn.Sigmoid())

    def forward(self, g, x):
        return x * self.psi(F.relu(self.W_g(g) + self.W_x(x)))


class StableHybrid3DUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_filters=16):
        super().__init__()

        self.inc = ResSEBlock(in_channels, base_filters)
        self.down = nn.ModuleList([
            nn.Sequential(nn.MaxPool3d(2), ResSEBlock(base_filters * (2 ** i), base_filters * (2 ** (i + 1))))
            for i in range(3)
        ])

        bottleneck_dim = base_filters * 8
        self.bottleneck_down = nn.MaxPool3d(2)
        self.bottleneck_tail = nn.Sequential(
            ResSEBlock(bottleneck_dim, bottleneck_dim * 2),
            GlobalGatedTail3D(bottleneck_dim * 2),
            GlobalGatedTail3D(bottleneck_dim * 2)
        )

        self.up = nn.ModuleList(
            [nn.ConvTranspose3d(base_filters * (2 ** i), base_filters * (2 ** (i - 1)), 2, 2) for i in range(4, 0, -1)])
        self.att = nn.ModuleList([AttentionGate(base_filters * (2 ** (i - 1)), base_filters * (2 ** (i - 1)),
                                                base_filters * (2 ** (i - 2)) if i > 1 else 8) for i in
                                  range(4, 0, -1)])
        self.conv_up = nn.ModuleList(
            [ResSEBlock(base_filters * (2 ** i), base_filters * (2 ** (i - 1))) for i in range(4, 0, -1)])
        self.outc = nn.Conv3d(base_filters, out_channels, 1)

    def forward(self, x):
        enc = [self.inc(x)]
        for layer in self.down: enc.append(layer(enc[-1]))

        b_in = self.bottleneck_down(enc[-1])
        d = self.bottleneck_tail(b_in)

        for i in range(4):
            up = self.up[i](d)
            diff = [enc[3 - i].size(j) - up.size(j) for j in range(2, 5)]
            up = F.pad(up, [diff[2] // 2, diff[2] - diff[2] // 2, diff[1] // 2, diff[1] - diff[1] // 2, diff[0] // 2,
                            diff[0] - diff[0] // 2])
            d = self.conv_up[i](torch.cat([self.att[i](g=up, x=enc[3 - i]), up], dim=1))
        return self.outc(d)


# ==========================================
# 2. 损失函数 (ET 偏执型权重分配 + 强转 float32)
# ==========================================
class DiceFocalBoundaryLoss(nn.Module):
    def __init__(self, weights=[1.0, 1.5, 3.0], boundary_weight=0.1):
        super().__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32).cuda()
        self.boundary_weight = boundary_weight

    def extract_boundary(self, mask):
        dilation = F.max_pool3d(mask, kernel_size=3, stride=1, padding=1)
        erosion = -F.max_pool3d(-mask, kernel_size=3, stride=1, padding=1)
        return dilation - erosion

    def forward(self, logits, targets):
        logits = logits.float()
        targets = targets.float()

        probs = torch.sigmoid(logits)
        pred_boundary = self.extract_boundary(probs)
        gt_boundary = self.extract_boundary(targets)

        loss = 0
        for i in range(3):
            dice = 1 - (2. * (probs[:, i] * targets[:, i]).sum() + 1e-4) / (
                        probs[:, i].sum() + targets[:, i].sum() + 1e-4)
            focal = F.binary_cross_entropy_with_logits(logits[:, i], targets[:, i])
            boundary_loss = F.l1_loss(pred_boundary[:, i], gt_boundary[:, i])

            channel_loss = dice + focal + self.boundary_weight * boundary_loss
            loss += self.weights[i] * channel_loss

        return loss / 3


# ==========================================
# 3. 数据集加载器
# ==========================================
class BraTS2020Dataset(Dataset):
    def __init__(self, case_folders, phase='train', patch_size=(96, 96, 96)):
        self.case_folders = case_folders
        self.phase = phase
        self.patch_size = patch_size

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
# 4. 后处理：临床解剖学级联约束 (剔除激进规则)
# ==========================================
def predict_with_tta(model, img_tensor):
    probs = torch.sigmoid(model(img_tensor))
    probs += torch.flip(torch.sigmoid(model(torch.flip(img_tensor, dims=[2]))), dims=[2])
    probs += torch.flip(torch.sigmoid(model(torch.flip(img_tensor, dims=[3]))), dims=[3])
    probs += torch.flip(torch.sigmoid(model(torch.flip(img_tensor, dims=[4]))), dims=[4])
    return (probs / 4.0).cpu().numpy().squeeze()


def advanced_refine(prob_map, threshold=0.5, constrain_mask=None, min_volume=0):
    mask = (prob_map > threshold).astype(np.uint8)

    if constrain_mask is not None:
        mask = np.where(constrain_mask > 0, mask, 0)

    if mask.sum() < min_volume:
        return np.zeros_like(mask)

    labels = measure.label(mask)
    if labels.max() == 0: return mask

    regions = measure.regionprops(labels)
    regions.sort(key=lambda x: x.area, reverse=True)
    largest_reg = regions[0]

    new_mask = np.zeros_like(mask)
    new_mask[labels == largest_reg.label] = 1

    main_centroid = np.array(largest_reg.centroid)
    for reg in regions[1:]:
        if reg.area > min_volume:
            dist = np.linalg.norm(np.array(reg.centroid) - main_centroid)
            if dist < 50:
                new_mask[labels == reg.label] = 1

    struct = generate_binary_structure(3, 1)
    new_mask = binary_closing(new_mask, structure=struct).astype(np.uint8)

    return new_mask


# ==========================================
# 5. 主控流水线
# ==========================================
def run_all():
    # 🌟 强制精准定位数据目录
    DATA_DIR = Path("data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
    if not DATA_DIR.exists():
        DATA_DIR = Path("data/BraTS2020_TrainingData")

    BEST_MODEL = "./saved_models/nirvana_hybrid_best.pth"  # 沿用你已有的满级权重
    SPLIT_RECORD = "./saved_models/test_split.txt"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR = "./logs"
    TB_DIR = f"./runs/brats_nirvana_final_{timestamp}"

    os.makedirs("./saved_models", exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StableHybrid3DUNet().to(device)
    criterion = DiceFocalBoundaryLoss(weights=[1.0, 1.5, 3.0], boundary_weight=0.1).to(device)

    print(f"👉 正在扫描数据目录: {DATA_DIR.absolute()}")
    if not DATA_DIR.exists():
        print(f"❌ 找不到数据目录，请检查路径！")
        return

    all_cases = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    if len(all_cases) == 0:
        print("❌ 目录内无病人文件夹！")
        return

    if not os.path.exists(BEST_MODEL):
        print("\n🚨 准备开启 [Res-SE + 解剖学约束] 的巅峰突破...")

        random.shuffle(all_cases)
        split_idx = int(len(all_cases) * 0.8)
        train_cases, test_cases = all_cases[:split_idx], all_cases[split_idx:]

        with open(SPLIT_RECORD, 'w') as f:
            for case in test_cases: f.write(case.name + '\n')

        train_loader = DataLoader(BraTS2020Dataset(train_cases, phase='train'), batch_size=2, shuffle=True,
                                  num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(BraTS2020Dataset(test_cases, phase='val'), batch_size=1)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scaler = GradScaler('cuda')
        best_mean_dice = 0.0

        tb_writer = SummaryWriter(log_dir=TB_DIR)
        train_csv_path = f"{LOG_DIR}/train_log_{timestamp}.csv"

        os.makedirs(LOG_DIR, exist_ok=True)
        with open(train_csv_path, mode='w', newline='') as f:
            csv.writer(f).writerow(['Epoch', 'Train_Loss', 'Val_WT_Dice'])

        for epoch in range(1, 201):
            model.train()
            epoch_loss = 0.0
            train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/200]")
            for imgs, masks, _ in train_pbar:
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                with autocast('cuda'):
                    loss = criterion(model(imgs), masks)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_train_loss = epoch_loss / len(train_loader)
            tb_writer.add_scalar('Loss/Train', avg_train_loss, epoch)

            current_val_dice = "N/A"
            if epoch % 10 == 0:
                model.eval();
                val_dices = []
                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc=f"🔍 Epoch {epoch} 验证中", leave=False)
                    for v_imgs, v_masks, _ in val_pbar:
                        logits = model(v_imgs.to(device)).float()
                        v_prob = torch.sigmoid(logits).cpu().numpy().squeeze()
                        v_gt = v_masks.numpy().squeeze()
                        d = 2. * ((v_prob[0] > 0.5) * v_gt[0]).sum() / ((v_prob[0] > 0.5).sum() + v_gt[0].sum() + 1e-5)
                        val_dices.append(d)

                current_mean = np.mean(val_dices)
                current_val_dice = f"{current_mean:.4f}"
                tb_writer.add_scalar('Dice/Val_WT', current_mean, epoch)

                tqdm.write(f"✅ Epoch {epoch} | Loss: {avg_train_loss:.4f} | Val WT Dice: {current_mean:.4f}")
                if current_mean > best_mean_dice:
                    best_mean_dice = current_mean
                    os.makedirs("./saved_models", exist_ok=True)
                    torch.save(model.state_dict(), BEST_MODEL)
                    tqdm.write("🌟 已保存涅槃版最优权重！")

            os.makedirs(LOG_DIR, exist_ok=True)
            with open(train_csv_path, mode='a', newline='') as f:
                csv.writer(f).writerow([epoch, f"{avg_train_loss:.4f}", current_val_dice])

        tb_writer.close()
    else:
        print(f"✅ 检测到历史最优权重，直接跳过训练，执行纯粹的黄金阈值评估！")

    # --- B. 执行评估 ---
    print("\n🚀 启动终极解剖学级联评估 (搭载黄金阈值)...")
    model.load_state_dict(torch.load(BEST_MODEL, map_location=device))
    model.eval()

    if os.path.exists(SPLIT_RECORD):
        with open(SPLIT_RECORD, 'r') as f:
            valid_test_names = set([line.strip() for line in f.readlines()])
        test_cases = [case for case in all_cases if case.name in valid_test_names]
    else:
        test_cases = all_cases

    test_loader = DataLoader(BraTS2020Dataset(test_cases, phase='test'), batch_size=1)
    results = {'WT': [], 'TC': [], 'ET': []}
    bad_cases = []

    test_csv_path = f"{LOG_DIR}/test_results_nirvana_final_pure_{timestamp}.csv"
    csv_rows = [['Case_ID', 'WT_Dice', 'WT_HD95', 'TC_Dice', 'TC_HD95', 'ET_Dice', 'ET_HD95']]

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating with TTA")
        for i, (imgs, masks, _) in enumerate(pbar):
            case_id = test_cases[i].name
            imgs = imgs.to(device)

            prob = predict_with_tta(model, imgs)
            gt = masks.numpy().squeeze()

            # ======================================================
            # 🏆 纯粹的黄金阈值级联约束 (撤销了激进的误杀规则)
            # ======================================================
            wt_p = advanced_refine(prob[0], threshold=0.50, min_volume=200)
            tc_p = advanced_refine(prob[1], threshold=0.55, constrain_mask=wt_p, min_volume=100)
            et_p = advanced_refine(prob[2], threshold=0.70, constrain_mask=tc_p, min_volume=50)

            preds = [wt_p, tc_p, et_p]
            case_metrics = []

            for c, key in enumerate(['WT', 'TC', 'ET']):
                p, g = preds[c], gt[c]
                if p.sum() > 0 and g.sum() > 0:
                    d, h = metric.binary.dc(p, g), metric.binary.hd95(p, g)
                else:
                    d, h = (1.0, 0.0) if p.sum() == 0 and g.sum() == 0 else (0.0, 373.0)

                results[key].append([d, h])
                case_metrics.extend([f"{d:.4f}", f"{h:.2f}"])

                if key == 'ET' and h > 15: bad_cases.append((case_id, h))

            csv_rows.append([case_id] + case_metrics)
            pbar.set_postfix(WT_HD95=f"{results['WT'][-1][1]:.2f}", ET_HD95=f"{results['ET'][-1][1]:.2f}")

    avg_row = ['AVERAGE']
    for k in ['WT', 'TC', 'ET']:
        arr = np.array(results[k])
        avg_row.extend([f"{np.mean(arr[:, 0]):.4f}", f"{np.mean(arr[:, 1]):.2f}"])
    csv_rows.append(avg_row)

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(test_csv_path, mode='w', newline='') as f:
        csv.writer(f).writerows(csv_rows)

    print("\n" + "=" * 50)
    print("🎯 Res-SE 涅槃版 终极评估报告 (黄金阈值锁死版)")
    for k in ['WT', 'TC', 'ET']:
        arr = np.array(results[k])
        print(f"{k:<3} -> Dice: {np.mean(arr[:, 0]):.4f} | HD95: {np.mean(arr[:, 1]):.2f}")
    print("=" * 50)

    if bad_cases:
        print("\n🚨 异常病例残余追踪 (ET HD95 > 15):")
        for cid, hval in bad_cases: print(f"- {cid}: {hval:.2f}")


if __name__ == "__main__":
    run_all()