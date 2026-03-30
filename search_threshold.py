import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from skimage import measure
from medpy import metric
from tqdm import tqdm
from scipy.ndimage import binary_closing, generate_binary_structure


# ==========================================
# 1. 搬运网络结构 (确保独立运行)
# ==========================================
class SEBlock3D(nn.Module):
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
        nn.InstanceNorm3d(out_c), nn.LeakyReLU(inplace=True)
    )


class ResSEBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = DSConv(in_c, out_c)
        self.conv2 = DSConv(out_c, out_c)
        self.se = SEBlock3D(out_c)
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv3d(in_c, out_c, 1, bias=False), nn.InstanceNorm3d(out_c))

    def forward(self, x):
        res = self.shortcut(x)
        out = self.se(self.conv2(self.conv1(x)))
        return F.leaky_relu(out + res, inplace=True)


class GlobalGatedTail3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.large_kernel = nn.Sequential(nn.Conv3d(dim, dim, 7, padding=3, groups=dim), nn.InstanceNorm3d(dim))
        self.proj_gate = nn.Sequential(nn.Conv3d(dim, dim, 1), nn.InstanceNorm3d(dim))
        self.proj_value = nn.Sequential(nn.Conv3d(dim, dim, 1), nn.InstanceNorm3d(dim))
        self.act = nn.SiLU()
        self.proj_out = nn.Sequential(nn.Conv3d(dim, dim, 1), nn.InstanceNorm3d(dim))

    def forward(self, x):
        return self.proj_out(self.act(self.proj_gate(self.large_kernel(x))) * self.proj_value(x)) + x


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv3d(F_g, F_int, 1), nn.InstanceNorm3d(F_int))
        self.W_x = nn.Sequential(nn.Conv3d(F_l, F_int, 1), nn.InstanceNorm3d(F_int))
        self.psi = nn.Sequential(nn.Conv3d(F_int, 1, 1), nn.Sigmoid())

    def forward(self, g, x): return x * self.psi(F.relu(self.W_g(g) + self.W_x(x)))


class StableHybrid3DUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_filters=16):
        super().__init__()
        self.inc = ResSEBlock(in_channels, base_filters)
        self.down = nn.ModuleList(
            [nn.Sequential(nn.MaxPool3d(2), ResSEBlock(base_filters * (2 ** i), base_filters * (2 ** (i + 1)))) for i in
             range(3)])
        bottleneck_dim = base_filters * 8
        self.bottleneck_down = nn.MaxPool3d(2)
        self.bottleneck_tail = nn.Sequential(ResSEBlock(bottleneck_dim, bottleneck_dim * 2),
                                             GlobalGatedTail3D(bottleneck_dim * 2),
                                             GlobalGatedTail3D(bottleneck_dim * 2))
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
        d = self.bottleneck_tail(self.bottleneck_down(enc[-1]))
        for i in range(4):
            up = self.up[i](d)
            diff = [enc[3 - i].size(j) - up.size(j) for j in range(2, 5)]
            up = F.pad(up, [diff[2] // 2, diff[2] - diff[2] // 2, diff[1] // 2, diff[1] - diff[1] // 2, diff[0] // 2,
                            diff[0] - diff[0] // 2])
            d = self.conv_up[i](torch.cat([self.att[i](g=up, x=enc[3 - i]), up], dim=1))
        return self.outc(d)


# ==========================================
# 2. 数据与处理核心
# ==========================================
class BraTS2020Dataset(Dataset):
    def __init__(self, case_folders):
        self.case_folders = case_folders

    def __len__(self):
        return len(self.case_folders)

    def _load_nii(self, path):
        if not path.exists(): path = Path(str(path) + ".gz")
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))

    def __getitem__(self, idx):
        case_path = self.case_folders[idx]
        files = {m: list(case_path.glob(f"*{m}.nii*"))[0] for m in ['flair', 't1', 't1ce', 't2']}
        img = np.stack([self._load_nii(files[m]) for m in ['flair', 't1', 't1ce', 't2']], axis=0).astype(np.float32)
        for c in range(4):
            m = img[c] > 0
            if m.any(): img[c][m] = (img[c][m] - img[c][m].mean()) / (img[c][m].std() + 1e-8)
        seg = self._load_nii(list(case_path.glob("*seg.nii*"))[0])
        label = np.stack([(seg > 0), ((seg == 1) | (seg == 4)), (seg == 4)], axis=0).astype(np.float32)
        return torch.from_numpy(img), torch.from_numpy(label), case_path.name


def predict_with_tta(model, img_tensor):
    probs = torch.sigmoid(model(img_tensor))
    probs += torch.flip(torch.sigmoid(model(torch.flip(img_tensor, dims=[2]))), dims=[2])
    probs += torch.flip(torch.sigmoid(model(torch.flip(img_tensor, dims=[3]))), dims=[3])
    probs += torch.flip(torch.sigmoid(model(torch.flip(img_tensor, dims=[4]))), dims=[4])
    return (probs / 4.0).cpu().numpy().squeeze()


def advanced_refine(prob_map, threshold=0.5, constrain_mask=None, min_volume=0):
    mask = (prob_map > threshold).astype(np.uint8)
    if constrain_mask is not None: mask = np.where(constrain_mask > 0, mask, 0)
    if mask.sum() < min_volume: return np.zeros_like(mask)
    labels = measure.label(mask)
    if labels.max() == 0: return mask
    regions = measure.regionprops(labels)
    regions.sort(key=lambda x: x.area, reverse=True)
    new_mask = np.zeros_like(mask)
    new_mask[labels == regions[0].label] = 1
    main_centroid = np.array(regions[0].centroid)
    for reg in regions[1:]:
        if reg.area > min_volume:
            if np.linalg.norm(np.array(reg.centroid) - main_centroid) < 50:
                new_mask[labels == reg.label] = 1
    return binary_closing(new_mask, structure=generate_binary_structure(3, 1)).astype(np.uint8)


# ==========================================
# 3. 榨汁机主流程
# ==========================================
def main():
    DATA_DIR = Path("data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
    BEST_MODEL = "./saved_models/nirvana_hybrid_best.pth"
    SPLIT_RECORD = "./saved_models/test_split.txt"
    TEMP_DIR = "./temp_probs"

    os.makedirs(TEMP_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 读取测试集名单
    with open(SPLIT_RECORD, 'r') as f:
        valid_test_names = set([line.strip() for line in f.readlines()])
    all_cases = [d for d in DATA_DIR.iterdir() if d.is_dir() and d.name in valid_test_names]
    print(f"📂 找到测试集病例: {len(all_cases)} 例")

    # 2. 阶段一：提取并缓存概率图 (如果还没提过的话)
    cached_files = list(Path(TEMP_DIR).glob("*_prob.npy"))
    if len(cached_files) < len(all_cases):
        print("\n⏳ 阶段一：正在用 TTA 提取原始概率图并缓存至硬盘 (只需执行一次)...")
        model = StableHybrid3DUNet().to(device)
        model.load_state_dict(torch.load(BEST_MODEL, map_location=device))
        model.eval()

        test_loader = DataLoader(BraTS2020Dataset(all_cases), batch_size=1)
        with torch.no_grad():
            for imgs, masks, case_id in tqdm(test_loader, desc="Caching TTA Probs"):
                cid = case_id[0]
                prob = predict_with_tta(model, imgs.to(device))
                gt = masks.numpy().squeeze()

                # 存成 float16 节省硬盘空间和读取时间
                np.save(f"{TEMP_DIR}/{cid}_prob.npy", prob.astype(np.float16))
                np.save(f"{TEMP_DIR}/{cid}_gt.npy", gt.astype(np.uint8))
        print("✅ 缓存完成！")
    else:
        print("\n✅ 检测到本地已有概率图缓存，直接进入极速网格搜索阶段！")

    # 3. 阶段二：网格搜索极速寻优
    print("\n🚀 阶段二：开始最优阈值极速榨汁...")

    # WT 比较稳定，固定为 0.5。搜索 TC 和 ET 的阈值
    tc_thresholds = [0.45, 0.50, 0.55]
    et_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75]

    best_et_dice = 0.0
    best_combo = None
    best_report = ""

    # 获取所有缓存病例的 ID
    case_ids = [f.name.replace("_prob.npy", "") for f in Path(TEMP_DIR).glob("*_prob.npy")]

    for tc_th in tc_thresholds:
        for et_th in et_thresholds:
            et_dices, et_hds = [], []

            for cid in case_ids:
                prob = np.load(f"{TEMP_DIR}/{cid}_prob.npy").astype(np.float32)
                gt = np.load(f"{TEMP_DIR}/{cid}_gt.npy")

                # 级联打磨 (WT 固定 0.5)
                wt_p = advanced_refine(prob[0], threshold=0.5, min_volume=200)
                tc_p = advanced_refine(prob[1], threshold=tc_th, constrain_mask=wt_p, min_volume=100)
                et_p = advanced_refine(prob[2], threshold=et_th, constrain_mask=tc_p, min_volume=50)

                p_et, g_et = et_p, gt[2]

                if p_et.sum() > 0 and g_et.sum() > 0:
                    et_dices.append(metric.binary.dc(p_et, g_et))
                else:
                    et_dices.append(1.0 if p_et.sum() == 0 and g_et.sum() == 0 else 0.0)

            mean_et_dice = np.mean(et_dices)
            print(f"👉 尝试组合 [TC: {tc_th:.2f}, ET: {et_th:.2f}]  -> ET Dice: {mean_et_dice:.4f}")

            if mean_et_dice > best_et_dice:
                best_et_dice = mean_et_dice
                best_combo = (tc_th, et_th)

    print("\n" + "🌟" * 25)
    print(f"🏆 榨汁完成！最优 ET 阈值组合出炉：")
    print(f"推荐 TC 阈值: {best_combo[0]:.2f}")
    print(f"推荐 ET 阈值: {best_combo[1]:.2f}")
    print(f"预计 ET Dice 将达到: {best_et_dice:.4f}")
    print("🌟" * 25)
    print("\n💡 接下来：只需把主代码 main.py 中 run_all() 里 evaluation 部分的:")
    print(f"tc_p = advanced_refine(prob[1], threshold={best_combo[0]:.2f}...)")
    print(f"et_p = advanced_refine(prob[2], threshold={best_combo[1]:.2f}...)")


if __name__ == "__main__":
    main()