import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path


def calculate_dice(pred, gt):
    """计算单个类别的 Dice 分数"""
    intersection = np.logical_and(pred, gt).sum()
    union = pred.sum() + gt.sum()
    if union == 0:
        return 1.0  # 如果真值和预测都为空，认为预测正确
    return 2.0 * intersection / (union + 1e-8)


def check_accuracy():
    # 🚨 确保这两个路径与你电脑上的实际路径一致
    test_dir = Path("data/BraTS2020_Split/my_test_set")
    pred_dir = Path("./test_results")

    if not pred_dir.exists():
        print(f"❌ 找不到预测结果目录: {pred_dir}")
        return

    results = {'WT': [], 'TC': [], 'ET': []}

    # 获取所有预测文件
    pred_files = list(pred_dir.glob("*_pred.nii.gz"))
    print(f"统计开始，共发现 {len(pred_files)} 个预测结果...\n")

    for pred_file in pred_files:
        case_id = pred_file.name.replace("_pred.nii.gz", "")

        # 1. 加载预测值 (0, 1, 2, 4)
        p_img = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_file)))

        # 2. 加载真值 (0, 1, 2, 4)
        gt_path = test_dir / case_id / f"{case_id}_seg.nii"
        if not gt_path.exists():
            gt_path = Path(str(gt_path) + ".gz")

        if not gt_path.exists():
            print(f"⚠️ 跳过 {case_id}: 找不到对应的真值标签文件。")
            continue

        g_img = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))

        # --- 3. 按照 BraTS 标准划分子区域 ---
        # WT (Whole Tumor): 标签 1, 2, 4 的并集
        p_wt = (p_img > 0);
        g_wt = (g_img > 0)
        # TC (Tumor Core): 标签 1, 4 的并集
        p_tc = np.logical_or(p_img == 1, p_img == 4);
        g_tc = np.logical_or(g_img == 1, g_img == 4)
        # ET (Enhancing Tumor): 仅标签 4
        p_et = (p_img == 4);
        g_et = (g_img == 4)

        # --- 4. 计算 Dice ---
        d_wt = calculate_dice(p_wt, g_wt)
        d_tc = calculate_dice(p_tc, g_tc)
        d_et = calculate_dice(p_et, g_et)

        results['WT'].append(d_wt)
        results['TC'].append(d_tc)
        results['ET'].append(d_et)

        print(f"[{case_id}] | WT: {d_wt:.4f} | TC: {d_tc:.4f} | ET: {d_et:.4f}")

    # --- 5. 打印汇总结果 ---
    if len(results['WT']) > 0:
        print("\n" + "=" * 40)
        print("🎯 测试集整体平均指标 (Mean Dice):")
        print(f"Average WT (Whole Tumor):     {np.mean(results['WT']):.4f}")
        print(f"Average TC (Tumor Core):     {np.mean(results['TC']):.4f}")
        print(f"Average ET (Enhancing Tumor): {np.mean(results['ET']):.4f}")
        print("=" * 40)
    else:
        print("😭 未能成功计算任何病例，请检查路径。")


# 🚨 注意：这一行必须顶格写，不能有任何空格！
if __name__ == "__main__":
    check_accuracy()