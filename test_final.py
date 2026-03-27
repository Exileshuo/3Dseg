import os
import torch
import numpy as np
import SimpleITK as sitk
from skimage import measure  # 用于后处理连通域统计
from torch.utils.data import DataLoader
from model import Lite3DUNet_Attn as Model

# 配置
TEST_DATA_DIR = "data/BraTS2020_Split/my_test_set"
MODEL_PATH = "./saved_models/lite_attn_epoch_100.pth"
PRED_DIR = "./test_results/refined_predictions"
MIN_VOLUME = 50  # 剔除少于 50 像素的小孤岛

os.makedirs(PRED_DIR, exist_ok=True)


def remove_small_objects(mask, min_size):
    """后处理：移除 3D 掩码中的微小干扰块"""
    labels = measure.label(mask)
    new_mask = np.copy(mask)
    for region in measure.regionprops(labels):
        if region.area < min_size:
            new_mask[labels == region.label] = 0
    return new_mask


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载数据集
    from brats_exp.dataset import BraTS2020Dataset
    dataset = BraTS2020Dataset(data_dir=TEST_DATA_DIR, phase='test')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = Model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    with torch.no_grad():
        for i, (images, labels, _) in enumerate(loader):
            case_id = dataset.case_folders[i].name
            images = images.to(device)

            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().squeeze()  # [3, D, H, W]

            # 1. 执行后处理去噪 (针对每个通道)
            refined_preds = np.zeros_like(preds)
            for c in range(3):
                refined_preds[c] = remove_small_objects(preds[c], MIN_VOLUME)

            # 2. 还原 BraTS 标签 (WT->2, TC->1, ET->4)
            final_mask = np.zeros(refined_preds.shape[1:], dtype=np.uint8)
            final_mask[refined_preds[0] == 1] = 2  # WT
            final_mask[refined_preds[1] == 1] = 1  # TC覆盖
            final_mask[refined_preds[2] == 1] = 4  # ET覆盖

            # 3. 保存结果
            ref_path = dataset.case_folders[i] / f"{case_id}_t1.nii"
            ref_itk = sitk.ReadImage(str(ref_path))
            out_itk = sitk.GetImageFromArray(final_mask)
            out_itk.CopyInformation(ref_itk)
            sitk.WriteImage(out_itk, os.path.join(PRED_DIR, f"{case_id}_pred.nii.gz"))
            print(f"✅ Case {case_id} processed with post-processing.")


if __name__ == "__main__":
    main()