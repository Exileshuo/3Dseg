import os
import random
import numpy as np
import torch
import SimpleITK as sitk
from pathlib import Path
from torch.utils.data import Dataset


class BraTS2020Dataset(Dataset):
    def __init__(self, data_dir, phase='train', patch_size=(64, 64, 64)):
        self.data_dir = Path(data_dir)
        self.phase = phase
        self.patch_size = patch_size

        # 获取所有子文件夹 (不限制命名)
        all_folders = [f for f in self.data_dir.iterdir() if f.is_dir()]

        self.case_folders = []
        for folder in all_folders:
            case_id = folder.name
            is_valid = True

            # 基础的 4 个模态文件必须存在
            required_suffixes = ['t1', 't1ce', 't2', 'flair']
            if self.phase == 'train':
                required_suffixes.append('seg')  # 只有训练时强制要求必须有标签

            for suffix in required_suffixes:
                file_path = folder / f"{case_id}_{suffix}.nii"
                if not file_path.exists():
                    is_valid = False
                    break

            if is_valid:
                self.case_folders.append(folder)

        self.case_folders = sorted(self.case_folders, key=lambda x: x.name)
        print(f"✅ 成功从 {self.data_dir.name} 加载 {len(self.case_folders)} 个有效病例用于 {self.phase} 阶段。")

    def __len__(self):
        return len(self.case_folders)

    def _normalize(self, data):
        mask = data > 0
        if mask.sum() > 0:
            mean = data[mask].mean()
            std = data[mask].std()
            data = (data - mean) / (std + 1e-8)
            data[~mask] = 0
        return data

    def __getitem__(self, idx):
        case_folder = self.case_folders[idx]
        case_id = case_folder.name

        # 1. 加载图像并归一化
        images = []
        modalities = ['t1', 't1ce', 't2', 'flair']
        for mod in modalities:
            img_path = case_folder / f"{case_id}_{mod}.nii"
            itk_img = sitk.ReadImage(str(img_path))
            img_arr = sitk.GetArrayFromImage(itk_img).astype(np.float32)
            img_arr = self._normalize(img_arr)
            images.append(img_arr)

        image_stack = np.stack(images, axis=0)

        # 2. 动态加载标签 (应对验证集没有标签的情况)
        label_path = case_folder / f"{case_id}_seg.nii"
        if label_path.exists():
            itk_label = sitk.ReadImage(str(label_path))
            label_arr = sitk.GetArrayFromImage(itk_label).astype(np.uint8)
            has_label = True
        else:
            # 如果没有标签，生成一个全零的占位符
            label_arr = np.zeros_like(image_stack[0], dtype=np.uint8)
            has_label = False

        # 3. 训练期随机裁剪
        if self.phase == 'train':
            _, D, H, W = image_stack.shape
            pD, pH, pW = self.patch_size
            start_d = random.randint(0, max(0, D - pD))
            start_h = random.randint(0, max(0, H - pH))
            start_w = random.randint(0, max(0, W - pW))

            image_stack = image_stack[:, start_d:start_d + pD, start_h:start_h + pH, start_w:start_w + pW]
            label_arr = label_arr[start_d:start_d + pD, start_h:start_h + pH, start_w:start_w + pW]

        # 4. 标签转化为 3 通道 One-hot (WT, TC, ET)
        WT = (label_arr > 0).astype(np.float32)
        TC = ((label_arr == 1) | (label_arr == 4)).astype(np.float32)
        ET = (label_arr == 4).astype(np.float32)
        label_stack = np.stack([WT, TC, ET], axis=0)

        # 把 has_label 作为一个标志位传回，方便后续判断是否计算 Dice
        return torch.from_numpy(image_stack), torch.from_numpy(label_stack), has_label