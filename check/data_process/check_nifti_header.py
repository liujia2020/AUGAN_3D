import nibabel as nib
import numpy as np
import os

# [!!] 请将此路径修改为您要检查的 NIfTI 文件
FILE_PATH = "project_assets/Ultrasound_NIfTI_Dataset_Z185/train_hq/Sim_0001_hq.nii"

def inspect_nifti_header(file_path):
    print(f"--- 正在检查 NIfTI 头文件: {file_path} ---")
    
    if not os.path.exists(file_path):
        print(f"[错误] 文件未找到: {file_path}")
        return

    try:
        # 1. 加载 NIfTI 文件
        nii_file = nib.load(file_path)

        # 2. 获取头文件 (header)
        header = nii_file.header
        
        # 3. 获取数据形状 (这对应 3D Slicer 的 Image Dimensions)
        shape = nii_file.shape
        print(f"\n[数据形状] (Data Array Shape): {shape}")
        
        # 4. 获取数据类型
        dtype = nii_file.get_data_dtype()
        print(f"[数据类型] (Data Type): {dtype}")

        # 5. 获取仿射矩阵 (Affine Matrix)
        # 这是存储方向和间距的“真相”
        affine = nii_file.affine
        print(f"\n[仿射矩阵] (Affine Matrix):\n{affine}")

        # 6. 从头文件中提取 "Zooms" (即 3D Slicer 的 Image Spacing)
        # get_zooms() 会读取头文件中的 pixdim 字段
        spacing = header.get_zooms()
        print(f"\n[!!! 关键信息 !!!]")
        print(f"[头文件间距] (Header Spacing / Zooms): ({spacing[0]:.4f}, {spacing[1]:.4f}, {spacing[2]:.4f}) mm")
        
        if (spacing[0] == 1.0 and spacing[1] == 1.0 and spacing[2] == 1.0):
            print("[诊断]: 间距为 (1, 1, 1)。这强烈表明元数据是默认值，而非真实物理间距。")

        # 7. 检查数据范围 (这对应 3D Slicer 的 Scalar Range)
        # 为避免加载大文件，我们只在小文件上这样做
        # (您的文件很小，所以这没问题)
        print("\n[正在计算数据范围...]")
        data_array = nii_file.get_fdata()
        min_val = np.min(data_array)
        max_val = np.max(data_array)
        mean_val = np.mean(data_array)
        
        print(f"[数据范围] (Scalar Range): Min={min_val:.4f}, Max={max_val:.4f}")
        print(f"[数据均值] (Scalar Mean): {mean_val:.4f}")
        print("\n--- 检查完毕 ---")

    except Exception as e:
        print(f"[错误] 处理文件时出错: {e}")

if __name__ == "__main__":
    inspect_nifti_header(FILE_PATH)