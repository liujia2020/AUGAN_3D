# import torch
# import logging
# import numpy as np
# import scipy.io as sio
# from torch.utils.data import Dataset
# from cubdl_master.example_picmus_torch import dispaly_img
# def load_dataset(opt, phase, test_type):
#     """
#     Loads ultrasound image data from .mat files, processes it, and creates a PyTorch Dataset.
#     """
#     # Define dataset names based on test_type
#     dataset_names = []
#     if test_type == 0:  # train mode
#         dataset_names = [
#             "simulation_resolution_distorsion_iq", 
#             "simulation_contrast_speckle_iq",
#             "experiments_resolution_distorsion_iq", 
#             "experiments_contrast_speckle_iq", 
#             "experiments_carotid_long_iq", 
#             "experiments_carotid_cross_iq"
#         ]
#     # --- 添加缺失的 elif 分支 ---
#     elif test_type == 1:
#         dataset_names = ["simulation_resolution_distorsion_iq"]
#     elif test_type == 2:
#         dataset_names = ["simulation_contrast_speckle_iq"]
#     elif test_type == 3:
#         dataset_names = ["experiments_resolution_distorsion_iq"]
#     elif test_type == 4:
#         dataset_names = ["experiments_contrast_speckle_iq"]
#     elif test_type == 5:
#         dataset_names = ["experiments_carotid_long_iq", "experiments_carotid_cross_iq"]
#     elif test_type == 6: # all datasets for a comprehensive test
#         dataset_names = [
#             "simulation_resolution_distorsion_iq", "simulation_contrast_speckle_iq",
#             "experiments_resolution_distorsion_iq", "experiments_contrast_speckle_iq",
#             "experiments_carotid_long_iq", "experiments_carotid_cross_iq"
#         ]
#     # --- 添加结束 ---

#     all_single_angle_images_list = [] # 存放所有低质量输入图像 -->单角度
#     all_compound_images_list = [] # 存放所有高质量输入图片 -->多角度
    
#     # Define target dimensions for the network input 
#     # 原始图像是(508,387)
#     target_height = 512
#     target_width = 384

#     if opt.load:
#         # ... (循环加载文件的代码保持不变) ...
#         for dataset_name in dataset_names:
#             phase_suffix = '_train' if phase == 'train' else '_test'
#             mat_file_path = f'./img_data1/{dataset_name}{phase_suffix}.mat'
#             data_key_in_mat = f'{dataset_name}_data'

#             try:
#                 mat_data_dict = sio.loadmat(mat_file_path)
#             except FileNotFoundError:
#                 logging.warning(f'File not found, skipping: {mat_file_path}')
#                 continue
            
#             raw_images = mat_data_dict[data_key_in_mat]
            
#             single_angle_images = raw_images[:-1, :, :] # 除了最后一张，都是单角度的图片
#             compound_image = raw_images[-1, :, :][np.newaxis, ...] # 最后一张是高质量图片，保持维度(1, H, w)

#             # --- Correctly Crop and Pad the images ---
#             single_angle_cropped = single_angle_images[:, :, :target_width]
#             compound_cropped = compound_image[:, :, :target_width]

#             padded_single = np.zeros((single_angle_cropped.shape[0], target_height, target_width))
#             padded_compound = np.zeros((compound_cropped.shape[0], target_height, target_width))
            
#             h_orig = single_angle_cropped.shape[1] # Original height (508)
#             padded_single[:, :h_orig, :] = single_angle_cropped
#             padded_compound[:, :h_orig, :] = compound_cropped
            
#             all_single_angle_images_list.append(padded_single)
#             all_compound_images_list.append(padded_compound)
            
#             logging.info(f'Dataset loaded and processed: {dataset_name}')

#     # --- 这里添加一个检查，防止列表为空 ---
#     if not all_single_angle_images_list:
#         logging.error(f"No datasets loaded for test_type={test_type} and phase='{phase}'. Check file paths and test_type setting.")
#         # 返回一个空的 Dataset 或引发错误，取决于你希望如何处理
#         # return None # 或者 raise ValueError("No data loaded")
#         # 为了能继续运行，我们暂时返回None，但你应该检查文件是否存在
#         return AugmentedImageDataset(np.array([]), np.array([])) # 返回空数据集避免后续错误
        
#     logging.info('All datasets loaded successfully.')

#     # Combine all loaded data...
#     final_single_angle_images = np.concatenate(all_single_angle_images_list, axis=0)
#     final_compound_images = np.concatenate(all_compound_images_list, axis=0)

#     num_datasets = len(all_compound_images_list)
#     # --- 添加保护，防止除以零 ---
#     if num_datasets == 0:
#         num_angles_per_dataset = 0
#     else:
#         num_angles_per_dataset = final_single_angle_images.shape[0] // num_datasets
    
#     repeated_compound_images = np.repeat(final_compound_images, num_angles_per_dataset, axis=0)

#     final_single_angle_images = final_single_angle_images[:, np.newaxis, :, :]
#     repeated_compound_images = repeated_compound_images[:, np.newaxis, :, :]

#     image_dataset = AugmentedImageDataset(
#         low_quality_images=final_single_angle_images,
#         high_quality_images=repeated_compound_images,
#     )
#     return image_dataset

# class AugmentedImageDataset(Dataset):
#     """
#     A custom PyTorch Dataset class that takes low and high quality images,
#     applies data augmentation to the low quality images, and returns image pairs.
#     """
#     def __init__(self, low_quality_images, high_quality_images):
#         super(AugmentedImageDataset, self).__init__()

#         base_images_tensor = torch.from_numpy(low_quality_images).float()
#         target_images_tensor = torch.from_numpy(high_quality_images).float()
        
#         def add_gaussian_noise(images_tensor):
#             """Adds Gaussian noise to a tensor of images."""
#             noise = torch.randn_like(images_tensor) * 0.01 # A small amount of noise
#             return images_tensor + noise

#         # --- Data Augmentation ---
#         images_with_noise = add_gaussian_noise(base_images_tensor)
        
#         # Flip the images
#         vertically_flipped_images = torch.flip(base_images_tensor, [2])
#         horizontally_flipped_images = torch.flip(base_images_tensor, [3])
        
#         # Combine original and augmented images
#         self.augmented_input_images = torch.cat([
#             base_images_tensor, 
#             images_with_noise, 
#             horizontally_flipped_images, 
#             vertically_flipped_images
#         ])
        
#         # Repeat target images to match the number of augmented input images
#         self.target_images = target_images_tensor.repeat(4, 1, 1, 1)

#         self.len = self.augmented_input_images.shape[0]



#     def __getitem__(self, index):
#         return self.augmented_input_images[index], self.target_images[index]

#     def __len__(self):
#         return self.len


# """  Display the original image, reconstructed image and target image"""
# # def test_image(data, data1, target, xlims, zlims, i, phase, name):
# """ 显示原始图像、重建图像和目标图像 """
# def test_image(low_quality_image, generated_image, high_quality_image, xlims, zlims, i, phase, name):
#     """
#     显示并保存输入图像、生成图像和目标图像的对比图。
#     """
#     # --- 内部变量名修改开始 ---
    
#     # 处理 low_quality_image (之前的 data)
#     lq_image_np = low_quality_image.detach().numpy()
#     lq_image_np = np.squeeze(lq_image_np)
#     lq_image_np -= np.max(lq_image_np) # 归一化处理

#     # 处理 generated_image (之前的 data1)
#     gen_image_np = generated_image.detach().numpy()
#     gen_image_np = np.squeeze(gen_image_np)
#     gen_image_np -= np.max(gen_image_np) # 归一化处理

#     # 处理 high_quality_image (之前的 target)
#     hq_image_np = high_quality_image.detach().numpy()
#     hq_image_np = np.squeeze(hq_image_np)
#     hq_image_np -= np.max(hq_image_np) # 归一化处理

#     # 调用显示函数时，传入修改后的变量
#     dispaly_img(lq_image_np, gen_image_np, hq_image_np, xlims, zlims, [1], i, phase, name)

import os # <-- 确保导入了 os 模块
import torch
import logging
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from cubdl_master.example_picmus_torch import dispaly_img

# --- load_dataset 函数 ---
def load_dataset(opt, phase, test_type):
    """
    Loads ultrasound image data from .mat files, processes it, and creates a PyTorch Dataset.
    """
    # ... (dataset_names 的定义保持不变) ...
    dataset_names = []
    if test_type == 0: # train mode
        dataset_names = [
            "simulation_resolution_distorsion_iq",
            "simulation_contrast_speckle_iq",
            "experiments_resolution_distorsion_iq",
            "experiments_contrast_speckle_iq",
            "experiments_carotid_long_iq",
            "experiments_carotid_cross_iq"
        ]
    elif test_type == 1: dataset_names = ["simulation_resolution_distorsion_iq"]
    elif test_type == 2: dataset_names = ["simulation_contrast_speckle_iq"]
    elif test_type == 3: dataset_names = ["experiments_resolution_distorsion_iq"]
    elif test_type == 4: dataset_names = ["experiments_contrast_speckle_iq"]
    elif test_type == 5: dataset_names = ["experiments_carotid_long_iq", "experiments_carotid_cross_iq"]
    elif test_type == 6: dataset_names = ["simulation_resolution_distorsion_iq", "simulation_contrast_speckle_iq","experiments_resolution_distorsion_iq", "experiments_contrast_speckle_iq", "experiments_carotid_long_iq", "experiments_carotid_cross_iq"]


    all_single_angle_images_list = []
    all_compound_images_list = []

    target_height = 512
    target_width = 384

    # --- 检查 opt.dataroot 是否存在 ---
    # !! 注意: 确保你的 opt 对象里确实有 dataroot 这个属性 !!
    # 这个属性是在 options/base_options.py 里添加的
    if not hasattr(opt, 'dataroot') or not opt.dataroot:
        # 如果没有 dataroot，则退回到使用旧路径，或者报错
        # 为了兼容你之前的代码，我们先尝试退回旧路径并给出警告
        logging.warning("Option --dataroot not found or empty. Falling back to default './project_assets'. Make sure options/base_options.py is updated.")
        # 使用默认的新基础路径 (如果你还没改 base_options.py，这里也会找不到)
        # 或者直接硬编码新路径 (不推荐)
        # data_root_dir = './project_assets' # 强制使用新路径
        # 或者尝试从 opt.checkpoints_dir 推断 (如果 checkpionts_dir 已更新)
        # data_root_dir = os.path.dirname(os.path.dirname(opt.checkpoints_dir)) # 即 project_assets 目录
        # 最稳妥：如果 opt.dataroot 不存在，就用你之前移动到的固定路径
        data_root_dir = './project_assets' # 假设你把它放在这里了
        if not os.path.isdir(data_root_dir):
             logging.error(f"Default data root directory '{data_root_dir}' not found. Please specify --dataroot or check directory structure.")
             return AugmentedImageDataset(np.array([]), np.array([])) # 返回空数据集
    else:
        data_root_dir = opt.dataroot # 使用命令行传入或默认的 dataroot
    # --- dataroot 检查结束 ---


    if opt.load: # opt.load 这个参数似乎与加载 .mat 文件无关，保留原始逻辑
        for dataset_name in dataset_names:
            phase_suffix = '_train' if phase == 'train' else '_test'

            # --- 核心修改：确保使用 data_root_dir 和新文件夹名 'datasets_mat' ---
            # 1. 构建 .mat 文件所在的正确目录路径
            mat_dir = os.path.join(data_root_dir, 'datasets_mat') # 使用 data_root_dir 指向 project_assets
            # 2. 构建完整的文件路径
            mat_file_path = os.path.join(mat_dir, f'{dataset_name}{phase_suffix}.mat')
            # --- 修改结束 ---

            data_key_in_mat = f'{dataset_name}_data'

            try:
                mat_data_dict = sio.loadmat(mat_file_path)
            except FileNotFoundError:
                # --- 让警告信息显示正确的尝试路径 ---
                logging.warning(f'File not found, skipping: {mat_file_path}') # 显示新的路径
                # --- 修改结束 ---
                continue
            # ... (后续处理 raw_images, single_angle_images 等的代码保持不变) ...
            raw_images = mat_data_dict[data_key_in_mat]
            single_angle_images = raw_images[:-1, :, :]
            compound_image = raw_images[-1, :, :][np.newaxis, ...]
            single_angle_cropped = single_angle_images[:, :, :target_width]
            compound_cropped = compound_image[:, :, :target_width]
            padded_single = np.zeros((single_angle_cropped.shape[0], target_height, target_width))
            padded_compound = np.zeros((compound_cropped.shape[0], target_height, target_width))
            h_orig = single_angle_cropped.shape[1]
            padded_single[:, :h_orig, :] = single_angle_cropped
            padded_compound[:, :h_orig, :] = compound_cropped
            all_single_angle_images_list.append(padded_single)
            all_compound_images_list.append(padded_compound)
            logging.info(f'Dataset loaded and processed: {dataset_name}')


    # --- 处理未加载到文件的情况 ---
    if not all_single_angle_images_list:
        logging.error(f"No datasets loaded for test_type={test_type} and phase='{phase}'. Check file paths (e.g., {os.path.join(data_root_dir, 'datasets_mat')}) and test_type setting.")
        return AugmentedImageDataset(np.array([]), np.array([])) # 返回空数据集
    # --- 处理结束 ---

    logging.info('All datasets loaded successfully.')
    # ... (后续 np.concatenate, np.repeat, 创建 AugmentedImageDataset 实例的代码保持不变) ...
    final_single_angle_images = np.concatenate(all_single_angle_images_list, axis=0)
    final_compound_images = np.concatenate(all_compound_images_list, axis=0)
    num_datasets = len(all_compound_images_list)
    if num_datasets == 0: num_angles_per_dataset = 0
    else: num_angles_per_dataset = final_single_angle_images.shape[0] // num_datasets
    repeated_compound_images = np.repeat(final_compound_images, num_angles_per_dataset, axis=0)
    final_single_angle_images = final_single_angle_images[:, np.newaxis, :, :]
    repeated_compound_images = repeated_compound_images[:, np.newaxis, :, :]
    image_dataset = AugmentedImageDataset(
        low_quality_images=final_single_angle_images,
        high_quality_images=repeated_compound_images,
    )
    return image_dataset


# --- AugmentedImageDataset 类 (保持不变, 但添加了空数组检查) ---
class AugmentedImageDataset(Dataset):
    """
    A custom PyTorch Dataset class that takes low and high quality images,
    applies data augmentation to the low quality images, and returns image pairs.
    """
    def __init__(self, low_quality_images, high_quality_images):
        super(AugmentedImageDataset, self).__init__()

        # --- 添加对空数组输入的处理 ---
        if low_quality_images.size == 0 or high_quality_images.size == 0:
            logging.warning("AugmentedImageDataset initialized with empty arrays.")
            self.augmented_input_images = torch.empty(0)
            self.target_images = torch.empty(0)
            self.len = 0
            return # 直接返回
        # --- 处理结束 ---

        base_images_tensor = torch.from_numpy(low_quality_images).float()
        target_images_tensor = torch.from_numpy(high_quality_images).float()

        def add_gaussian_noise(images_tensor):
            noise = torch.randn_like(images_tensor) * 0.01
            return images_tensor + noise

        images_with_noise = add_gaussian_noise(base_images_tensor)
        vertically_flipped_images = torch.flip(base_images_tensor, [2])
        horizontally_flipped_images = torch.flip(base_images_tensor, [3])

        self.augmented_input_images = torch.cat([
            base_images_tensor,
            images_with_noise,
            horizontally_flipped_images,
            vertically_flipped_images
        ])
        self.target_images = target_images_tensor.repeat(4, 1, 1, 1)
        self.len = self.augmented_input_images.shape[0]

    def __getitem__(self, index):
        if index >= self.len: raise IndexError("Index out of range")
        return self.augmented_input_images[index], self.target_images[index]

    def __len__(self):
        return self.len

# --- test_image 函数 (保持不变) ---
def test_image(low_quality_image, generated_image, high_quality_image, xlims, zlims, i, phase, name):
    lq_image_np = low_quality_image.detach().numpy(); lq_image_np = np.squeeze(lq_image_np); lq_image_np -= np.max(lq_image_np)
    gen_image_np = generated_image.detach().numpy(); gen_image_np = np.squeeze(gen_image_np); gen_image_np -= np.max(gen_image_np)
    hq_image_np = high_quality_image.detach().numpy(); hq_image_np = np.squeeze(hq_image_np); hq_image_np -= np.max(hq_image_np)
    dispaly_img(lq_image_np, gen_image_np, hq_image_np, xlims, zlims, [1], i, phase, name)