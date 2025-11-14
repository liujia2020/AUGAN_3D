import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time
import warnings # 导入 warnings 模块
import logging # <-- 确保导入 logging
import nibabel as nib # [!!] 改造点：添加 NIfTI 依赖
import torch.nn.functional as F # [!!] 改造点：添加 Pad 依赖

from options.test_options import TestOptions
from models import create_model
from data_process import load_dataset # [!!] 依赖 3D 改造版
from torch.utils.data import DataLoader
from metrics import image_evaluation
from cubdl_master.example_picmus_torch import load_datasets, create_network # [!!] 改造点：重新引入（用于指标）

warnings.filterwarnings("ignore", category=UserWarning)

def de_normalize(tensor_np_minus1_to_1):
    """[!!] 改造点：添加反归一化"""
    """将 [-1, 1] 范围的 NumPy 数组反归一化回 [-60, 0] dB 范围"""
    tensor_np_0_to_1 = (tensor_np_minus1_to_1 + 1.0) / 2.0
    # [!!] 战术修正：(x * 60) - 60 是错误的。
    # 应该是 x * (MAX - MIN) + MIN
    # (x * (0 - (-60))) + (-60) = (x * 60) - 60。
    # 不，这个逻辑是对的。
    return (tensor_np_0_to_1 * (0.0 - (-60.0))) + (-60.0)


if __name__ == '__main__':
    # --- 1. 初始化和加载设置 ---
    
    # [!!] 改造点：重新引入 plane_wave_data (用于 metrics.py)
    try:
        plane_wave_data = load_datasets("simulation", "resolution_distorsion", "iq")
        # [!!] 战术修正：我们不再需要 xlims, zlims
        # das, iqdata, xlims, zlims = create_network(plane_wave_data, [1])
    except Exception as e:
        print(f"Warning: PICMUS dataset (for metrics) not available. Error: {e}")
        plane_wave_data = None
        # xlims, zlims = [0, 1], [0, 1]

    opt = TestOptions().parse()
    opt.batch_size = 1 # [!!] 致命修正：测试时必须 B=1
    opt.serial_batches = True
    opt.no_flip = True
    test_type = opt.test_type
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # --- 2. 创建模型并加载权重 ---
    model = create_model(opt)
    model.setup(opt)
    if opt.eval: model.eval()
    image_evaluator = image_evaluation()

    # --- 3. 加载测试数据集 ---
    print(f"Loading test dataset for test_type = {test_type}...")
    # [!!] 依赖 3D 改造版：opt.phase='test' 将触发 __getitem__ 返回完整容积
    image_dataset = load_dataset(opt, opt.phase) 
    if len(image_dataset) == 0:
        print(f"Error: No data loaded for test_type={test_type} and phase='{opt.phase}'. Skipping test.")
        exit()
        
    test_loader = DataLoader(dataset=image_dataset, num_workers=0, batch_size=opt.batch_size, shuffle=False)
    test_bar = tqdm(test_loader, desc=f"Testing {opt.name} (Type {test_type})", unit="volume")
    image_counter = 1
    images_output_dir = os.path.join('./project_assets/output_visualizations', opt.name, 'test', f'type_{test_type}')
    os.makedirs(images_output_dir, exist_ok=True)
    print(f"Output NIfTI volumes will be saved to: {images_output_dir}")

    # [!!] 改造点：重写整个循环
    for lq_volume_tensor, hq_volume_tensor, lq_path_key in test_bar:

            if lq_volume_tensor.nelement() == 0 or lq_path_key[0] == "Error": 
                logging.warning(f"Skipping empty or error batch (from data_process.py).")
                continue # 跳过空数据

            # [!!] 战术修正：lq_path_key 是一个 list[str] (B=1)，取第一个
            lq_path = lq_path_key[0]
            if lq_path == "N/A": continue # (不应该发生，但作为保护)

            # [!!] 致命修正 1：滑动窗口 (OOM 修复)]
            # 废弃 model.test()。我们必须手动推理。
            patch_size = (64, 64, 64) # [!!] 必须与训练时相同
            B, C, D, H, W = lq_volume_tensor.shape # (B=1, C=1, D=185, H=128, W=128)
            
            # (在 CPU 上创建空的输出容积)
            gen_volume_tensor = torch.zeros_like(lq_volume_tensor)

            model.netG.eval() # 确保 G 处于评估模式
            with torch.no_grad():
                # (简化的非重叠滑动窗口)
                for d in range(0, D, patch_size[0]):
                    for h in range(0, H, patch_size[1]):
                        for w in range(0, W, patch_size[2]):
                            # 计算边界
                            d_end = min(d + patch_size[0], D)
                            h_end = min(h + patch_size[1], H)
                            w_end = min(w + patch_size[2], W)
                            
                            d_start, h_start, w_start = d, h, w

                            lq_patch = lq_volume_tensor[:, :, d_start:d_end, h_start:h_end, w_start:w_end]

                            # [!!] 战术修正：如果 patch 太小 (在边缘)，需要 Pad 到 64
                            pad_d = patch_size[0] - lq_patch.shape[2]
                            pad_h = patch_size[1] - lq_patch.shape[3]
                            pad_w = patch_size[2] - lq_patch.shape[4]

                            # (右侧和底部/后部填充)
                            lq_patch_padded = F.pad(lq_patch, (0, pad_w, 0, pad_h, 0, pad_d), mode='reflect') # 使用反射填充

                            # (在 GPU 上运行推理)
                            gen_patch_padded = model.netG(lq_patch_padded.to(device))

                            # (裁剪掉填充)
                            gen_patch = gen_patch_padded[:, :, :lq_patch.shape[2], :lq_patch.shape[3], :lq_patch.shape[4]]

                            # (拼接回 CPU 上的完整容积)
                            gen_volume_tensor[:, :, d_start:d_end, h_start:h_end, w_start:w_end] = gen_patch.cpu()

            # [!!] 致命修正 2：NIfTI 保存 (多进程修复)]
            # 移除 2D plt.imshow
            # 在主进程中安全地执行轻量级 I/O
            try:
                lq_nii_for_header = nib.load(lq_path)
                affine = lq_nii_for_header.affine
                header = lq_nii_for_header.header
            except Exception as e:
                logging.warning(f"Could not load NIfTI header from {lq_path}, using default affine. Error: {e}")
                affine = np.array([ # 默认 1mm 各向同性
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
                header = nib.Nifti1Header() # 空头文件


            # [!!] 致命修正 3：反归一化 (指标修复)]
            # .squeeze() 移除 B 和 C 维度 -> (D, H, W)
            gen_np = de_normalize(gen_volume_tensor.squeeze().cpu().numpy())
            hq_np = de_normalize(hq_volume_tensor.squeeze().cpu().numpy())
            
            # (可选) 保存 LQ 反归一化后的版本
            lq_np = de_normalize(lq_volume_tensor.squeeze().cpu().numpy())

            # 保存 NIfTI 文件
            base_filename = os.path.basename(lq_path).replace('_lq.nii', '').replace('.nii.gz', '').replace('.nii', '')
            
            gen_nii_path = os.path.join(images_output_dir, f'{base_filename}_generated.nii.gz')
            gen_nii = nib.Nifti1Image(gen_np, affine, header)
            nib.save(gen_nii, gen_nii_path)
            
            # (可选) 保存 HQ 和 LQ 以便对比
            hq_nii_path = os.path.join(images_output_dir, f'{base_filename}_ground_truth.nii.gz')
            hq_nii = nib.Nifti1Image(hq_np, affine, header)
            nib.save(hq_nii, hq_nii_path)
            
            lq_nii_path = os.path.join(images_output_dir, f'{base_filename}_low_quality_input.nii.gz')
            lq_nii = nib.Nifti1Image(lq_np, affine, header)
            nib.save(lq_nii, lq_nii_path)


            # [!!] 致命修正 4：正确评估 (指标修复)]
            image_evaluator.evaluate(
                gen_np, hq_np, opt, plane_wave_data, test_type, image_counter
            )

            image_counter += 1


    # ... (循环结束后的结果打印和保存代码保持不变) ...
    print(f"\n--- Test Results for Type {test_type} ---")
    image_evaluator.print_results(opt)
    # (保存指标到文件的代码...)
    try:
        metrics_output_dir = os.path.join('./project_assets/output_visualizations', opt.name, 'test') # 修正指标保存的基础路径
        metrics_file_path = os.path.join(metrics_output_dir, f'type_{test_type}_metrics.txt') # 文件名在 type_specific_dir 里
        # 确保目录存在 (因为 metrics.py 不再创建它)
        type_specific_dir_for_metrics = os.path.join(metrics_output_dir, f'type_{test_type}')
        os.makedirs(type_specific_dir_for_metrics, exist_ok=True)
        # 将指标文件保存在类型子目录中
        metrics_file_path_in_type_dir = os.path.join(type_specific_dir_for_metrics, 'metrics.txt')

        with open(metrics_file_path_in_type_dir, 'w') as f: # 使用正确的路径
            f.write(f"Test Results for Type {test_type} ({opt.name})\n")
            f.write("-------------------------------------------\n")
            # 重新获取 message (print_results 不再保存到文件)
            message = ''
            message += '----------------- Evaluations ---------------\n'
            comment = '\t(Average over all test images, NaN ignored)'
            for key in image_evaluator.result.keys():
                 mean_value = np.nanmean(image_evaluator.result[key]) # 使用 nanmean
                 label = key + ':' if key not in ['PSNR'] else 'PSNR [dB]:'
                 if key == 'CR': label = 'Contrast:'
                 message += '{:>25}: {:<30}{}\n'.format(label, str(mean_value), comment)
            message += '----------------- End -------------------'
            f.write(message) # 将重新生成的消息写入文件
            f.write('\n')
        print(f"Metrics saved to: {metrics_file_path_in_type_dir}") # 打印正确的保存路径
    except Exception as e:
        print(f"Error saving metrics to file: {e}")

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from tqdm import tqdm
# import time
# import warnings # 导入 warnings 模块
# import logging # <-- 确保导入 logging
# from options.test_options import TestOptions
# from models import create_model
# from data_process import load_dataset # 不再需要导入 test_image，因为我们在 test.py 里处理了
# from torch.utils.data import DataLoader
# from metrics import image_evaluation
# from cubdl_master.example_picmus_torch import load_datasets, create_network, dispaly_img # <-- 导入 dispaly_img
# warnings.filterwarnings("ignore", category=UserWarning)
# if __name__ == '__main__':
#     # --- 获取 root logger 并保存原始级别 ---
#     logger = logging.getLogger()
#     original_logging_level = logger.getEffectiveLevel()
#     # --- 添加结束 ---

#     # --- 1. 初始化和加载设置 ---
#     try:
#         plane_wave_data = load_datasets("simulation", "resolution_distorsion", "iq")
#         das, iqdata, xlims, zlims = create_network(plane_wave_data, [1])
#     except Exception as e:
#         print(f"Warning: PICMUS dataset not available, using default limits. Error: {e}")
#         plane_wave_data = None
#         xlims, zlims = [0, 1], [0, 1]

#     opt = TestOptions().parse()
#     opt.batch_size = 1
#     opt.serial_batches = True
#     opt.no_flip = True
#     test_type = opt.test_type

#     # --- 2. 创建模型并加载权重 ---
#     model = create_model(opt)
#     model.setup(opt)
#     if opt.eval: model.eval()
#     image_evaluator = image_evaluation()

#     # --- 3. 加载测试数据集 ---
#     print(f"Loading test dataset for test_type = {test_type}...")
#     image_dataset = load_dataset(opt, opt.phase, test_type)
#     if len(image_dataset) == 0:
#         print(f"Error: No data loaded for test_type={test_type} and phase='{opt.phase}'. Skipping test.")
#         exit()
#     test_loader = DataLoader(dataset=image_dataset, num_workers=0, batch_size=1, shuffle=False)
#     test_bar = tqdm(test_loader, desc=f"Testing {opt.name} (Type {test_type})", unit="image")
#     image_counter = 1
#     images_output_dir = os.path.join('./project_assets/output_visualizations', opt.name, 'test', f'type_{test_type}')
#     os.makedirs(images_output_dir, exist_ok=True)
#     print(f"Output images will be saved to: {images_output_dir}")


#     for low_quality_image, high_quality_image in test_bar:
#             model.set_input(low_quality_image, high_quality_image)
#             start_time = time.time()
#             model.test()

#             # --- 5. 评估打分和保存结果 ---
#             # !! 不再需要 try...finally 和 logger.setLevel !!
#             # !! 也不再需要 with warnings.catch_warnings() !!

#             # --- 保存图片 ---
#             # (这部分代码保持不变)
#             lq_image_np = model.low_quality_image[0].cpu().detach().numpy(); lq_image_np = np.squeeze(lq_image_np); lq_image_np -= np.max(lq_image_np)
#             gen_image_np = model.generated_image[0].cpu().detach().numpy(); gen_image_np = np.squeeze(gen_image_np); gen_image_np -= np.max(gen_image_np)
#             hq_image_np = model.high_quality_image[0].cpu().detach().numpy(); hq_image_np = np.squeeze(hq_image_np); hq_image_np -= np.max(hq_image_np)
#             current_image_save_path = os.path.join(images_output_dir, f'{image_counter}_test.png')
#             plt.figure(); extent = [xlims[0] * 1e3, xlims[1] * 1e3, zlims[1] * 1e3, zlims[0] * 1e3]
#             plt.subplot(131); plt.imshow(lq_image_np, vmin=-60, vmax=0, cmap="gray", extent=extent, origin="upper"); plt.title(f"{image_counter} test LR image", fontsize=10)
#             plt.subplot(132); plt.imshow(gen_image_np, vmin=-60, vmax=0, cmap="gray", extent=extent, origin="upper"); plt.title(f"{image_counter} test generated image", fontsize=10)
#             plt.subplot(133); plt.imshow(hq_image_np, vmin=-60, vmax=0, cmap="gray", extent=extent, origin="upper"); plt.title(f"{image_counter} test HR image", fontsize=10)
#             plt.savefig(current_image_save_path); plt.close()

#             # --- 调用评估 (UserWarning 会被全局设置忽略) ---
#             image_evaluator.evaluate(
#                 gen_image_np, hq_image_np, opt, plane_wave_data, test_type, image_counter
#             )

#             image_counter += 1


#     # ... (循环结束后的结果打印和保存代码保持不变) ...
#     print(f"\n--- Test Results for Type {test_type} ---")
#     image_evaluator.print_results(opt)
#     # (保存指标到文件的代码...)
#     try:
#         metrics_output_dir = os.path.join('./project_assets/output_visualizations', opt.name, 'test') # 修正指标保存的基础路径
#         metrics_file_path = os.path.join(metrics_output_dir, f'type_{test_type}_metrics.txt') # 文件名在 type_specific_dir 里
#         # 确保目录存在 (因为 metrics.py 不再创建它)
#         type_specific_dir_for_metrics = os.path.join(metrics_output_dir, f'type_{test_type}')
#         os.makedirs(type_specific_dir_for_metrics, exist_ok=True)
#         # 将指标文件保存在类型子目录中
#         metrics_file_path_in_type_dir = os.path.join(type_specific_dir_for_metrics, 'metrics.txt')

#         with open(metrics_file_path_in_type_dir, 'w') as f: # 使用正确的路径
#             f.write(f"Test Results for Type {test_type} ({opt.name})\n")
#             f.write("-------------------------------------------\n")
#             # 重新获取 message (print_results 不再保存到文件)
#             message = ''
#             message += '----------------- Evaluations ---------------\n'
#             comment = '\t(Average over all test images, NaN ignored)'
#             for key in image_evaluator.result.keys():
#                  mean_value = np.nanmean(image_evaluator.result[key]) # 使用 nanmean
#                  label = key + ':' if key not in ['PSNR'] else 'PSNR [dB]:'
#                  if key == 'CR': label = 'Contrast:'
#                  message += '{:>25}: {:<30}{}\n'.format(label, str(mean_value), comment)
#             message += '----------------- End -------------------'
#             f.write(message) # 将重新生成的消息写入文件
#             f.write('\n')
#         print(f"Metrics saved to: {metrics_file_path_in_type_dir}") # 打印正确的保存路径
#     except Exception as e:
#         print(f"Error saving metrics to file: {e}")