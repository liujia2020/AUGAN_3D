import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time
import warnings # 导入 warnings 模块
import logging # <-- 确保导入 logging
from options.test_options import TestOptions
from models import create_model
from data_process import load_dataset # 不再需要导入 test_image，因为我们在 test.py 里处理了
from torch.utils.data import DataLoader
from metrics import image_evaluation
from cubdl_master.example_picmus_torch import load_datasets, create_network, dispaly_img # <-- 导入 dispaly_img
warnings.filterwarnings("ignore", category=UserWarning)
if __name__ == '__main__':
    # --- 获取 root logger 并保存原始级别 ---
    logger = logging.getLogger()
    original_logging_level = logger.getEffectiveLevel()
    # --- 添加结束 ---

    # --- 1. 初始化和加载设置 ---
    try:
        plane_wave_data = load_datasets("simulation", "resolution_distorsion", "iq")
        das, iqdata, xlims, zlims = create_network(plane_wave_data, [1])
    except Exception as e:
        print(f"Warning: PICMUS dataset not available, using default limits. Error: {e}")
        plane_wave_data = None
        xlims, zlims = [0, 1], [0, 1]

    opt = TestOptions().parse()
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    test_type = opt.test_type

    # --- 2. 创建模型并加载权重 ---
    model = create_model(opt)
    model.setup(opt)
    if opt.eval: model.eval()
    image_evaluator = image_evaluation()

    # --- 3. 加载测试数据集 ---
    print(f"Loading test dataset for test_type = {test_type}...")
    image_dataset = load_dataset(opt, opt.phase, test_type)
    if len(image_dataset) == 0:
        print(f"Error: No data loaded for test_type={test_type} and phase='{opt.phase}'. Skipping test.")
        exit()
    test_loader = DataLoader(dataset=image_dataset, num_workers=0, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc=f"Testing {opt.name} (Type {test_type})", unit="image")
    image_counter = 1
    images_output_dir = os.path.join('./project_assets/output_visualizations', opt.name, 'test', f'type_{test_type}')
    os.makedirs(images_output_dir, exist_ok=True)
    print(f"Output images will be saved to: {images_output_dir}")


    for low_quality_image, high_quality_image in test_bar:
            model.set_input(low_quality_image, high_quality_image)
            start_time = time.time()
            model.test()

            # --- 5. 评估打分和保存结果 ---
            # !! 不再需要 try...finally 和 logger.setLevel !!
            # !! 也不再需要 with warnings.catch_warnings() !!

            # --- 保存图片 ---
            # (这部分代码保持不变)
            lq_image_np = model.low_quality_image[0].cpu().detach().numpy(); lq_image_np = np.squeeze(lq_image_np); lq_image_np -= np.max(lq_image_np)
            gen_image_np = model.generated_image[0].cpu().detach().numpy(); gen_image_np = np.squeeze(gen_image_np); gen_image_np -= np.max(gen_image_np)
            hq_image_np = model.high_quality_image[0].cpu().detach().numpy(); hq_image_np = np.squeeze(hq_image_np); hq_image_np -= np.max(hq_image_np)
            current_image_save_path = os.path.join(images_output_dir, f'{image_counter}_test.png')
            plt.figure(); extent = [xlims[0] * 1e3, xlims[1] * 1e3, zlims[1] * 1e3, zlims[0] * 1e3]
            plt.subplot(131); plt.imshow(lq_image_np, vmin=-60, vmax=0, cmap="gray", extent=extent, origin="upper"); plt.title(f"{image_counter} test LR image", fontsize=10)
            plt.subplot(132); plt.imshow(gen_image_np, vmin=-60, vmax=0, cmap="gray", extent=extent, origin="upper"); plt.title(f"{image_counter} test generated image", fontsize=10)
            plt.subplot(133); plt.imshow(hq_image_np, vmin=-60, vmax=0, cmap="gray", extent=extent, origin="upper"); plt.title(f"{image_counter} test HR image", fontsize=10)
            plt.savefig(current_image_save_path); plt.close()

            # --- 调用评估 (UserWarning 会被全局设置忽略) ---
            image_evaluator.evaluate(
                gen_image_np, hq_image_np, opt, plane_wave_data, test_type, image_counter
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