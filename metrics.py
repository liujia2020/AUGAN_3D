# 文件: metrics.py

# ... (文件顶部的 import 保持不变) ...
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import torch # 确保导入了 torch

from sklearn import metrics
import logging # 导入 logging

import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

# --- 从 cubdl_master 导入必要的函数 ---
try:
    from cubdl_master.PixelGrid import make_pixel_grid
except ImportError:
    logging.error("无法从 cubdl_master.PixelGrid 导入 make_pixel_grid。请确保 cubdl_master 包在 Python 路径中。")
    def make_pixel_grid(*args, **kwargs):
        logging.error("make_pixel_grid 未正确导入，FWHM 计算将失败。")
        raise NotImplementedError("make_pixel_grid is not available.")
# --- 导入结束 ---

# --- 指标计算函数 (contrast, cnr, gcnr, MI, ..., Compute_6dB_Resolution 保持不变) ---
# (这里省略这些函数的代码，假设它们是正确的)
# Compute contrast ratio
def contrast(img1, img2):
    mean2 = img2.mean()
    if mean2 == 0 or np.isnan(mean2) or np.isnan(img1.mean()): return np.nan
    return img1.mean() / mean2
# Compute contrast-to-noise ratio
def cnr(img1, img2):
    var_sum = img1.var() + img2.var()
    if var_sum == 0 or np.isnan(var_sum) or np.isnan(img1.mean()) or np.isnan(img2.mean()): return np.nan
    return np.abs(img1.mean() - img2.mean()) / np.sqrt(var_sum)
# Compute the generalized contrast-to-noise ratio
def gcnr(img1, img2):
    if img1.size == 0 or img2.size == 0 or np.all(np.isnan(img1)) or np.all(np.isnan(img2)): return np.nan
    try:
        a = np.concatenate((img1.flatten(), img2.flatten())); a = a[~np.isnan(a)]
        if a.size == 0: return np.nan
        _, bins = np.histogram(a, bins=256)
        f, _ = np.histogram(img1[~np.isnan(img1)], bins=bins, density=True)
        g, _ = np.histogram(img2[~np.isnan(img2)], bins=bins, density=True)
        f_sum, g_sum = f.sum(), g.sum()
        if f_sum == 0 or g_sum == 0: return np.nan
        f /= f_sum; g /= g_sum
        return 1 - np.sum(np.minimum(f, g))
    except Exception as e: logging.error(f"Error calculating GCNR: {e}"); return np.nan
def MI(img1, img2):
    try:
        image1 = np.squeeze(img1); image2 = np.squeeze(img2)
        result_NMI = metrics.normalized_mutual_info_score(image1.flatten(), image2.flatten())
        return result_NMI
    except Exception as e: logging.error(f"Error calculating MI: {e}"); return np.nan
def snr(img):
    std_val = img.std()
    if std_val == 0 or np.isnan(std_val) or np.isnan(img.mean()): return np.nan
    return img.mean() / std_val
def l1loss(img1, img2): return np.abs(img1 - img2).mean()
def l2loss(img1, img2): return np.sqrt(((img1 - img2) ** 2).mean())
def psnr(img1, img2):
    loss = l2loss(img1, img2)
    if loss == 0 or np.isnan(loss): return np.inf
    dynamic_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())
    if dynamic_range == 0 or np.isnan(dynamic_range): return np.nan
    # 添加保护防止 log10(负数或零)
    if dynamic_range / loss <= 0: return np.nan
    return 20 * np.log10(dynamic_range / loss)
def ncc(img1, img2):
    mean1, mean2 = img1.mean(), img2.mean()
    term1_sq_sum = ((img1 - mean1) ** 2).sum(); term2_sq_sum = ((img2 - mean2) ** 2).sum()
    denominator = np.sqrt(term1_sq_sum * term2_sq_sum)
    if denominator == 0 or np.isnan(denominator): return np.nan
    numerator = ((img1 - mean1) * (img2 - mean2)).sum()
    return numerator / denominator
def Compute_6dB_Resolution(x_axis, y_signal):
    if x_axis is None or y_signal is None or len(x_axis) < 2 or len(y_signal) < 2 or len(x_axis) != len(y_signal): logging.warning("Invalid input for Compute_6dB_Resolution."); return np.nan
    y_signal_squeezed = np.squeeze(y_signal); max_y = np.max(y_signal_squeezed); threshold = max_y - 6
    if np.all(y_signal_squeezed < threshold): logging.warning("Signal does not cross -6dB threshold."); return np.nan
    try:
        coeff = 10; nb_sample = np.size(x_axis); nb_interp = nb_sample * coeff
        if x_axis[0] == x_axis[nb_sample-1]: logging.warning("x_axis has identical start and end points."); return np.nan
        x_interp = np.linspace(x_axis[0], x_axis[nb_sample-1], nb_interp)
        y_interp = np.interp(x_interp, x_axis, y_signal_squeezed)
        ind = np.where(y_interp >= threshold)[0]
        if ind.size < 2: logging.warning("Could not find valid indices crossing -6dB threshold."); return np.nan
        idx1 = np.min(ind); idx2 = np.max(ind); res = x_interp[idx2] - x_interp[idx1]
        if res <= 0 or res > (x_axis[-1] - x_axis[0]): logging.warning(f"Calculated FWHM ({res}) seems unreasonable.")
        return res
    except Exception as e: logging.error(f"Error in Compute_6dB_Resolution: {e}"); return np.nan


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    # (channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


# --- image_evaluation 类 (最终修正版) ---
class image_evaluation():
    def __init__(self):
        # 初始化用于存储每次评估结果的列表
        self.result = {
            'CR': [], 'CNR': [], 'sSNR': [], 'GCNR': [],
            'PSNR': [], 'NCC': [], 'L1loss': [], 'L2loss': [],
            'FWHM': [], 'SSIM': [], 'MI': []
        }
        # 初始化用于在 evaluate 函数内部临时存储当次计算的平均值的变量
        self.current_average_score_FWHM = np.nan
        self.current_average_score_CR = np.nan
        self.current_average_score_CNR = np.nan
        self.current_average_score_sSNR = np.nan
        self.current_average_score_GCNR = np.nan
        # 增加 grid 属性，避免重复计算
        self.x_matrix = None
        self.z_matrix = None


    def evaluate(self, img1, img2, opt, plane_wave_data, test_type, i):
        """
        评估生成的图像 img1 与目标图像 img2 之间的指标。
        """
        # 归一化（注意：这里是减去最大值）
        img1_norm = img1 - np.max(img1)
        img2_norm = img2 - np.max(img2)

        # 重置当前轮次的平均分
        self.current_average_score_FWHM = np.nan
        self.current_average_score_CR = np.nan
        self.current_average_score_CNR = np.nan
        self.current_average_score_sSNR = np.nan
        self.current_average_score_GCNR = np.nan

        image = img1_norm # 使用归一化后的生成图像进行评估

        # --- 预先计算 grid (如果需要且尚未计算) ---
        needs_grid = (test_type in [1, 3, 2, 4]) # FWHM 和 CR/CNR 都需要 grid
        grid_calculated_ok = True
        if needs_grid and (self.x_matrix is None or self.z_matrix is None):
            if plane_wave_data and hasattr(plane_wave_data, 'ele_pos') and hasattr(plane_wave_data, 'c') and hasattr(plane_wave_data, 'fc'):
                 try:
                     xlims = [plane_wave_data.ele_pos[0, 0], plane_wave_data.ele_pos[-1, 0]]
                     zlims = [5e-3, 55e-3]
                     wvln = plane_wave_data.c / plane_wave_data.fc
                     dx = wvln / 3
                     dz = dx
                     # 确保 make_pixel_grid 已导入
                     grid = make_pixel_grid(xlims, zlims, dx, dz)
                     self.x_matrix = grid[:,:,0]
                     self.z_matrix = grid[:,:,2]
                     logging.info("Grid calculated for metrics.")
                 except Exception as e:
                     logging.error(f"Image {i}: Error calculating grid for metrics: {e}. Skipping grid-dependent metrics.")
                     grid_calculated_ok = False
            else:
                logging.warning(f"Image {i}: Skipping grid calculation (plane_wave_data missing required attributes).")
                grid_calculated_ok = False
        # --- grid 计算结束 ---

        # --- FWHM 计算 (test_type 1 或 3) ---
        if test_type in [1, 3] and grid_calculated_ok and hasattr(plane_wave_data, 'x_axis') and hasattr(plane_wave_data, 'z_axis'):
            # (FWHM 计算逻辑基本不变，但依赖于 self.x_matrix, self.z_matrix 已正确计算)
            maskROI = np.zeros((508,387)) # 假设 FWHM 分析基于 508x387 区域
            # 定义 sca 点坐标...
            if test_type == 1:
                if i <= 45: sca = np.array([[0,0,0.01],[0,0,0.015],[0,0,0.02],[0,0,0.025],[0,0,0.03],[0,0,0.035],[0,0,0.04],[0,0,0.045],[-0.015,0,0.02],[-0.01,0,0.02],[-0.005,0,0.02],[0.005,0,0.02],[0.01,0,0.02],[0.015,0,0.02],[-0.015,0,0.04],[-0.01,0,0.04],[-0.005,0,0.04],[0.005,0,0.04],[0.01,0,0.04],[0.015,0,0.04]])
                else: sca = np.array([[0,0,0.015],[0,0,0.02],[0,0,0.025],[0,0,0.03],[0,0,0.035],[0,0,0.04],[0,0,0.045],[0,0,0.05],[-0.015,0,0.02],[-0.01,0,0.02],[-0.005,0,0.02],[0.005,0,0.02],[0.01,0,0.02],[0.015,0,0.02],[-0.015,0,0.04],[-0.01,0,0.04],[-0.005,0,0.04],[0.005,0,0.04],[0.01,0,0.04],[0.015,0,0.04]])
            elif test_type == 3:
                if i <= 30: sca = np.array([[-0.0005,0,0.0096],[-0.0004,0,0.0187],[-0.0004,0,0.028],[-0.0002,0,0.0376],[-0.0001,0,0.047],[-0.0105,0,0.0375],[0.0098,0,0.0376]])
                elif i > 30 and i <= 45: sca = np.array([[0.0005,0,0.0096],[0.0004,0,0.0187],[0.0004,0,0.028],[0.0002,0,0.0376],[0.0001,0,0.047],[0.0105,0,0.0375],[-0.0098,0,0.0376]])
                else: sca = np.array([[-0.0005,0,0.0504],[-0.0004,0,0.0413],[-0.0004,0,0.032],[-0.0002,0,0.0224],[-0.0001,0,0.013],[-0.0105,0,0.0225],[0.0098,0,0.0224]])

            # 检查 grid 尺寸是否匹配 FWHM ROI (508, 387)
            if self.x_matrix.shape != (508, 387) or self.z_matrix.shape != (508, 387):
                 logging.warning(f"Image {i}: Grid shape {self.x_matrix.shape} != (508, 387). Skipping FWHM.")
            else:
                fwhm_calculation_possible = True
                for k in range(sca.shape[0]):
                    x, z = sca[k][0], sca[k][2]
                    try:
                        mask = (k+1) * ((self.x_matrix > (x-0.0018)) & (self.x_matrix < (x+0.0018)) & (self.z_matrix > (z-0.0018))& (self.z_matrix<(z+0.0018)))
                        maskROI = maskROI + mask
                    except ValueError as e:
                        logging.error(f"Image {i}: Error creating FWHM mask for point {k}: {e}. Skipping FWHM.")
                        fwhm_calculation_possible = False; break

                if fwhm_calculation_possible:
                    patchImg1 = np.full((508, 387), np.min(image))
                    h_img, w_img = image.shape
                    h_copy, w_copy = min(h_img, 508), min(w_img, 384)
                    patchImg1[0:h_copy, 0:w_copy] = image[0:h_copy, 0:w_copy]
                    if w_copy >= 381: patchImg1[:, 384:387] = patchImg1[:, 381:384]

                    score1 = np.full((sca.shape[0], 2), np.nan)
                    global_min_value = np.min(image)

                    for k in range(sca.shape[0]):
                        patchMask = np.copy(maskROI); patchImg = np.copy(patchImg1)
                        patchImg[maskROI != (k+1)] = global_min_value
                        patchMask[maskROI != (k+1)] = 0
                        [idzz, idxx] = np.where(patchMask == (k+1))
                        if idxx.size == 0 or idzz.size == 0: logging.warning(f"Image {i}: FWHM mask for point {k} empty."); continue

                        min_idxx, max_idxx = np.min(idxx), np.max(idxx)
                        min_idzz, max_idzz = np.min(idzz), np.max(idzz)
                        if max_idxx >= len(plane_wave_data.x_axis) or max_idzz >= len(plane_wave_data.z_axis): logging.warning(f"Image {i}: FWHM indices for point {k} out of bounds."); continue

                        try:
                            x_patch = plane_wave_data.x_axis[min_idxx : max_idxx+1] * 1e3
                            z_patch = plane_wave_data.z_axis[min_idzz : max_idzz+1] * 1e3
                            sub_patch = patchImg[min_idzz : max_idzz+1, min_idxx : max_idxx+1]
                            if sub_patch.size == 0: continue
                            max_flat_idx = np.argmax(sub_patch)
                            idz_rel, idx_rel = np.unravel_index(max_flat_idx, sub_patch.shape)
                            idz_abs, idx_abs = idz_rel + min_idzz, idx_rel + min_idxx

                            signalLateral = patchImg[idz_abs, min_idxx : max_idxx+1]
                            signalAxial = patchImg[min_idzz : max_idzz+1, idx_abs]
                            if len(signalAxial) != len(z_patch) or len(signalLateral) != len(x_patch): logging.warning(f"Image {i}: Signal/axis length mismatch for point {k}."); continue

                            res_axial = Compute_6dB_Resolution(z_patch, signalAxial)
                            res_lateral = Compute_6dB_Resolution(x_patch, signalLateral)
                            score1[k, 0] = res_axial; score1[k, 1] = res_lateral
                        except Exception as e: logging.error(f"Image {i}: Error during FWHM calc for point {k}: {e}"); continue

                    self.current_average_score_FWHM = np.nanmean(score1)
        # --- FWHM 计算结束 ---

        # --- CR/CNR/sSNR/GCNR 计算 (test_type 2 或 4) ---
        if test_type in [2, 4] and grid_calculated_ok and self.x_matrix is not None and self.z_matrix is not None:
             # 检查 grid 尺寸是否匹配 image 尺寸
             if self.x_matrix.shape != image.shape:
                 logging.warning(f"Image {i}: Grid shape {self.x_matrix.shape} != image shape {image.shape}. Skipping CR/CNR/sSNR/GCNR.")
             else:
                # 定义囊肿参数...
                if test_type == 2:
                    self.occlusionDiameter = np.array([0.008]*9); self.r, self.rin, self.rout1 = 0.004, 0.004 - 6.2407e-4, 0.004 + 6.2407e-4
                    self.rout2 = 1.2 * np.sqrt(self.rin**2 + self.rout1**2); self.xcenter = [0]*3 + [-0.012]*3 + [0.012]*3; self.zcenter = [0.018, 0.03, 0.042] * 3
                elif test_type == 4:
                    self.occlusionDiameter = np.array([0.0045, 0.0045]); self.r, self.rin, self.rout1 = 0.0022, 0.0022 - 6.2407e-4, 0.0022 + 6.2407e-4
                    self.rout2 = 1.2 * np.sqrt(self.rin**2 + self.rout1**2)
                    if i <=30: self.xcenter, self.zcenter = [-1.0e-04]*2, [0.0149, 0.0428]
                    elif i > 30 and i <= 45: self.xcenter, self.zcenter = [1.0e-04]*2, [0.0149, 0.0428]
                    else: self.xcenter, self.zcenter = [-1.0e-04]*2, [0.0172, 0.0451]

                score2 = np.full(len(self.occlusionDiameter), np.nan) # CR
                score3 = np.full(len(self.occlusionDiameter), np.nan) # CNR
                score4 = np.full(len(self.occlusionDiameter), np.nan) # sSNR
                score5 = np.full(len(self.occlusionDiameter), np.nan) # GCNR

                for k in range(len(self.occlusionDiameter)):
                    xc, zc = self.xcenter[k], self.zcenter[k]
                    dist_sq = (self.x_matrix - xc)**2 + (self.z_matrix - zc)**2
                    maskInside = dist_sq <= (self.rin**2)
                    maskOutside = (dist_sq >= (self.rout1**2)) & (dist_sq <= (self.rout2**2))
                    inside_pixels = image[maskInside]; outside_pixels = image[maskOutside]
                    if inside_pixels.size == 0 or outside_pixels.size == 0: logging.warning(f"Image {i}: Not enough pixels for cyst {k}."); continue

                    mean_in, var_in = np.mean(inside_pixels), np.var(inside_pixels)
                    mean_out, var_out, std_out = np.mean(outside_pixels), np.var(outside_pixels), np.std(outside_pixels)

                    score2[k] = np.abs(mean_in - mean_out) # CR (线性)
                    if np.sqrt(var_in + var_out) > 1e-9: score3[k] = np.abs(mean_in - mean_out) / np.sqrt(var_in + var_out) # CNR
                    if std_out > 1e-9: score4[k] = np.abs(mean_out) / std_out # sSNR
                    score5[k] = gcnr(inside_pixels, outside_pixels) # GCNR

                self.current_average_score_CR = np.nanmean(score2)
                self.current_average_score_CNR = np.nanmean(score3)
                self.current_average_score_sSNR = np.nanmean(score4)
                self.current_average_score_GCNR = np.nanmean(score5)
        # --- CR/CNR/sSNR/GCNR 计算结束 ---

        # --- 计算通用指标 ---
        self.PSNR = psnr(img1_norm, img2_norm)
        self.MI = MI(img1_norm, img2_norm)

        # --- SSIM 计算 (确保输入是 4D Tensor) ---
        self.SSIM = np.nan # 默认值
        try:
            ima1_tensor = torch.from_numpy(img1_norm).float() # (H, W)
            ima1 = ima1_tensor.unsqueeze(0).unsqueeze(1)      # (1, 1, H, W)
            ima2_tensor = torch.from_numpy(img2_norm).float() # (H, W)
            ima2 = ima2_tensor.unsqueeze(0).unsqueeze(1)      # (1, 1, H, W)

            if ima1.dim() != 4 or ima2.dim() != 4:
                 logging.error(f"Image {i}: SSIM input tensors are not 4D! ima1 shape: {ima1.shape}, ima2 shape: {ima2.shape}")
            else:
                # 移到 CPU 计算 SSIM
                ssim_val = ssim(ima1.cpu(), ima2.cpu())
                self.SSIM = ssim_val.item() # 获取标量值
        except Exception as e:
            logging.error(f"Image {i}: Error calculating SSIM: {e}")
        # --- SSIM 计算结束 ---

        self.L1Loss = l1loss(img1_norm, img2_norm)
        self.L2Loss = l2loss(img1_norm, img2_norm)
        self.NCC = ncc(img1_norm, img2_norm)

        # --- 存储当次计算结果 ---
        self.result['FWHM'].append(self.current_average_score_FWHM)
        self.result['CR'].append(self.current_average_score_CR)
        self.result['CNR'].append(self.current_average_score_CNR)
        self.result['sSNR'].append(self.current_average_score_sSNR)
        self.result['GCNR'].append(self.current_average_score_GCNR)
        self.result['L1loss'].append(self.L1Loss)
        self.result['L2loss'].append(self.L2Loss)
        self.result['PSNR'].append(self.PSNR)
        self.result['NCC'].append(self.NCC)
        # 确保存入的是 float 或 nan
        ssim_to_append = float(self.SSIM) if isinstance(self.SSIM, (int, float, np.number)) and not np.isnan(self.SSIM) else np.nan
        self.result['SSIM'].append(ssim_to_append)
        self.result['MI'].append(self.MI)


    def print_results(self, opt):
        """打印所有测试图像的平均指标分数 (忽略 NaN)。"""
        message = ''
        message += '----------------- Evaluations ---------------\n'
        comment = '\t(Average over all test images, NaN ignored)'
        for key in self.result.keys():
             mean_value = np.nanmean(self.result[key]) # 使用 nanmean
             label = key + ':' if key not in ['PSNR'] else 'PSNR [dB]:' # 特殊处理 PSNR 标签
             if key == 'CR': label = 'Contrast:' # 重命名 CR
             message += '{:>25}: {:<30}{}\n'.format(label, str(mean_value), comment)
        message += '----------------- End -------------------'
        print(message)
        # 调用 save_result 保存
        self.save_result(opt, message)

    def save_result(self, opt, message):
            """将评估结果（通常是平均值）保存到对应测试类型的图片文件夹中。"""
            try:
                # --- 修改路径构建逻辑 ---
                # 基础路径指向 images 文件夹下的 test 目录
                output_base_dir = os.path.join('./project_assets/output_visualizations', opt.name, 'test')
                
                # base_dir = os.path.join('./images', opt.name, 'test') # 例如 ./images/augan_run_2/test/
                test_type_str = f"type_{opt.test_type}" if hasattr(opt, 'test_type') else "summary"
                # 完整的、特定于类型的子目录路径
                # type_specific_dir = os.path.join(base_dir, test_type_str) # 例如 ./images/augan_run_2/test/type_1/
                type_specific_dir = os.path.join(output_base_dir, test_type_str) # <-- 这里用到了 output_base_dir
                # --- 修改结束 ---

                os.makedirs(type_specific_dir, exist_ok=True) # 确保这个子目录存在

                # --- 修改文件名 ---
                # 直接命名为 metrics.txt，因为它已经在特定类型的文件夹里了
                file_name = os.path.join(type_specific_dir, 'metrics.txt') # 例如 ./images/augan_run_2/test/type_1/metrics.txt
                # --- 修改结束 ---

                with open(file_name, 'w') as opt_file: # 使用 'w' 覆盖模式
                    opt_file.write(f"Evaluation Results for {opt.name} ({test_type_str})\n")
                    opt_file.write(message)
                    opt_file.write('\n')
                logging.info(f"Evaluation results saved to {file_name}") # 更新日志信息
            except Exception as e:
                logging.error(f"Failed to save evaluation results: {e}")

# --- 文件末尾 ---