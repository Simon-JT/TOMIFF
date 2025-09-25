# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from Evaluator import Evaluator
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


fused_dir = "test_results/FMB_result/Gray"
ir_dir = "test_data/FMB_test/ir"
vi_dir = "test_data/FMB_test/vi"


fused_files = sorted([f for f in os.listdir(fused_dir) if f.endswith('.png')])

def get_test_file_extension():
    """检测测试数据文件的扩展名"""
    ir_files = os.listdir(ir_dir)
    if ir_files:
        # 获取第一个文件的扩展名
        first_file = ir_files[0]
        _, ext = os.path.splitext(first_file)
        return ext
    return '.png'  # 默认返回png

def evaluate_one(fname):
    fused_path = os.path.join(fused_dir, fname)
    
    # 获取测试数据的文件扩展名
    test_ext = get_test_file_extension()
    
    # 根据测试数据的扩展名决定文件名
    base_name = os.path.splitext(fname)[0]  # 去掉.png扩展名
    if test_ext == '.jpg':
        # 如果测试数据是jpg格式，需要转换文件名
        test_fname = base_name + '.jpg'
    else:
        # 如果测试数据是png格式，直接使用相同文件名
        test_fname = fname
    
    ir_path = os.path.join(ir_dir, test_fname)
    vi_path = os.path.join(vi_dir, test_fname)

    # 读取融合结果图像
    fused_img = cv2.imread(fused_path, cv2.IMREAD_GRAYSCALE)
    if fused_img is None:
        print(f"Warning: Cannot read fused image {fused_path}")
        return None
    
    # 读取测试数据图像
    ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if ir_img is None:
        print(f"Warning: Cannot read IR image {ir_path}")
        return None
        
    vi_img = cv2.imread(vi_path, cv2.IMREAD_GRAYSCALE)
    if vi_img is None:
        print(f"Warning: Cannot read VI image {vi_path}")
        return None

    # 转换为float32类型
    fused_img = np.array(fused_img, dtype=np.float32)
    ir_img = np.array(ir_img, dtype=np.float32)
    vi_img = np.array(vi_img, dtype=np.float32)

    # 确保图像值在0-255范围内
    fused_img = np.clip(fused_img, 0, 255)
    ir_img = np.clip(ir_img, 0, 255)
    vi_img = np.clip(vi_img, 0, 255)
                

    current_metrics = np.array([
        Evaluator.EN(fused_img),
        Evaluator.SD(fused_img),
        Evaluator.SF(fused_img),
        Evaluator.MI(fused_img, ir_img, vi_img),
        Evaluator.VIFF(fused_img, ir_img, vi_img),
        Evaluator.Qabf(fused_img, ir_img, vi_img),
        Evaluator.SCD(fused_img, ir_img, vi_img),
        Evaluator.SSIM(fused_img, ir_img, vi_img)
    ])
    
    if not np.any(np.isnan(current_metrics)):
        return current_metrics
    else:
        print(f"Warning: NaN metrics detected in {fname}")
        return None

metric_result = np.zeros(8)  # 增加到8个指标
num = 0

with ProcessPoolExecutor() as executor:
    results = list(tqdm(executor.map(evaluate_one, fused_files), total=len(fused_files), ncols=100))

for res in results:
    if res is not None:
        metric_result += res
        num += 1

metric_result /= num

print("="*80)
print("融合结果评估（平均值）")
print("="*80)
print("{:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(
    "EN", "SD", "SF", "MI", "VIF", "Qabf", "SCD", "SSIM"
))
print("{:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f}".format(
    metric_result[0], metric_result[1], metric_result[2],
    metric_result[3], metric_result[4], metric_result[5],
    metric_result[6], metric_result[7]
))
