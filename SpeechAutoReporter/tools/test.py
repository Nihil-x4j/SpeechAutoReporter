'''
测试agent时调用的工具集
'''

from typing import List, Tuple, Optional, Union
import uuid
import cv2
import numpy as np
import os
import math
import random
import re
from tools.VectorDB import VectorDBManager



def test_tool(image_path: Optional[str]):
    """
    处理眼底图像以执行指示性的视神经测量。

    此工具加载由 'image_path' 指定的图像文件，在与视盘分析相关的特定坐标
    （例如 (165,423), (240,423), (202,353)）上叠加预定义的绿色十字标记，
    并添加绿色文本注释，指示一个计算出的度量值（例如 "视盘度量: 1.09 单位"）。
    它主要用于与视神经乳头特征相关的初步视觉评估或自动化分析流程。

    参数:
        image_path (str): 眼底图像的文件路径。
                          图像文件应能被 OpenCV 读取。

    返回:
        Optional[np.ndarray]: 如果图像处理成功，则返回一个包含叠加标记和文本的 NumPy 数组格式的图像。
                              如果无法加载图像或者 image_path 为 None，则返回 None。
    """
    def find_key_coordinates(input_str):
        dict_ = {
            '1': [(230,299),(302,299),(268,263)],
            '2': [(241,308),(309,308),(274,272)],   
            '3': [(248,306),(309,306),(280,270)],   
            '4': [(266,305),(332,305),(300,269)],   
            '5': [(257,311),(327,311),(293,275)],   
            '0': [(230,299),(302,299),(293,275)],   
        }
        for key in dict_:
            if key != '0' and key in input_str:
                return dict_[key]
        return dict_.get('0', [])  # 默认返回 '0' 对应的坐标列表
    
    def calculate_onsd(left_point, right_point, pixel_to_mm_ratio=1/70):
        """
        计算视神经鞘直径（ONSD）
        
        参数:
            left_point (tuple): 左边界点坐标 (x, y)
            right_point (tuple): 右边界点坐标 (x, y)
            pixel_to_mm_ratio (float): 每像素对应的实际长度（毫米/像素），默认为0.1
        返回:
            pixel_distance (float): 像素距离
            mm_distance (float): 实际长度（毫米）
        """
        x1, y1 = left_point
        x2, y2 = right_point
        
        # 计算欧几里得距离
        pixel_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        mm_distance = pixel_distance * pixel_to_mm_ratio
        
        return mm_distance
    print(image_path)
    TOOL_PROCESSED_IMAGE_DIR = r'C:\Users\94373\Desktop\RAG\SpeechAutoReporter\tools\pictures'
    
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # --- Parameters for drawing ---
    cross_color = (0, 255, 0)  # Green in BGR format
    text_color = (0, 255, 0)   # Green in BGR format
    cross_size = 15           # Size of the arms of the cross
    thickness = 2             # Thickness of the lines
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    text_position = (10, 30)  # (x, y) from top-left

    # --- Draw green cross markers ---
    points = find_key_coordinates(os.path.basename(image_path))
    for x,y in points:
        # Horizontal line
        cv2.line(image, (x - cross_size, y), (x + cross_size, y), cross_color, thickness)
        # Vertical line
        cv2.line(image, (x, y - cross_size), (x, y + cross_size), cross_color, thickness)
    distance_info = f"Distance: {calculate_onsd(points[0],points[1])} mm。"
    # --- Add green distance description ---
    cv2.putText(image, distance_info, text_position, font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    try:
        original_filename_base = os.path.splitext(os.path.basename(image_path))[0]
        unique_suffix = uuid.uuid4().hex[:8] # 短一点的唯一标识符
        # 使用原始文件名基础，加上 "_processed_" 和唯一后缀
        output_filename = f"{original_filename_base}_processed_{unique_suffix}.png" # 固定保存为png
        saved_path = os.path.join(TOOL_PROCESSED_IMAGE_DIR, output_filename)
        
        cv2.imwrite(saved_path, image)
        print(f"工具 'test_tool' 已将处理后的图像保存到: {saved_path}")
        return f"图像处理成功，已保存至'{saved_path}'。视神经鞘直径测量结果为{calculate_onsd(points[0],points[1])} mm"
    except Exception as e:
        print(f"错误：在 test_tool 中保存图像失败: {e}")
        return f"图像处理遇到问题（保存失败）。（错误：{e}）"

def reporter_tool(text):
    db_manager = VectorDBManager('C:\\Users\\94373\\Desktop\\RAG\\chroma_db')
    retrieved_results = db_manager.retrieve(
            collection_name="disease_templates",
            query=f'{text}',
            top_k=5
        )
    print(f"语音识别成功,检索到相关疾病模板'{retrieved_results}'")
    return f"语音识别成功,检索到相关疾病模板'{retrieved_results}'"

def get_plaque_dimensions():
    """
    获取并返回指定动脉粥样硬化斑块的二维尺寸。
    此函数用于提取斑块在长度和厚度两个维度上的具体测量结果。

    :return: 一个表示斑块大小的字符串，格式为 '长度mm×厚度mm' (例如 '8.3mm×2.1mm')。
    :rtype: str
    """
    length = round(random.uniform(2.0, 15.0), 1)  # 斑块长度，单位mm
    thickness = round(random.uniform(1.0, 4.5), 1) # 斑块厚度，单位mm
    return f"{length}mm×{thickness}mm"

def get_stenotic_lumen_diameters():
    """
    测量并返回目标狭窄血管段的原始管径及狭窄后的残余管径。
    这些数据用于精确评估血管的狭窄程度。

    :return: 一个包含两个浮点数的元组: (原始管径_mm, 残余管径_mm)，单位均为毫米。
             例如 (6.5, 2.1)，表示原始管径6.5mm，狭窄后残余管径2.1mm。
    :rtype: tuple[float, float]
    """
    original_diameter = round(random.uniform(3.0, 8.0), 1) # 原始管径，单位mm

    # 假设狭窄导致管径减少10%到80%
    stenosis_percentage_by_diameter = random.uniform(0.10, 0.80)
    residual_diameter = round(original_diameter * (1 - stenosis_percentage_by_diameter), 1)

    # 确保残余管径小于原始管径且为正值
    if residual_diameter >= original_diameter:
        residual_diameter = round(original_diameter * 0.85, 1)
    if residual_diameter <= 0.1:
        residual_diameter = 0.1
    if residual_diameter >= original_diameter and original_diameter > 0.1:
        residual_diameter = original_diameter - 0.1
    elif residual_diameter >= original_diameter and original_diameter <= 0.1:
        residual_diameter = original_diameter * 0.5

    return original_diameter, residual_diameter

def get_flow_velocities_at_stenosis():
    """
    测量并返回指定动脉狭窄处的峰值收缩期流速 (PSV) 以及狭窄远端特定位置的血流速度。
    这些流速数据对于评估狭窄对血流动力学的影响至关重要。

    :return: 一个包含两个浮点数的元组: (狭窄处峰值流速_cms, 狭窄远段流速_cms)，单位均为厘米/秒 (cm/s)。
             例如 (180.5, 45.2)。
    :rtype: tuple[float, float]
    """
    stenosis_psv = round(random.uniform(100.0, 350.0), 1)  # cm/s

    distal_psv = round(random.uniform(20.0, max(30.0, stenosis_psv * 0.7 - 20)), 1)
    if distal_psv >= stenosis_psv:
        distal_psv = round(stenosis_psv * 0.6, 1)
    if distal_psv < 15.0 and stenosis_psv > 50 :
        distal_psv = round(random.uniform(15.0, 30.0),1)

    return stenosis_psv, distal_psv

def get_common_carotid_artery_psv():
    """
    获取并返回指定颈总动脉段的峰值收缩期血流速度 (PSV)。
    此函数用于评估该血管段的基础血流动力学状态。

    :return: 一个浮点数，表示峰值收缩期流速，单位为厘米/秒 (cm/s)。例如 75.6。
    :rtype: float
    """
    normal_psv = round(random.uniform(60.0, 100.0), 1) # 颈总动脉正常PSV范围参考值
    return normal_psv


if __name__ == "__main__":

    dummy_image_path = "/root/autodl-tmp/RAG/SpeechAutoReporter/tools/5.jpeg"
    output_dir = '/root/autodl-tmp/RAG/SpeechAutoReporter/tools/pictures'
    os.makedirs(output_dir, exist_ok=True)

    result_success = test_tool(dummy_image_path)
    print(f"Result: {result_success}")