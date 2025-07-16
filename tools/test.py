from typing import List, Tuple, Optional, Union
import uuid
import cv2
import numpy as np
import os
import re
from VectorDB import VectorDBManager

def test_tool(image_path: Optional[str]):
    print(image_path)
    TOOL_PROCESSED_IMAGE_DIR = '/root/autodl-tmp/RAG/SpeechAutoReporter/tools/pictures'
    """
    处理眼底图像以执行指示性的视神经测量,并从数据库中检索相关疾病报告模板。

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
    for (x, y) in [
        (165,423), 
        (240,423),  
        (202,353)   
    ]:
        # Horizontal line
        cv2.line(image, (x - cross_size, y), (x + cross_size, y), cross_color, thickness)
        # Vertical line
        cv2.line(image, (x, y - cross_size), (x, y + cross_size), cross_color, thickness)
    distance_info = f"Distance: {76/70} mm。"
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
        db_manager = VectorDBManager('/root/autodl-tmp/RAG/eval_chroma_db')
        retrieved_results = db_manager.retrieve(
                collection_name="disease_templates",
                query=f'所见眼球上部视神经鞘，使用工具测量直径结果为{76/70} mm',
                top_k=5
            )
        return f"图像处理成功，已保存至'{saved_path}'。视神经鞘直径测量结果为{76/70} mm,检索到相关模板'{retrieved_results}'"
    except Exception as e:
        print(f"错误：在 test_tool 中保存图像失败: {e}")
        return f"图像处理遇到问题（保存失败）。（错误：{e}）"

if __name__ == "__main__":

    dummy_image_path = "/root/autodl-tmp/RAG/SpeechAutoReporter/tools/test.jpeg"
    output_dir = '/root/autodl-tmp/RAG/SpeechAutoReporter/tools/pictures'
    os.makedirs(output_dir, exist_ok=True)

    result_success = test_tool(dummy_image_path)
    print(f"Result: {result_success}")