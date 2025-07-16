import gradio as gr
from transcribe import stream_transcribe  # 您自定义的模块
from PIL import Image, ImageDraw, ImageFont  # Pillow library
import io, re 
import numpy as np  # 用于创建示例图像
import os  # 用于处理文件路径和目录
import shutil  # 用于文件复制
import uuid  # 用于生成唯一文件名
from typing import List, Tuple, Optional, Union  # For type hinting
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from tools.test import test_tool
from llm import DashScopeLLM
import json


# --- 配置保存上传图片的目录 ---
UPLOAD_IMAGE_DIR = "user_uploaded_images"
os.makedirs(UPLOAD_IMAGE_DIR, exist_ok=True) # 程序启动时即创建目录

# --- 0. 修改后的 Agent 函数 (现在接收图像路径列表) ---
def multimodal_agent_logic(
    image_path: Optional[str],  # Agent 接收图像文件路径列表
    text_from_speech: Optional[str]
) -> Tuple[Optional[Union[Image.Image, str]], Optional[str]]:
    print(image_path)
    print(text_from_speech)
    """
    模拟多模态 Agent，接收图像文件路径列表和语音文本，
    并动态决定输出图像、文本或两者。

    Args:
        list_of_image_paths (Optional[List[str]]): 保存到本地的图像文件路径列表。
        text_from_speech (str): 从语音转录中提取的文本。

    Returns:
        tuple: (image_data_to_display, text_data_to_display)
               image_data_to_display可以是 PIL图像对象、新图像文件路径或None。
               text_data_to_display可以是字符串或None。
    """
    agent_image_output: Optional[Union[Image.Image, str]] = None
    agent_text_output: Optional[str] = None

    try:
        llm = DashScopeLLM()
    except Exception as e:
        print(f"警告: DashScopeLLM 初始化失败: {e}")

    optic_nerve_tool = FunctionTool.from_defaults(fn=test_tool)
    try:
        agent = ReActAgent.from_tools([optic_nerve_tool], llm=llm, verbose=True)
    except Exception as e:
        print(f"错误: ReActAgent 初始化失败: {e}")
        agent = None
        
    input_query = '你是一个医学影像语音报告系统，接下来我会给你提供一段音频语音识别的内容（医学内容）和输入图像(如果有)，由于语音识别精度问题你需要自己理解内容并给出合适的回复。对于报告生成要求，工具调用会可能会提供检索出相关模板，请你按照最相关的模板内容回答，回答风格精准精炼像一个工具。语音识别结果：'+text_from_speech
    if image_path:
        input_query += f"图像路径:'{image_path}'"
    llm = DashScopeLLM()
    
    tools = [
        {
            'type':'function',
            'function':{
                'name': 'test_tool',
                'description': """
                                处理眼底图像以执行指示性的视神经测量，并检索出相关的报告内容。
                                此工具加载由 'image_path' 指定的图像文件，在与视盘分析相关的特定坐标
                                （例如 (165,423), (240,423), (202,353)）上叠加预定义的绿色十字标记，
                                并添加绿色文本注释，指示一个计算出的度量值（例如 "视盘度量: 1.09 单位"）。
                                它主要用于与视神经乳头特征相关的初步视觉评估或自动化分析流程。
                                """,
                'parameters': {
                    "type": "object",
                    "properties": {
                        "image_path": {
                        "title": "Image Path",
                        "description": "眼底图像的文件路径。图像文件应能被 OpenCV 读取。",
                        "anyOf": [{"type": "string" },{"type": "null"}]
                        }
                    },
                    "required": ["image_path"]
                }
            }
        }
    ]
    
    function_mapper = {
        "test_tool": test_tool
    }
    messages = [
        {
            "role": "user", 
            "content": f"{input_query}"
        }]
    response = llm.chat(messages=messages, tools=tools)
    print(response.raw)
    function_output = None
    if response.raw['choices'][0]['message'].get('tool_calls', None):
        func = response.raw['choices'][0]['message']['tool_calls'][0]
        function_output = function_mapper[func['function']['name']](**json.loads(func['function']['arguments']))
        function_output = function_output if function_output else ''
        messages.append(response.raw['choices'][0]['message'])
        messages.append({"role": "tool", "content": function_output, "tool_call_id": func['id']})
        response = llm.chat(messages=messages)
        print(response.raw)
    if function_output:
        match = re.search(r"已保存至'(.*?)'。", function_output)
        if match:
            agent_image_output =  match.group(1)

    agent_text_output = str(response)
    return agent_image_output, agent_text_output


# 1. 定义核心处理函数 (修改后以保存图片到本地并传递路径)
def process_inputs_for_dynamic_agent_output(
    temp_image_path: Optional[str],  # Gradio传入的临时图片路径列表
    audio_input_path: Optional[str]
) -> Tuple[Optional[Union[Image.Image, str]], Optional[str]]:
    """
    接收Gradio的临时图片路径，将其保存到本地指定文件夹，
    然后将这些新路径传递给Agent。
    """
    saved_image_paths_for_agent: List[str] = []

    if temp_image_path:
        print(f"收到{temp_image_path}")
        try:
            # 尝试获取原始文件扩展名，或从内容判断
            pil_img_temp = Image.open(temp_image_path)
            original_format = pil_img_temp.format
            pil_img_temp.close() # 打开仅为获取格式，然后关闭

            if original_format:
                extension = f".{original_format.lower()}"
            else: # 如果PIL无法确定格式，尝试从路径获取或默认
                _, ext_from_path = os.path.splitext(temp_image_path)
                extension = ext_from_path if ext_from_path else ".png" # 默认.png

            # 创建一个新的唯一文件名并保存到指定目录
            unique_filename = f"{uuid.uuid4()}{extension}"
            destination_path = os.path.join(UPLOAD_IMAGE_DIR, unique_filename)
            
            shutil.copy(temp_image_path, destination_path) # 从临时路径复制到目标路径
            saved_image_paths_for_agent.append(destination_path)
            print(f"图片已从Gradio临时位置复制并保存到: {destination_path}")
        except Exception as e:
            print(f"处理/保存来自 {temp_image_path} 的图片时出错: {e}")

    else:
        print("没有图片被上传或Gradio未提供临时路径。")

    # --- 处理语音输入 (逻辑与之前相同) ---
    transcribed_text = "语音未识别"
    if audio_input_path is not None:
        print(f"尝试转录音频: {audio_input_path}")
        try:
            transcribed_text_result = stream_transcribe(audio_input_path)
            if transcribed_text_result is not None and transcribed_text_result.strip():
                transcribed_text = transcribed_text_result
            else:
                transcribed_text = "语音识别为空或失败"
            print(f"语音识别结果: {transcribed_text}")
        except Exception as e:
            print(f"调用 stream_transcribe 时发生错误: {e}")
            transcribed_text = f"语音识别错误: {str(e)}"
    else:
        transcribed_text = "未检测到语音输入"
        print("No audio input detected.")

    # --- 调用 Agent, 传递保存后的图像路径列表 ---
    agent_image_data, agent_text_data = multimodal_agent_logic(
        saved_image_paths_for_agent[0] if saved_image_paths_for_agent[0] else None, # 传递新路径列表
        transcribed_text if transcribed_text else None
    )
    
    return agent_image_data, agent_text_data

if __name__ == "__main__":
    image_component_input = gr.Image(
        type="filepath",  # <--- 重要更改: Gradio将提供文件路径
        label="🖼️ 上传图片 (Upload Image) "
    )
    audio_component_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="🎤 语音输入 (Speak or Upload Audio)")

    output_image_display = gr.Image(label="🖼️ Agent - 图像输出 (Image Output)")
    output_text_display = gr.Textbox(label="💬 Agent - 文本输出 (Text Output)", lines=15, interactive=True)

    description = """
    这是一个yuy演示界面：
    1.  上传图片。图片将被保存到服务器本地的 `user_uploaded_images` 文件夹。
    2.  通过麦克风说话或上传一个音频文件。
    3.  Agent 将接收保存后的图片路径和语音，并根据指令动态输出图像或文字。
    """

    iface = gr.Interface(
        fn=process_inputs_for_dynamic_agent_output,
        inputs=[image_component_input, audio_component_input],
        outputs=[output_image_display, output_text_display],
        title="✨ 多模态 Agent ✨",
        description=description,
        allow_flagging="never",
        examples=[
            [None, None],
        ]
    )
    print(f"Gradio UI 即将启动... 上传的图片将保存到 '{os.path.abspath(UPLOAD_IMAGE_DIR)}' 文件夹。")
    iface.launch()