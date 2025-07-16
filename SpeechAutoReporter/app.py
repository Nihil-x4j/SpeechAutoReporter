import gradio as gr #用于构建简单的可视化页面

from PIL import Image, ImageDraw, ImageFont  # Pillow library
import io, re #io是输入输出库，re是正则表达式库
import numpy as np  # 用于创建示例图像
import os  # 用于处理文件路径和目录
import shutil  # 用于文件复制
import uuid  # 用于生成唯一文件名
from typing import List, Tuple, Optional, Union  # 用于规范化变量格式
#from llama_index.core.tools import FunctionTool # llamaindex调用工具的定义
#from llama_index.core.agent import ReActAgent # llamaindex自带的工具调用推理框架
from tools.test import * # 从tools.test中找到有用的工具
from llm import DashScopeLLM # 从llm找到DashScopeLLM的定义
import json
from pprint import pprint
from tools.transcribe import stream_transcribe


# --- 配置保存上传图片的目录 ---
UPLOAD_IMAGE_DIR = "user_uploaded_images"
os.makedirs(UPLOAD_IMAGE_DIR, exist_ok=True) # 程序启动时即创建目录

# --- 多模态agent实现 ---
def multimodal_agent_logic(
    image_path: Optional[str],  # Agent 接收图像文件路径列表
    text_from_speech: Optional[str] # 从语言中识别到的文本
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

    # 这部分本身可以用llamaindex实现，但是llm是自定义的，存在一些部分不符合格式要求，下面使用手动实现
    # optic_nerve_tool = FunctionTool.from_defaults(fn=test_tool)
    # try:
    #     agent = ReActAgent.from_tools([optic_nerve_tool], llm=llm, verbose=True)
    # except Exception as e:
    #     print(f"错误: ReActAgent 初始化失败: {e}")
    #     agent = None
    
    # 这个是整个agent应用中重要的提示词部分，需要让llm理解怎么进行多轮推理，同时规范输入输出格式，但是这部分经常会失效，需要结合llm的参数和prompt内容设置
    input_query = f'''
        你是一个医学影像语音报告系统，接下来我会给你提供一段音频语音识别的内容（医学内容）和输入图像(如果有)，你需要给我返回相应的完整病例报告。有以下要求：
        1. 你需要根据语音识别内容使用相关工具检索出相关病例模板，请一定不能把语言识别内容原封不动输入工具。
            重要！这一步至关重要，如果检索到错误模板，后续将全错。请你一定认真纠错，主要从多音字方面考虑，语言识别大部分错误源自多音字,比如炎症等。
            1.1 语言识别结果可能有比较高的错误率，首先你需要对语言识别到的信息做纠正和提炼。
            1.2 完成内容纠错后输入相关工具进行检索
        2. 你会得到若干模板，选择内容最相关的一个完善病例模板内容。
            2.1 你需要分析模板中哪些信息存在缺失？能否调用提供的工具进行补充？
            2.2 如果不需要补充，直接按照检索到的模板填充后输出。
            2.3 如果工具无法补充关键信息，请对关键信息处使用“未提供”填充，然后输出
        3. 更新现有信息并重复步骤2，直到模板信息完全补全或存在无法获取的信息。
        补充：最后输出时需要你严格遵守模板内容，并将最终返回的病例报告结果使用<final></final>包裹方便我解析。

        返回内容请一定尽可能严格参照模板！，包括格式！，除非存在检索错误（逻辑冲突）或大量错误情况你可以适当推测，对于出现的所有测量内容，严禁编造！工具无法获取就使用‘未提供’占位。
        语音识别文本（可能为空）:'{text_from_speech}'
        图像路径（可能为空）:'{image_path}'
    ''' 

    llm = DashScopeLLM()
    
    # 这里需要传入需要使用的工具定义，llm会根据工具的参数描述去自动选择，这里的json格式要求参考百炼api要求
    # https://bailian.console.aliyun.com/?tab=api#/api/?type=model&url=https%3A%2F%2Fhelp.aliyun.com%2Fdocument_detail%2F2712576.html
    tools = [
        {
            'type':'function',
            'function':{
                'name': 'test_tool',
                'description': """
                                处理眼底图像以执行指示性的视神经测量。
                                此工具加载由 'image_path' 指定的图像文件，在与视盘分析相关的特定坐标
                                （例如 (165,423), (240,423), (202,353)）上叠加预定义的绿色十字标记，
                                并添加绿色文本注释，指示一个计算出的度量值（例如 "视盘度量: 1.09 单位"）。
                                它主要用于与视神经乳头特征相关的初步视觉评估或自动化分析流程。<tool></tool>
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
        },
        {
            'type':'function',
            'function':{
                'name': 'reporter_tool',
                'description': """
                                根据文本内容从疾病数据库中检索出相关的报告模板。
                                此工具加载由语音识别文本结果并检索出最相关的几个文档。
                                """,
                'parameters': {
                    "type": "object",
                    "properties": {
                        "text": {
                        "title": "文本内容",
                        "description": "用户输入的语音识别文本字符串。",
                        "type": "string"
                        }
                    },
                    "required": ["text"]
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'get_plaque_dimensions',
                'description': """获取并返回斑块的二维尺寸。
    此函数用于提取斑块在长度和厚度两个维度上的具体测量结果。""",
                'parameters': {
                    'type': 'object',
                    'properties': {}, # 此函数无参数
                    'required': []
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'get_stenotic_lumen_diameters',
                'description': """测量并返回目标狭窄血管段的原始管径及狭窄后的残余管径。
    这些数据用于精确评估血管的狭窄程度。""",
                'parameters': {
                    'type': 'object',
                    'properties': {}, # 此函数无参数
                    'required': []
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'get_flow_velocities_at_stenosis',
                'description': """测量并返回指定动脉狭窄处的峰值收缩期流速 (PSV) 以及狭窄远端特定位置的血流速度。
    这些流速数据对于评估狭窄对血流动力学的影响至关重要。""",
                'parameters': {
                    'type': 'object',
                    'properties': {}, # 此函数无参数
                    'required': []
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'get_common_carotid_artery_psv',
                'description': """获取并返回指定颈总动脉段的峰值收缩期血流速度（PSV）。
    此函数用于评估该血管段的基础血流动力学状态。""",
                'parameters': {
                    'type': 'object',
                    'properties': {}, # 此函数无参数
                    'required': []
                }
            }
        }
        # 新增的函数定义结束
    ]
    
    # 定义调用的工具函数映射，因为llm调用工具时会给出调用的工具名称，此处可以通过变量名找到实际的函数
    function_mapper = {
        "test_tool": test_tool,
        "reporter_tool": reporter_tool,
        "get_plaque_dimensions": get_plaque_dimensions,
        "get_stenotic_lumen_diameters": get_stenotic_lumen_diameters,
        "get_flow_velocities_at_stenosis": get_flow_velocities_at_stenosis,
        "get_common_carotid_artery_psv": get_common_carotid_artery_psv
    }
    # 用户对话的内容，这是最开头的一条，来自用户的提问和要求
    messages = [
        {
            "role": "user",  # 设置角色
            "content": f"{input_query}" # 设置角色的对话内容
        }]
    
    '''
    下面进入手动agent的循环部分，目前来说，没法做到自动化任务编排，可能是因为提示词和选用模型的工具调用能力存在问题，进一步改进可以首先优化提示词。
    '''
    response = llm.chat(messages=messages, tools=tools) #第一次对话部分
    print(response.raw) # 查看一下大模型的输出，用来监督回答是否正确
    function_output = None # 工具输出的内容
    agent_text_output = '' # 可视化输出的文本内容
    # 下面是agent的结束条件，如果回答中出现了<final></final>标签，说明agent流程结束，不在进行对话
    while '<final>' not in response.raw['choices'][0]['message']['content'] or '</final>' not in response.raw['choices'][0]['message']['content']:
        # 回答中出现工具调用部分就使用对于的工具，同时添加对应的调用结果进对话历史记录中
        # 有时候response.raw['choices'][0]['message']不一定正确，因为不同api平台返回内容位置不同，这里需要根据response.raw内容查看具体把那部分内容提取
        if response.raw['choices'][0]['message'].get('tool_calls', None): # 从llm的回复中查看是否有工具调用的请求
            func = response.raw['choices'][0]['message']['tool_calls'][0] # 获取调用工具的名字
            agent_text_output+=f"调用工具{func['function']['name']}\n" # 添加进要输出的文本列表
            function_output = function_mapper[func['function']['name']](**json.loads(func['function']['arguments'])) # 在之前定义的工具字典中找到llm请求的工具，然后传入llm返回中工具调用的参数
            function_output = function_output if function_output else ''
            messages.append(response.raw['choices'][0]['message']) # 在对话中添加llm输出的内容
            messages.append({"role": "tool", "content": function_output, "tool_call_id": func['id']}) # 在对话中添加工具调用的结果，这里role设置成"tool"
            messages.append({"role": "user", "content": '结合现有信息判断是否需要进一步的工具调用'}) # 这里似乎和qwen的训练过程有关系，需要额外的user请求，而且需要再次传递工具集的描述"tool_call_id"
            '''
            这里需要进行说明，一般来说，llm对话时用户可以不进行打断，比如这个对话序列：user->assistant->tool->assistant->...
            但是实际使用时发现，qwen模型的回复经常需要用户确认打断agent流程，因此需要更新序列为：user->assistant->tool->user->assistant->...
            所以需要这个步骤，messages.append({"role": "user", "content": '结合现有信息判断是否需要进一步的工具调用', "tool_call_id": func['id']})
            如果去掉这一行，模型只会调用一次工具
            '''
            response = llm.chat(messages=messages, tools=tools) # 下一轮llm对话
        else :
            '''
            此处同理，需要加上一段user对话才能让llm真正结束agent流程。
            '''
            messages.append(response.raw['choices'][0]['message'])
            messages.append({"role": "user", "content": '请继续,如果你认为已经完成，请输出包含<final>标签的报告，我将视为已结束'})
            response = llm.chat(messages=messages, tools=tools)
    # 这个是为了可视化图片，为了适配标注工具，这里只是为了demo演示做的非常简单，实际上应该约定一下各类工具的返回格式。
    if function_output:
        match = re.search(r"已保存至'(.*?)'。", function_output)
        if match:
            agent_image_output =  match.group(1)

    agent_text_output += str(response)
    return agent_image_output, agent_text_output


# 1. 定义核心处理函数 (修改后以保存图片到本地并传递路径)
def process_inputs_for_dynamic_agent_output(
    temp_image_path: Optional[str],  # Gradio传入的临时图片路径列表
    audio_input_path: Optional[str] # Gradio传入的音频图片路径列表
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

            # if original_format:
            #     extension = f".{original_format.lower()}"
            # else: # 如果PIL无法确定格式，尝试从路径获取或默认
            #     _, ext_from_path = os.path.splitext(temp_image_path)
            #     extension = ext_from_path if ext_from_path else ".png" # 默认.png

            # 创建一个新的唯一文件名并保存到指定目录
            #unique_filename = f"{uuid.uuid4()}{extension}"
            #destination_path = os.path.join(UPLOAD_IMAGE_DIR, unique_filename)
            destination_path = os.path.join(UPLOAD_IMAGE_DIR, os.path.basename(temp_image_path)) #定义了上传的图片保存的路径
            shutil.copy(temp_image_path, destination_path) # 从临时路径复制到目标路径
            saved_image_paths_for_agent.append(destination_path)
            print(f"图片已从Gradio临时位置复制并保存到: {destination_path}")
        except Exception as e:
            print(f"处理/保存来自 {temp_image_path} 的图片时出错: {e}")

    else:
        saved_image_paths_for_agent = []
        print("没有图片被上传或Gradio未提供临时路径。")

    # --- 处理语音输入 (逻辑与之前相同) ---
    transcribed_text = "语音未识别"
    if audio_input_path is not None:
        print(f"尝试转录音频: {audio_input_path}")
        try:
            transcribed_text_result = stream_transcribe(audio_input_path) # 把语言输入转换成识别的文本
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
    # 把第一张图片和识别的图像文本传入multimodal_agent_logic，让llm输出处理后的图像和最终输出的文本内容
    agent_image_data, agent_text_data = multimodal_agent_logic(
        saved_image_paths_for_agent[0] if saved_image_paths_for_agent else None, # 传递新路径列表
        transcribed_text if transcribed_text else None
    )
    
    return agent_image_data, agent_text_data

if __name__ == "__main__":
    """
    为了简单演示，前端可视化界面直接使用gradio搭建，gradio提供了默认模板，构建非常简单
    """
    # 定义图像上传组件
    image_component_input = gr.Image(
        type="filepath",  # <--- 重要更改: Gradio将提供文件路径
        label="🖼️ 上传图片 (Upload Image) "
    )
    # 定义音频组件
    audio_component_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="🎤 语音输入 (Speak or Upload Audio)")
    # 定义输出显示框
    output_image_display = gr.Image(label="🖼️ Agent - 图像输出 (Image Output)")
    output_text_display = gr.Textbox(label="💬 Agent - 文本输出 (Text Output)", lines=15, interactive=True)
    # 界面描述
    description = """
    这是一个yuy演示界面：
    1.  上传图片。图片将被保存到服务器本地的 `user_uploaded_images` 文件夹。
    2.  通过麦克风说话或上传一个音频文件。
    3.  Agent 将接收保存后的图片路径和语音，并根据指令动态输出图像或文字。
    """
    # gadio的用法，iface应该是一个网页容器，设置好组件和对应的函数就可以直接启动了
    iface = gr.Interface(
        fn=process_inputs_for_dynamic_agent_output, # 处理各种输入和核心函数，下面的都是一些页面上显示的东西
        inputs=[image_component_input, audio_component_input], # 前面定义的两个输入内容
        outputs=[output_image_display, output_text_display], # 前面定义的两个输出内容
        title="✨ 多模态 Agent ✨", # 网页标题
        description=description, # 网页描述
        allow_flagging="never", # 用户是否标记输出内容，暂时没理解什么意思
        examples=[
            [None, None], # 默认网页显示的内容
        ]
    )
    print(f"Gradio UI 即将启动... 上传的图片将保存到 '{os.path.abspath(UPLOAD_IMAGE_DIR)}' 文件夹。")
    iface.launch(server_port=7855) # 启动服务器
    #iface.launch()