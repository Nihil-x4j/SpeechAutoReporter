'''
符合llama_index格式的llm对话类，因为api本身的输入输出不符合llamaindex框架要求，需要重新实现自己的llm类
详细内容太多了，需要参考llama_index官网的函数说明文档
https://llama-index.readthedocs.io/zh/latest/index.html
https://docs.llamaindex.org.cn/en/stable/getting_started/installation/
'''

import requests #用于发出网络请求的库
import json #用于处理json相关内容
from copy import deepcopy #用于深拷贝
from pprint import pprint #用于更好的打印，pprint会根据变量内容自动换行
from typing import Optional, List, Any, Sequence, Generator #用于变量类型约束

from llama_index.core.llms import ( # llamaindex的核心库，这里面的内容比较复杂，一般按照llamaindex官方给出的参考对话内容就行，不需要狠扣细节
    CustomLLM, # 定义用户自己的llm
    CompletionResponse, # llm要求的Completion返回格式
    CompletionResponseGen, # llm要求的流式输出返回格式
    LLMMetadata, # llm的元数据
    ChatMessage, # 同上
    ChatResponse,
    ChatResponseGen,
    MessageRole,
)
from llama_index.core.llms.callbacks import llm_completion_callback, llm_chat_callback# llamaindex的库
from llama_index.core.tools.types import ToolMetadata# llamaindex的库
from tools.test import *

url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
api_key = "sk-7ba7379293084b40bd1d50d03fa71af5" # 请替换为您的有效 API 密钥

# api请求头，加在request中
headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# 常用llm参数模板
base_payload_template = {
    "messages": [], # 和模型对话的消息
    "temperature": 0.0, # 模型温度，越高输出随机性越强
    "top_p": 0.5, #输出参数，越高输出随机性越强
    "max_tokens": 1000, #输出最大的token数量
    "repetition_penalty": 1.1, #输出内容重复性惩罚，一般在0到2，避免输出重复内容，值越高惩罚越强
    "enable_thinking": False #这个是通义千问平台特定的参数，是否启用思考模式，但是仅对部分模型有效，需要参考百炼平台文档
}

# 使用百炼api封装的llm类
class DashScopeLLM(CustomLLM):
    # "这些参数有很多是为了符合llamaindex设置的，实际上可能并不起作用，因为我们重新封装了阿里百炼平台的api，其中没有这些参数，chat部分中实际传入的参数才是api支持的参数"
    context_window: int = 4000 
    num_output: int = 512 
    model_name: str = ""
    is_chat_model: bool = True
    temperature: float = 0
    max_tokens_generation: int = 1000 

    def __init__(
        self,
        model: str = "qwen3-235b-a22b", #这里可以切换模型，但是百炼平台的模型存在很多问题，可能需要修改返回格式，有时候不同模型返回格式不统一。
        temperature: float = 0.1,
        max_tokens: int = 1000,
        context_window: int = 4000,
        num_output: int = 512,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model
        self.temperature = temperature
        self.max_tokens_generation = max_tokens
        self.context_window = context_window
        self.num_output = num_output

    @property #llamaindex要求的内容，复制即可
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
            is_chat_model=self.is_chat_model,
        )

    @llm_chat_callback() #需要自己实现的内容，把先前定义的模板中深度拷贝，然后替换内容
    def chat(self, messages, **kwargs: Any) -> ChatResponse:
        payload = deepcopy(base_payload_template)
        payload['model'] = self.model_name#请求的模型名称
        payload['messages'] = messages#请求的具体内容
        payload['temperature'] = self.temperature#模型输出的温度
        payload['max_tokens'] = self.max_tokens_generation#模型输出的最大长度
        tools = kwargs.get("tools")
        
        if tools:
            payload["tools"] = tools

        
        # 上面这些内容才是实际被传入的参数

        try:
            #pprint(payload)
            response = requests.post(url, headers=headers, json=payload)# 使用requests方法调用百炼api平台模型
            response_json = response.json()# 把response转成json格式用于读取
            pprint(response_json)# 查看返回内容
            response.raise_for_status()# 查看返回状态码
        except requests.exceptions.RequestException as e:
            raise ValueError(f"API request failed: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse API response as JSON: {e}. Response text: {response.text}")

        if not response_json.get("choices"):
            raise ValueError(f"API response missing 'choices' field: {response_json}")
        
        choice = response_json['choices'][0]
        message_data = choice['message']#这两句话用于提取api返回的模型输出内容

        assistant_message = ChatMessage(
            role=MessageRole.ASSISTANT, # 或者从响应中确定角色
            content=message_data.get('content'), # content中都是文本内容，而不是字典
        )

        return ChatResponse(message=assistant_message, raw=response_json)#返回封装好的信息，可以直接获取文本内容(message部分)，也可以获得llm完整回复raw，我们一般使用xxx.raw直接查看原始内容，因为其中可能包括工具调用请求

    # callback函数暂时没有研究明白是什么意思，暂时找着官网案例进行写了，这部分可能和llamaindex自带的ReAct框架有关系
    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        # 'complete' 不传递 tools, tool_choice, parallel_tool_calls
        chat_response = self.chat(messages, **kwargs) 
        
        text_content = ""
        if chat_response.message and chat_response.message.content:
            text_content = chat_response.message.content
        
        if chat_response.message and chat_response.message.additional_kwargs.get("tool_calls"):
            print("Warning: 'complete' method received tool_calls. These will be ignored for CompletionResponse.")

        return CompletionResponse(text=text_content, raw=chat_response.raw)
    
    # 流式输出为空，暂时不需要实现
    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """
        This LLM does not support streaming completion.
        """
        raise NotImplementedError(
            "This LLM (DashScopeLLM) does not support streaming completion."
        )
        
    # 流式输出为空，暂时不需要实现
    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """
        This LLM does not support streaming chat.
        """
        raise NotImplementedError(
            "This LLM (DashScopeLLM) does not support streaming chat."
        )

# 下面是对上面定义llm方法的测试
if __name__ == "__main__":
    from tools.test import *
    llm = DashScopeLLM()
    input_query = '''
你是一个医学影像语音报告系统助手。您的核心任务是全面理解用户的请求，并利用您所掌握的一系列可用工具来达成目标，最终生成一个完整且准确的医学报告。

我希望你的工作流程：
1.先根据用户输入提示调用工具检索相关模板。
2.根据模板检索结果分析对于给定模板缺失哪些项（描述或者测量值）
3.判断描述或测量值能不能调用其他工具进行补充。
4.重复2步骤直至完善模板

重要要求：你生成的回复一定要严格基于检索到的病例模板，不要输出额外的内容，因为你的输出结果将作为pdf被打印。对于模板中缺失的数据，可以使用“未提及”“未测量”填充（如果实在找不到获取不到这些信息）

用户提供信息："双侧颈总动脉管径对称，内中膜不均匀性增厚。左侧内侧壁探及回声斑块，各段血流速度正常。双侧颈内动脉管径对称，内中膜不均匀性增厚，后内侧探及规则型回声斑块，表面纤维帽结构完整，彩色血流充盈完整，各段血流速度正常。双侧颈外动脉血流未见异常。双侧椎动脉管径对称，血流速度正常。"
图片路径:"D://dd//test.jpg"
'''
    #input_query = "帮我先测量视神经直径，再检索生成报告：双侧颈总动脉管径对称，内中膜不均匀性增厚。左侧内侧壁探及回声斑块，各段血流速度正常。双侧颈内动脉管径对称，内中膜不均匀性增厚，后内侧探及规则型回声斑块，表面纤维帽结构完整，彩色血流充盈完整，各段血流速度正常。双侧颈外动脉血流未见异常。双侧椎动脉管径对称，血流速度正常。"
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
        # 新增的函数定义开始
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
        }
    ]
    
    function_mapper = {
        "test_tool": test_tool,
        "reporter_tool": reporter_tool,
        "get_plaque_dimensions": get_plaque_dimensions
    }
    
    messages = [
        {
            "role": "user", 
            "content": f"{input_query}"
        }]
    response = llm.chat(messages=messages, tools=tools)
    
    while response.raw['choices'][0]['message'].get('tool_calls', None):
        func = response.raw['choices'][0]['message']['tool_calls'][0]
        function_output = function_mapper[func['function']['name']](**json.loads(func['function']['arguments']))
        function_output = function_output if function_output else ''
        messages.append(response.raw['choices'][0]['message'])
        messages.append({"role": "tool", "content": function_output, "tool_call_id": func['id']})
        messages.append({"role": "user", "content": '结合现有信息判断是否需要进一步的工具调用', "tool_call_id": func['id']})
        response = llm.chat(messages=messages, tools=tools)