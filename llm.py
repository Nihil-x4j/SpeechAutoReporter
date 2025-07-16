import requests
import json
from copy import deepcopy
from pprint import pprint
from typing import Optional, List, Any, Sequence, Generator

from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    MessageRole,
)
from llama_index.core.llms.callbacks import llm_completion_callback, llm_chat_callback
from llama_index.core.tools.types import ToolMetadata

# --- 您之前的设置保持不变 ---
url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
api_key = "sk-7ba7379293084b40bd1d50d03fa71af5" # 请替换为您的有效 API 密钥

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

base_payload_template = {
    "messages": [],
    "temperature": 0.7,
    "top_p": 0.5,
    "max_tokens": 1000,
    "repetition_penalty": 1.1
}
# --- 结束设置 ---

class DashScopeLLM(CustomLLM):
    context_window: int = 4000
    num_output: int = 512
    model_name: str = "qwen-plus-latest"
    is_chat_model: bool = True
    temperature: float = 0.7
    max_tokens_generation: int = 1000 

    def __init__(
        self,
        model: str = "qwen-plus-latest",
        temperature: float = 0.7,
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

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
            is_chat_model=self.is_chat_model,
        )

    @llm_chat_callback()
    def chat(self, messages, **kwargs: Any) -> ChatResponse:
        payload = deepcopy(base_payload_template)
        payload['model'] = self.model_name
        payload['messages'] = messages
        payload['temperature'] = self.temperature
        payload['max_tokens'] = self.max_tokens_generation

        tools = kwargs.get("tools")
        if tools:
            payload["tools"] = tools

        try:
            #pprint(payload)
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            response_json = response.json()
        except requests.exceptions.RequestException as e:
            raise ValueError(f"API request failed: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse API response as JSON: {e}. Response text: {response.text}")

        if not response_json.get("choices"):
            raise ValueError(f"API response missing 'choices' field: {response_json}")
        
        choice = response_json['choices'][0]
        message_data = choice['message']

        assistant_message = ChatMessage(
            role=MessageRole.ASSISTANT, # 或者从响应中确定角色
            content=message_data.get('content'), # 如果有工具调用，content 可能为 null
        )

        return ChatResponse(message=assistant_message, raw=response_json)

    # complete 和 stream_complete 方法保持不变，它们不直接处理工具调用参数
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

if __name__ == "__main__":
    from tools.test import test_tool
    llm = DashScopeLLM()
    input_query = '你是一个医学影像语音报告系统助手，接下来我会给你提供一段音频语音识别的内容和输入图像(如果有)，请你给出合适的回复。文本内容："测量图像",图片地址:"/root/autodl-tmp/RAG/SpeechAutoReporter/tools/test.jpeg"}'
    #input_query = 'nishishui'
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
    pprint(response.raw)
    if response.raw['choices'][0]['message'].get('tool_calls', None):
        func = response.raw['choices'][0]['message']['tool_calls'][0]
        function_output = function_mapper[func['function']['name']](**json.loads(func['function']['arguments']))
        function_output = function_output if function_output else ''
        messages.append(response.raw['choices'][0]['message'])
        messages.append({"role": "tool", "content": function_output, "tool_call_id": func['id']})
        response = llm.chat(messages=messages)
        pprint(response.raw)