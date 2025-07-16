from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader,
)
from typing import Optional, List, Mapping, Any
import chromadb
from VectorDB import VectorDBManager
import os, re, json

# from transcribe import stream_transcribe
from llm import DashScopeLLM
Settings.llm = DashScopeLLM()
DATA_DIR = "/root/autodl-tmp/RAG/docs/chunks1.json"
CHROMA_PERSIST_DIR = "/root/autodl-tmp/RAG/chroma_db"

#content = stream_transcribe('/root/autodl-tmp/RAG/llama_index/test2.m4a', block_duration_seconds=30, overlap_seconds=1)



def process_audio_content(audio_content: str) -> dict:

    prompt = """请根据以下医疗问诊语音识别的对话内容，识别说话人身份（医生/患者），由于原始内容来自语音识别可能存在字词错误，请你进行修正,可以对错误字词替换，增添或者删除。
                输出格式要求：{"content"：[{"身份":"内容"},{"身份":"内容"}], "subject":"总结内容"}
                1. 身份只能来自：医生、患者、家属、噪声四类。
                2. 内容是说话人说的话
                3. 总结内容是对整个对话的总结，要求简洁明了，包含主要信息(来自什么科室的对话，疾病部位，症状)，尽可能简洁准确便于后续和模板匹配
                格式化答案部分用###包裹，便于后续处理，包裹部分内容确保为可以被jsons.loads()解析的格式化字符串
                如
                ###
                {"content":[{"医生":"你好，我是医生，请问有什么问题？"},{"患者":"我最近感觉头痛，想咨询一下。"}],
                "subject":"患者头痛，咨询医生"}
                ###
                注意，修正不正确的字词非常重要，因为其是后续基于文本检索的基础参考，请多考虑多音字、名称相关性等可能进行正确词汇猜测，保证修正句子不存在意义不明字词，前后逻辑顺畅没有冲突。
                对于总结内容，需要对病人异常症状进行总结概括，避免出现对话中没有的内容和推测的结果。
                原始内容：
            """ + f'{audio_content}'
    
    # 调用LLM处理
    response = Settings.llm.complete(prompt)
    try:
        # 解析LLM返回的JSON格式内容
        pattern = r'###(.*?)###'
        match = re.search(pattern, response.text, re.DOTALL)
        if match:
            res = match.group(1).strip()
        result = json.loads(res)
        print(f"处理结果：{result}")
        return result
    except json.JSONDecodeError:
        print("LLM返回格式解析失败，返回原始格式")

def generate_medical_description(retrieved_nodes: List, dialog_content: List[dict], query: str) -> str:
    # 构建上下文：将检索到的相关模板组合
    templates = "\n".join([f"模板{i+1}：{node.node.text}" for i, node in enumerate(retrieved_nodes)])
    
    # 构建对话内容
    # Convert dialog content list to formatted string
    dialog = json.dumps(dialog_content, ensure_ascii=False)
    
    prompt = f"""基于检索到的超声报告模板和患者对话内容，生成规范的病例描述。

参考模板内容：
{templates}

患者对话记录：
{dialog}

要求：
1. 严格参考模板的术语和表达方式
2. 保持医学描述的专业性和规范性
3. 如果检索到的多个模板有重复或冗余内容，需要在最相关目标模板中进行整合
4. 输出内容应该是根据模板条目纠正过的规范化病例描述，而不是单纯复制报告模板
5. 最重要的是，病例报告仅能出现对话部分诊断，其他额外的内容和推测的结果可能导致严重医疗事故，因此严谨出现对话内容外的表述和推出。
请生成规范的病例描述："""

    response = Settings.llm.complete(prompt)
    print(f"生成的病例描述：{response.text}")
    return response.text

res = process_audio_content(content)
query = res['subject']
content = res['content']
generate_medical_description(retrieve_topk(query, k=3), content, query)