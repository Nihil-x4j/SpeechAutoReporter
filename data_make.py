import json
# import re # re模块在此版本中未显式使用
from VectorDB import VectorDBManager # 假设 VectorDB.py 在同一目录下
from tqdm import tqdm
import os
import time
import dashscope
from dashscope import Generation
from typing import List, Dict, Tuple, Any, Set # 引入类型提示

# --- 配置参数 ---
SAMPLES_FILE_PATH = '/root/autodl-tmp/RAG/docs/MIMIC_test.json'  # 您的样本文件路径
TEMPLATES_FILE_PATH = '/root/autodl-tmp/RAG/SpeechAutoReporter/templates/超声报告模板/ct病例模板.jsonl'  # 您的模板知识库文件路径
COLLECTION_NAME = 'ct_templates'  # RAG 数据库中的集合名称
CHROMA_DB_PATH = "/root/autodl-tmp/RAG/eval_chroma_db" # RAG 数据库路径 (可以自定义)
EMBED_MODEL_NAME = 'bge-m3' # 使用的嵌入模型
# 指定需要评估的 Top-N 值列表，例如评估 Top-1, Top-3, Top-5 的成功率
TOP_N_VALUES_FOR_SUCCESS_RATE: List[int] = [1, 3, 5]

def call_dashscope_model(prompt: str, model_name: str = "qwen-plus-latest", retries: int = 3, retry_delay: int = 10) -> str:
    """
    使用 DashScope 平台调用大模型 API，输入提示词，返回模型输出结果，并处理网络异常重试。
    """
    dashscope.api_key = "sk-7ba7379293084b40bd1d50d03fa71af5"
    attempt = 0
    while attempt < retries:
        try:
            #print(prompt)
            response = Generation.call(
                model=model_name,
                messages = [
                    {'role': 'user', 'content': f'{prompt}'}
                ],
                parameters={
                    "temperature": 0.3,
                    "enable_thinking": True
                },
                extra_body={"enable_thinking": True,
                            "thinking_budget": 3000}
            )
            print(response)
            #output_text = response['output']["choices"][0]["message"]["content"].strip()
            output_text = response['output']["text"].strip()
            return output_text
        except Exception as e:
            attempt += 1
            if attempt >= retries:
                raise RuntimeError("达到最大重试次数，仍然无法完成请求。")
            else:
                print(f"尝试 {attempt}/{retries} 失败，等待 {retry_delay} 秒后重试...")
                print(f"{str(e)}")
                time.sleep(retry_delay)

def load_json_file(file_path: str) -> list:
    """加载标准 JSON 文件数据 (通常是一个包含对象列表的JSON数组)"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # print(f"Read content from {file_path}: {content[:200]}...") # Debugging
            data = json.loads(content)
        if not isinstance(data, list):
            print(f"Warning: Data loaded from {file_path} is not a list as expected. It's a {type(data)}. Please check the file content if issues arise.")
            # If the data is a single dictionary that *contains* the list, you might need to adjust
            # e.g., if data = {"samples": [...]}, then return data["samples"]
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {file_path}. Error: {e}")
        raise
    return data

def load_jsonl_data(file_path: str) -> list:
    """加载 JSONL 文件数据"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                print(line)
                data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        raise
    return data

def preprocess_template_data(templates_input: list) -> Tuple[list, list]:
    """
    预处理模板数据。
    返回: (template_doc_ids, template_documents)
    """
    disease_templates_map = {}
    for item in templates_input:
        for disease_name, regex_templates in item.items():
            if isinstance(regex_templates, list):
                disease_templates_map[disease_name] = "\n".join(regex_templates)
            else:
                disease_templates_map[disease_name] = str(regex_templates)
    
    doc_ids = list(disease_templates_map.keys())
    documents = list(disease_templates_map.values())
    
    if not doc_ids:
        print("Warning: No templates found after preprocessing. Check your templates.jsonl format.")
        
    return doc_ids, documents

def setup_rag_collection(db_manager: VectorDBManager, 
                         collection_name: str, 
                         doc_ids: list, 
                         documents: list, 
                         embed_model: str):
    """初始化或更新 RAG 集合中的模板数据"""
    if not doc_ids:
        print("Skipping RAG setup as no template documents were provided.")
        return

    # 确保 CHROMA_DB_PATH 的父目录存在
    # VectorDBManager 的 self.db (PersistentClient) 有一个 path 属性
    db_dir_path = getattr(db_manager.db, '_path', None) # ChromaDB PersistentClient stores path in _path
    if db_dir_path:
         db_dir = os.path.dirname(db_dir_path)
    else: # Fallback if path attribute is not found as expected
        db_dir = os.path.dirname(CHROMA_DB_PATH)

    if db_dir :
        os.makedirs(db_dir, exist_ok=True)

    if collection_name not in db_manager.collections:
        print(f"Collection '{collection_name}' not found, creating and inserting templates...")
    else:
        print(f"Collection '{collection_name}' already exists. Updating with new/modified templates (if any based on doc_ids).")

    try:
        db_manager.insert_or_update_documents(
            collection_name=collection_name,
            documents=documents,
            doc_ids=doc_ids,
            embed_model_name=embed_model
        )
        print(f"Successfully inserted/updated templates into '{collection_name}'.")
    except Exception as e:
        print(f"Error during RAG setup for collection '{collection_name}': {e}")
        raise



# --- 主程序入口 ---
def main():
    """主执行函数"""
    print("Starting evaluation and processing process...") # Updated print
    # Removed the TOP_N_VALUES_FOR_SUCCESS_RATE print as it might not be relevant for this specific task
    # print(f"Evaluation will be performed for Top-N values (success if all GT IDs retrieved): {TOP_N_VALUES_FOR_SUCCESS_RATE}")

    try:
        # 1. 加载数据
        print("\nStep 1: Loading data...")
        samples_raw = load_json_file(SAMPLES_FILE_PATH)
        templates = load_jsonl_data(TEMPLATES_FILE_PATH)
        print(f"Loaded {len(samples_raw)} raw sample entries and {len(templates)} template entries.")

        # 2. 预处理模板 (用于RAG填充)
        print("\nStep 2: Preprocessing templates for RAG...")
        template_doc_ids, template_documents = preprocess_template_data(templates)
        if not template_doc_ids:
            print("No templates to process for RAG. Cannot proceed with RAG setup.")
            return

        # 3. 初始化 RAG 并设置集合
        print("\nStep 3: Initializing VectorDBManager and setting up RAG collection...")
        db_manager = VectorDBManager(db_path=CHROMA_DB_PATH)
        setup_rag_collection(db_manager, COLLECTION_NAME, template_doc_ids, template_documents, EMBED_MODEL_NAME)
        
        # 4. 处理样本并逐行写入
        print("\nStep 4: Processing samples and writing to output (with resume support)...")
        
        processed_count_current_run = 0
        skipped_count_current_run = 0

        try:
            lines = sum(1 for line in open('/root/autodl-tmp/RAG/SpeechAutoReporter/templates/超声报告模板/sample3.jsonl', 'r', encoding='utf-8'))
            with open('/root/autodl-tmp/RAG/SpeechAutoReporter/templates/超声报告模板/sample3.jsonl', 'a', encoding='utf-8') as f_out:
                for idx, sample in tqdm(enumerate(samples_raw), desc="Processing Samples"):
    
                    if idx < lines:
                        tqdm.write(f"Sample already processed and found in output file. Skipping.")
                        skipped_count_current_run += 1
                        continue
                    
                    # Initialize status fields for the current sample
                    sample['processing_status'] = 'pending'
                    sample['error_message'] = None
                    sample['updated_chinese_caption'] = None
                    sample['retrieved_templates_for_generation'] = None

                    try:
                        original_caption = sample.get("caption")
                        if not original_caption:
                            tqdm.write(f"Warning: Sample {sample} has no caption. Marking as skipped_no_caption.")
                            sample['processing_status'] = 'skipped_no_caption'
                            # Fall through to write this status to the file
                        else:
                            retrieved_docs = db_manager.retrieve(
                                collection_name=COLLECTION_NAME,
                                query=original_caption,
                                top_k=3
                            )
                            sample['retrieved_templates_for_generation'] = retrieved_docs

                            context_templates_str = ""
                            if retrieved_docs:
                                context_templates_str += "请参考以下中文模板:\n"
                                for i, doc in enumerate(retrieved_docs):
                                    doc_id_str = doc.get('doc_id', 'N/A') if doc.get('doc_id') is not None else 'N/A'
                                    content_str = doc.get('content', '') if doc.get('content') is not None else ''
                                    context_templates_str += f"模板 {i+1} (来自文档ID: {doc_id_str}):\n{content_str}\n\n"
                            else:
                                context_templates_str = "未检索到相关模板。\n"

                            prompt = f"""你是医学报告生成领域的专家。你的任务是根据下面提供的英文医学影像描述，根据提供的中文模板（如果提供），将其改写成一份中文报告。注意，提供的模板不止一份时，选择最相关的一个。

{context_templates_str}
原始英文描述:
{original_caption}

请严格按照医学报告的格式和专业术语输出，仅提供生成的中文报告内容，不要包含任何额外的解释或对话。
重要内容！额外强调，医学报告具有严谨性和敏感性，对错误和格式要求很高，因此你需要在模板和因为描述没有内容冲突的情况下，尽可能完整贴合模板，存在冲突以实际描述为准，因为模板可能检索错误。
生成的中文报告:
"""
                            updated_chinese_caption = call_dashscope_model(prompt=prompt)
                            sample['updated_chinese_caption'] = updated_chinese_caption
                            sample['processing_status'] = 'success'

                    except Exception as e:
                        error_msg = f"Error processing sample {sample}: {e}"
                        tqdm.write(error_msg)
                        # import traceback # For deep debugging if needed
                        # tqdm.write(traceback.format_exc())
                        sample['processing_status'] = 'error'
                        sample['error_message'] = str(e)
                    
                    # Write the processed (or error-marked) sample to the file immediately
                    f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    f_out.flush() # Ensure it's written to disk immediately
                    
                    if sample['processing_status'] not in ['skipped_no_caption', 'error']:
                         processed_count_current_run +=1


            print(f"\nProcessing complete. {processed_count_current_run} samples processed in this run.")
            print(f"{skipped_count_current_run} samples were skipped as they were already processed in previous runs.")

        except IOError as e:
            print(f"Critical IO Error opening/writing to output file : {e}")
            raise # Re-raise to indicate critical failure

        # Step 5 is now integrated into Step 4; no separate saving step needed.


    except FileNotFoundError:
        print("Critical Error: A required data file was not found. Process cannot continue.")
    except Exception as e:
        print(f"An unexpected error occurred during the main process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Make sure VectorDB.py and its dependencies like chromadb, llama_index parts are accessible
    # Ensure dashscope.api_key is correctly set in call_dashscope_model or globally
    main()