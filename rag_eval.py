import json
# import re # re模块在此版本中未显式使用
from VectorDB import VectorDBManager # 假设 VectorDB.py 在同一目录下
from tqdm import tqdm
import os
from typing import List, Dict, Tuple, Any, Set # 引入类型提示

# --- 配置参数 ---
SAMPLES_FILE_PATH = '/root/autodl-tmp/RAG/SpeechAutoReporter/templates/超声报告模板/sample.jsonl'  # 您的样本文件路径
TEMPLATES_FILE_PATH = '/root/autodl-tmp/RAG/SpeechAutoReporter/templates/超声报告模板/超声报告模板.jsonl'  # 您的模板知识库文件路径
COLLECTION_NAME = 'disease_templates'  # RAG 数据库中的集合名称
CHROMA_DB_PATH = "/root/autodl-tmp/RAG/eval_chroma_db" # RAG 数据库路径 (可以自定义)
EMBED_MODEL_NAME = 'bge-m3' # 使用的嵌入模型
# 指定需要评估的 Top-N 值列表，例如评估 Top-1, Top-3, Top-5 的成功率
TOP_N_VALUES_FOR_SUCCESS_RATE: List[int] = [1, 3, 5]

# --- 辅助函数 ---

def load_jsonl_data(file_path: str) -> list:
    """加载 JSONL 文件数据"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
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

def transform_samples_for_gt_set_evaluation(samples_data: List[Dict[str, List[str]]]) -> List[Dict[str, Any]]:
    """
    将原始样本数据格式 ("疾病名称": ["描述1", "描述2"])
    转换为评估函数所需格式 [{"query": "描述1", "ground_truth_ids": ["疾病名称"]}, ...].
    """
    transformed_data = []
    for sample_entry in samples_data:
        for disease_name_true, descriptions in sample_entry.items():
            if not descriptions: # 如果某个疾病没有描述，则跳过
                continue
            # 确保 ground_truth_ids 是一个列表
            ground_truth_ids = [disease_name_true]
            for description in descriptions:
                transformed_data.append({
                    "query": description,
                    "ground_truth_ids": ground_truth_ids
                })
    return transformed_data

def evaluate_ground_truth_set_retrieval(
    db_manager: VectorDBManager,
    test_data: List[Dict[str, Any]], # 每个字典: {"query": str, "ground_truth_ids": List[str]}
    collection_name: str,
    top_n_values: List[int] # 例如: [1, 3, 5]
) -> Tuple[Dict[int, float], int]:
    """
    评估检索性能，基于一组真实标签文档ID是否完全包含在Top-N检索结果中。

    参数:
        db_manager: VectorDBManager 实例。
        test_data: 测试用例列表。每个用例是一个字典，包含
                   "query" (str) 和 "ground_truth_ids" (List[str])。
        collection_name: RAG数据库中的集合名称。
        top_n_values: 一个整数列表，代表进行Top-N评估的N值。

    返回:
        一个元组，包含:
            - 一个字典，键是N值，值是对应的成功率 (例如, {1: 0.75, 3: 0.90})。
            - 已评估的查询总数。
    """
    if not top_n_values:
        print("Error: top_n_values for evaluation cannot be empty.")
        return {}, 0
    if not test_data:
        print("No test data provided for evaluation.")
        return {n: 0.0 for n in top_n_values}, 0

    # 确保 top_n_values 是唯一的、排序的，并且是正整数
    sorted_top_n_values = sorted(list(set(n for n in top_n_values if isinstance(n, int) and n > 0)))
    if not sorted_top_n_values:
        print("Error: top_n_values must contain positive integers.")
        return {n: 0.0 for n in top_n_values}, 0 # Return 0 for original requested Ns

    max_n_for_retrieval = sorted_top_n_values[-1]
    
    success_counts = {n: 0 for n in sorted_top_n_values}
    evaluated_query_count = 0

    for test_case in tqdm(test_data, desc="Evaluating Ground Truth Set Retrieval"):
        query = test_case.get("query")
        # 确保 ground_truth_ids 是一个集合，以便使用 issubset
        gt_ids_list = test_case.get("ground_truth_ids", [])
        if isinstance(gt_ids_list, str): # 处理 ground_truth_ids 可能是单个字符串的情况
            ground_truth_ids_set = {gt_ids_list}
        elif isinstance(gt_ids_list, list):
            ground_truth_ids_set = set(gt_ids_list)
        else:
            ground_truth_ids_set = set()


        if not query or not ground_truth_ids_set:
            print(f"Warning: Skipping invalid test case (missing query or ground_truth_ids): {test_case}")
            continue
        
        evaluated_query_count += 1

        try:
            retrieved_results = db_manager.retrieve(
                collection_name=collection_name,
                query=query,
                top_k=max_n_for_retrieval
            )
            
            retrieved_doc_ids = [result['doc_id'] for result in retrieved_results]

            for n_val in sorted_top_n_values:
                # 获取当前N值对应的Top-N检索结果的文档ID集合
                current_top_n_retrieved_ids_set = set(retrieved_doc_ids[:n_val])
                
                # 检查所有真实标签ID是否都在当前Top-N结果中
                if ground_truth_ids_set.issubset(current_top_n_retrieved_ids_set):
                    success_counts[n_val] += 1
        
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            continue

    success_rates = {}
    if evaluated_query_count > 0:
        for n_val in sorted_top_n_values:
            success_rates[n_val] = success_counts[n_val] / evaluated_query_count
    else: # 如果没有评估任何查询
        for n_val in sorted_top_n_values:
            success_rates[n_val] = 0.0
            
    # 确保返回的字典包含所有原始请求的 N 值，即使它们无效也是0.0
    final_rates = {n: success_rates.get(n, 0.0) for n in top_n_values}
    return final_rates, evaluated_query_count


def print_success_rate_results(success_rates: Dict[int, float], num_evaluated: int):
    """打印基于“真实标签集完全包含”标准的成功率评估结果"""
    print("\n--- Ground Truth Set Retrieval Success Rate Results ---")
    if num_evaluated > 0:
        print(f"Total queries evaluated: {num_evaluated}")
        # 按K值排序打印结果
        for n_val in sorted(success_rates.keys()):
            rate = success_rates[n_val]
            print(f"Success Rate @Top-{n_val} (all GT IDs included): {rate:.4f}")
    else:
        print("No queries were evaluated. Check your test data or evaluation logic.")

# --- 主程序入口 ---
def main():
    """主执行函数"""
    print("Starting evaluation process...")
    print(f"Evaluation will be performed for Top-N values (success if all GT IDs retrieved): {TOP_N_VALUES_FOR_SUCCESS_RATE}")

    try:
        # 1. 加载数据
        print("\nStep 1: Loading data...")
        samples_raw = load_jsonl_data(SAMPLES_FILE_PATH)
        templates = load_jsonl_data(TEMPLATES_FILE_PATH)
        print(f"Loaded {len(samples_raw)} raw sample entries and {len(templates)} template entries.")

        # 2. 预处理模板 (用于RAG填充)
        print("\nStep 2: Preprocessing templates for RAG...")
        template_doc_ids, template_documents = preprocess_template_data(templates)
        if not template_doc_ids:
            print("No templates to process for RAG. Cannot proceed with RAG setup.")
            # 根据需求，这里可以选择退出或继续（如果RAG已预先填充）
            # 为安全起见，如果模板为空，我们可能不应该继续尝试填充RAG
            return


        # 3. 初始化 RAG 并设置集合
        print("\nStep 3: Initializing VectorDBManager and setting up RAG collection...")
        db_manager = VectorDBManager(db_path=CHROMA_DB_PATH)
        setup_rag_collection(db_manager, COLLECTION_NAME, template_doc_ids, template_documents, EMBED_MODEL_NAME)

        # 4. 转换样本数据为评估函数所需的格式
        print("\nStep 4: Transforming samples for evaluation...")
        test_data_for_eval = transform_samples_for_gt_set_evaluation(samples_raw)
        if not test_data_for_eval:
            print("No valid test data generated from samples. Exiting evaluation.")
            return
        print(f"Transformed into {len(test_data_for_eval)} test cases for evaluation.")


        # 5. 执行评估
        print("\nStep 5: Performing ground truth set retrieval evaluation...")
        success_rates, num_evaluated_queries = evaluate_ground_truth_set_retrieval(
            db_manager,
            test_data_for_eval,
            COLLECTION_NAME,
            TOP_N_VALUES_FOR_SUCCESS_RATE
        )

        # 6. 打印结果
        print("\nStep 6: Displaying results...")
        print_success_rate_results(success_rates, num_evaluated_queries)

    except FileNotFoundError:
        print("Critical Error: A required data file was not found. Evaluation cannot continue.")
    except Exception as e:
        print(f"An unexpected error occurred during the evaluation process: {e}")
        import traceback
        traceback.print_exc()


    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()