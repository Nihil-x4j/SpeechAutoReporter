'''
这个文件用于生成rag知识向量库，使用llamaindex实现，主要对llamaindex进行封装。
支持功能：
1. 在现有数据库中增删改查。
2. 保存已添加记录doc_records.json
'''

from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader,
)
from typing import Optional, List, Mapping, Any
import chromadb

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter 
#from llama_index.core.ingestion import IngestionPipeline
#from llama_index.vector_stores.chroma import ChromaVectorStore
#from llama_index.core import StorageContext
import os, re, json

import datetime

# 需要一个嵌入模型
embed_models = {
    'bge-m3': r'C:\Users\94373\Desktop\RAG\models\BAAI\bge-m3'    
}
# 数据库保存路径
CHROMA_PERSIST_DIR = r"C:\Users\94373\Desktop\RAG\chroma_db"
#封装的数据库管理类
class VectorDBManager:
    def __init__(self, db_path=CHROMA_PERSIST_DIR):
        self.db = chromadb.PersistentClient(path=db_path)
        self.collections = {
            collection_name.name : self.db.get_collection(collection_name.name)
            for collection_name in self.db.list_collections()
        }
        self.doc_records_path = os.path.join(db_path, "doc_records.json")
        self._load_doc_records()
        
        print(f"当前向量数据库为：{db_path}")
        print(f"数据库包含集合: {list(self.collections.keys())}")
    
    def _load_embed_model(self, embed_model_name):
        if embed_model_name not in embed_models:
            raise ValueError("没有该嵌入模型")
        return HuggingFaceEmbedding(
            model_name=embed_models[embed_model_name]
        )
    
    def _load_doc_records(self):
        """加载文档记录"""
        if os.path.exists(self.doc_records_path):
            with open(self.doc_records_path, 'r', encoding='utf-8') as f:
                self.doc_records = json.load(f)
            if set(self.doc_records.keys()) != set(self.collections.keys()):
                raise ValueError(f"数据库存储表和记录表不匹配")
        else:
            self.doc_records = {}
            self._save_doc_records()
    
    def _save_doc_records(self):
        """保存文档记录"""
        with open(self.doc_records_path, 'w', encoding='utf-8') as f:
            json.dump(self.doc_records, f, indent=2, ensure_ascii=False)
    
    def insert_or_update_documents(self, collection_name: str, documents: List[str], 
                        doc_ids: List[str] = None, node_parser=None, embed_model_name='bge-m3'):
        """
        插入文档或更新现有文档
        node_parser是传入的符合llamaindex格式的自定义切分器，可以用于切分文档，默认使用固定长度的切分
        """
        if doc_ids is None:
            doc_ids = [f'unknown_doc_{str(datetime.datetime.now().timestamp())}' for _ in range(len(documents))]
        else:
            if len(doc_ids) != len(documents):
                raise ValueError("文档id列表长度必须与文档列表长度相同")
            if len(doc_ids) != len(set(doc_ids)):
                raise ValueError("文档id存在重复请检查")
        
        #这段代码主要用于保证一个向量数据库内使用同样的嵌入
        if collection_name not in self.collections:
            collection = self.db.create_collection(collection_name)
            embed_model = embed_model_name
            self.collections[collection_name] = collection
            self.doc_records[collection_name] = {'doc_ids': {}, "embed_model": embed_model_name}
            self._save_doc_records()
        else:
            collection = self.collections[collection_name]
            embed_model = self.doc_records[collection_name]["embed_model"]
            if embed_model != embed_model_name:
                raise ValueError(f"当前集合使用{embed_model}作为嵌入，请保持一致")
        
        if node_parser is None:
            node_parser = SentenceSplitter()
            
        timestamp = str(datetime.datetime.now().timestamp())
        embed_model = self._load_embed_model(embed_model_name)
        
        #保存内容时会在元数据中记录文档的id和保存时间戳
        for i, (doc_id, doc_text) in enumerate(zip(doc_ids, documents)):
            embeddings=[]
            node_texts=[]
            node_ids = []
            metadatas = []
            
            doc = Document(text=doc_text, metadata={
                    "doc_id": doc_id,
                    "timestamp": timestamp
                })
            
            nodes = node_parser.get_nodes_from_documents([doc])

            for j, node in enumerate(nodes):
                embedding = embed_model.get_text_embedding(node.text)
                node_id = f"{doc_id}_node_{j}"
                
                embeddings.append(embedding)
                node_texts.append(node.text)
                node_ids.append(node_id)
                metadatas.append(node.metadata)
                
            if doc_id in self.doc_records[collection_name]['doc_ids']:
                collection.delete(where={"doc_id": doc_id})
                del self.doc_records[collection_name]['doc_ids'][doc_id]

            collection.add(
                embeddings=embeddings,
                documents=node_texts,
                ids=node_ids,
                metadatas=metadatas
            )
            self.doc_records[collection_name]['doc_ids'][doc_id] = len(nodes)
        
        self._save_doc_records()
    
    # 用于检索相似文档，输入是检索的向量集合名称、检索内容、检索个数，返回检索到内容的元数据和得分
    def retrieve(self, collection_name: str, query: str, top_k: int = 3):
        """检索相似文档"""
        if collection_name not in self.collections:
            raise ValueError(f"集合 {collection_name} 不存在")
        
        collection = self.collections[collection_name]
        embed_model_name = self.doc_records[collection_name]["embed_model"]
        embed_model = self._load_embed_model(embed_model_name)
        
        query_embedding = embed_model.get_text_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        formatted_results = []
        for i in range(len(results['ids'][0])):
            doc_id = results['metadatas'][0][i]['doc_id']
            formatted_results.append({
                'content': results['documents'][0][i],
                'doc_id': doc_id,
                'score': float(results['distances'][0][i])
            })
        print(formatted_results)
        return formatted_results
    
    def delete_collection(self, collection_name: str):
        """删除指定的集合"""
        if collection_name not in self.collections:
            raise ValueError(f"集合 {collection_name} 不存在")
        

        self.db.delete_collection(collection_name)
        del self.collections[collection_name]
        
        if collection_name in self.doc_records:
            del self.doc_records[collection_name]
        
        self._save_doc_records()
    
    def _get_all_nodes(self, collection_name: str):
        """获取集合中的所有节点"""
        if collection_name not in self.collections:
            raise ValueError(f"集合 {collection_name} 不存在")
        
        collection = self.collections[collection_name]
        all_ids = []
        all_documents = []
        all_metadatas = []
        
        # Chroma 的 get 方法默认返回最多 100 个结果，需分页获取
        limit = 100
        offset = 0
        while True:
            results = collection.get(
                limit=limit,
                offset=offset
            )
            if not results['ids']:
                break
            all_ids.extend(results['ids'])
            all_documents.extend(results['documents'])
            all_metadatas.extend(results['metadatas'])
            offset += limit
        
        return list(zip(all_ids, all_documents, all_metadatas))
    
    def view_all_documents(self):
        """打印所有集合中的所有文档节点内容"""
        for collection_name in self.collections:
            print(f"\n集合: {collection_name}")
            nodes = self._get_all_nodes(collection_name)
            for node_id, text, metadata in nodes:
                print(f"\nNode ID: {node_id}")
                print(f"Text: {text}")
                print(f"Metadata: {metadata}")
                
if __name__ == "__main__":
    TEST_COLLECTION_NAME = "disease_templates"
    TEST_DOCUMENTS = [
        "超声所见：视神经鞘直径测量结果为xxxxmm（认为小于5MM时为正常视神经）。超声提示：正常视神经鞘。"
    ]
    TEST_DOC_IDS = ["正常视神经鞘"]
    CHROMA_TEST_DIR = "C:\\Users\\94373\\Desktop\\RAG\\chroma_db"
    DOC_RECORDS_PATH = os.path.join(CHROMA_TEST_DIR, "doc_records.json")

    # 确保测试目录存在
    os.makedirs(CHROMA_TEST_DIR, exist_ok=True)

    # 初始化 VectorDBManager
    db_manager = VectorDBManager(db_path=CHROMA_TEST_DIR)

    # 插入测试文档
    print("插入测试文档...")
    print(db_manager._get_all_nodes(TEST_COLLECTION_NAME))
    db_manager.insert_or_update_documents(
        collection_name=TEST_COLLECTION_NAME,
        documents=TEST_DOCUMENTS,
        doc_ids=TEST_DOC_IDS,
        embed_model_name='bge-m3'
    )
    #print(db_manager._get_all_nodes(TEST_COLLECTION_NAME))
    # 检索测试
    print("\n进行相似性检索...")
    query = "视神经鞘直径为3mm"
    results = db_manager.retrieve(collection_name=TEST_COLLECTION_NAME, query=query, top_k=2)

    print("\n检索结果:")
    for i, result in enumerate(results):
        print(f"\n结果 {i + 1}:")
        print(f"内容: {result['content']}")
        print(f"文档ID: {result['doc_id']}")
        print(f"相似度得分: {result['score']:.4f}")

    # 验证 doc_records.json 是否正确更新
    # if os.path.exists(DOC_RECORDS_PATH):
    #     with open(DOC_RECORDS_PATH, 'r', encoding='utf-8') as f:
    #         records = json.load(f)
    #     print("\ndoc_records.json 内容:")
    #     print(json.dumps(records, indent=2, ensure_ascii=False))
    # else:
    #     print("doc_records.json 不存在。")

