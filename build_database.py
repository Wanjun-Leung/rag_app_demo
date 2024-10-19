# import subprocess
import sys
# subprocess.check_call(
#     [sys.executable, "-m", "pip", "install", "pysqlite3-binary"]
# )
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 调用Embedding API
# from langchain.embeddings.openai import OpenAIEmbeddings          # 使用 OpenAI Embedding
# from langchain.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint             # 使用百度千帆 Embedding
from zhipuai_embedding import ZhipuAIEmbeddings          # 使用我们自己封装的智谱 Embedding，需要将封装代码下载到本地使用

from langchain_community.vectorstores import Chroma

# 读取本地/项目的环境变量。
# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中  
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 如果你需要通过代理端口访问，你需要如下配置
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'

def build_database():
    # 获取folder_path下所有文件路径，储存在file_paths里
    file_paths = []
    folder_path = r'./data_base/knowledge_db' # r''表示原始字符串，不会对字符串中的特殊字符进行转义
    for root, dirs, files in os.walk(folder_path): # root: 当前目录路径, dirs: 当前路径下所有子目录（不包含路径root）, files: 当前路径下所有非目录子文件名（不包含路径root）
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    # print(file_paths[:3])

    # 遍历文件路径并把实例化的loader存放在loaders里
    loaders = []

    for file_path in file_paths:
        file_type = file_path.split('.')[-1]
        if file_type == 'pdf':
            loaders.append(PyMuPDFLoader(file_path))
        elif file_type == 'md':
            loaders.append(UnstructuredMarkdownLoader(file_path))

    # 下载文件并存储到text
    texts = []
    for loader in loaders: 
        texts.extend(loader.load())

    # text = texts[1]
    # print(f"每一个元素的类型：{type(text)}.", 
    #     f"该文档的描述性数据：{text.metadata}", 
    #     f"查看该文档的内容:\n{text.page_content[0:]}", 
    #     sep="\n------\n")  # sep 每个输出项之间的分隔符

    # 省略数据清洗部分

    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(texts)

    # 定义 Embeddings
    # embedding = OpenAIEmbeddings() 
    # embedding = QianfanEmbeddingsEndpoint()
    embedding = ZhipuAIEmbeddings()

    # 定义持久化路径
    persist_directory = './data_base/vector_db/chroma'       # 先删除旧的数据库文件（如果文件夹中有文件的话），windows电脑请手动删除
    # # 如果路径存在，删除文件夹
    # if os.path.exists(persist_directory):
    #     os.system(f"rm -rf {persist_directory}")

    # 选择 Chroma向量数据库 是因为它轻量级且数据存储在内存中，这使得它非常容易启动和开始使用
    # from_documetns 传入List[Document]，即langchain_core.documents.base.Document变量类型的列表，from_texts 传入List[str]
    # embedding传入我们定义的Embedding，persist_directory传入我们定义的持久化路径
    vectordb = Chroma.from_documents(
        documents=split_docs[:20], # 为了速度，只选择前 20 个切分的 doc 进行生成；使用千帆时因QPS限制，建议选择前 5 个doc
        embedding=embedding,
        persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
    )

    # 持久化数据库，执行了persist()操作以后向量数据库才真正的被保存到了本地，下次在需要使用该向量数据库时我们只需要从本地加载数据库即可，无需再根据原始文档来生成向量数据库了
    # 在 Chroma 0.4.x 版本之后已经不再需要，因为文档现在会自动持久化。
    # vectordb.persist()

    return vectordb

if __name__ == "__main__":
    vectordb = build_database()

    # 向量检索
    # 1. 相似度检索。如果只考虑检索出内容的相关性会导致内容过于单一，可能丢失重要信息
    question = "什么是大语言模型？"
    sim_docs = vectordb.similarity_search(question,k=3)
    print(f"检索到的内容数：{len(sim_docs)}")
    for i, sim_doc in enumerate(sim_docs):
        print(f"检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")

    # 2. MMR（最大边际相关性）检索。在保持相关性的同时，增加内容的丰富度。
    # .mmr_search(question, k=3, lambda_=0.5)不能用
    mmr_docs = vectordb.max_marginal_relevance_search(question, k=3, lambda_=0.5) # lambda_ 为 MMR 的超参数，控制相关性和多样性的权重，默认值为 0.5
    for i, sim_doc in enumerate(mmr_docs):
        print(f"MMR 检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")
