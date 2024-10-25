# import subprocess
import sys
# subprocess.check_call(
#     [sys.executable, "-m", "pip", "install", "pysqlite3-binary"]
# )
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


from zhipuai_embedding import ZhipuAIEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

# 获取环境变量 API_KEY
from dotenv import load_dotenv, find_dotenv
import os
_ = load_dotenv(find_dotenv())    # read local .env file



from build_database import build_database
def get_vectordb(persist_directory = r'./data_base/vector_db/chroma'):
    # 向量数据库持久化路径
    if not os.path.exists(persist_directory):
        vectordb = build_database()
    else:
        # 定义 Embeddings
        embedding = ZhipuAIEmbeddings()
        # 加载数据库
        vectordb = Chroma(
            persist_directory=persist_directory,  # 持久化允许我们将persist_directory目录保存到磁盘上
            embedding_function=embedding
        )
    return vectordb



# 创建QA链
# 1. 无记忆
# input | template | retriever | llm
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from zhipuai_llm import ZhipuAILLM

def get_qa_chain(question, zhipu_api_key = os.environ['ZHIPUAI_API_KEY']):
    vectordb = get_vectordb()
    retriever = vectordb.as_retriever()
    llm = ZhipuAILLM(temperature=0.95)
    # test
    # response = llm.invoke("你好")
    # print(response)

    template = """严格使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图利用自身参数知识回答问题或编造答
    案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    步骤：
    1. 阅读上下文。
    2. 阅读问题。
    3. 判断上下文是否包含问题的关键词及答案。
    4. 如果上下文包含答案，则指明“根据上下文”并回答问题。
    5. 如果问题包含答案，则指明“根据问题的上下文”并回答问题。
    6. 若上下文不包含答案，但是你知道答案，则指明“数据库中未提供答案，以下是根据参数知识生成的答案”。
    7. 若上下文不包含答案，且你也不知道答案，则指明“数据库中未提供答案，且我不知道答案”。
    {context}
    问题: {input}
    """


    # PromptTemplate接受一个字符串，ChatPromptTemplate接受一个列表，可以定义系统和用户Prompt
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                    template=template)
    question_answer_chain = create_stuff_documents_chain(llm, QA_CHAIN_PROMPT)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    result = rag_chain.invoke({"input": question})
    return result['answer']
    
    

# 2. 有记忆
# 添加历史对话记忆功能。将先前的对话嵌入到语言模型中的，使其具有连续对话的能力
# (query, conversation history) -> LLM -> rephrased query -> retriever -> LLM

# 对话检索链：在检索 QA 链的基础上，增加了处理对话历史的能力。
# 工作流程是:
# 将之前的对话与新问题合并生成一个完整的查询语句。
# 在向量数据库中搜索该查询的相关文档。
# 获取结果后,存储所有答案到对话记忆区。
# 用户可在 UI 中查看完整的对话流程。

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

def get_chat_qa_chain(question, zhipu_api_key = os.environ['ZHIPUAI_API_KEY']):
    vectordb = get_vectordb()
    retriever = vectordb.as_retriever()
    llm = ZhipuAILLM(temperature=0.95)

    # 定义一个模板，用于将历史对话与新问题合并成一个完整的查询语句
    contextualize_q_system_prompt = (
        """
        给定聊天记录和最新的用户问题，用户问题可能参考聊天记录中的上下文，
        形成一个独立的问题，该问题可以在没有聊天记录的情况下理解。
        【不要】回答问题，如果需要，只需重新表述问题，否则原样返回。
        步骤：
        1. 阅读聊天记录。
        2. 阅读用户问题。
        3. 如果用户问题参考了聊天记录中的上下文，则重新表述问题
        4. 在问题前面拼接“问题：”作为开头
        5. 如果聊天记录中的上下文存在问题的答案，则
            a. 【不要】回答问题
            b. 将答案作为上下文，拼接在问题前面
        """
    )
    CONTEXTUALIZE_Q_SYSTEM_PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"), # 历史对话是如何自动保存的？答案：在 create_history_aware_retriever 函数中，将 chat_history 保存到了内部的 chat_history 变量中，然后在 ChatPromptTemplate 中使用了 MessagesPlaceholder("chat_history")，将 chat_history 传递给了 ChatPromptTemplate
            ("human", "{input}"),
        ]
    )

    qa_prompt = """严格使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图利用自身参数知识回答问题或编造答
    案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    步骤：
    1. 阅读上下文。
    2. 阅读问题。
    3. 判断上下文是否包含问题的关键词及答案。
    4. 如果上下文包含答案，则根据上下文回答问题。
    6. 若上下文不包含答案，但是你知道答案，则指明“根据参数知识生成答案”。
    7. 若上下文不包含答案，且你也不知道答案，则指明“我不知道答案”。
    {context}
    {input}
    """
    # PromptTemplate接受一个字符串，ChatPromptTemplate接受一个列表，可以定义系统和用户Prompt
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                    template=qa_prompt)

    # 创建一个链，该链接收对话历史并返回文档。
    # 如果没有 chat_history，则输入将直接传递给检索器。如果有 chat_history，则将使用提示和LLM生成搜索查询。然后将该搜索查询传递给检索器。
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONTEXTUALIZE_Q_SYSTEM_PROMPT
    )
    # 只需将retriever替换为history_aware_retriever即可
    question_answer_chain = create_stuff_documents_chain(llm, QA_CHAIN_PROMPT)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    result = rag_chain.invoke({"input": question})
    return result['answer']

if __name__ == '__main__':
    # 向量库
    vectordb = get_vectordb()
    print(f"向量库中存储的数量：{vectordb._collection.count()}")
    question = "什么是prompt engineering?"
    docs = vectordb.similarity_search(question,k=3)
    print(f"检索到的内容数：{len(docs)}")
    for i, doc in enumerate(docs):
        print(f"检索到的第{i}个内容: \n {doc.page_content}", end="\n-----------------------------------------------------\n")

    # 无记忆
    question_1 = "什么是南瓜书？"
    print(get_qa_chain(question_1))
    question_2 = "王阳明是谁？"
    print(get_qa_chain(question_2))
    question_3 = "王阳明是中国明朝时期著名的哲学家、政治家、军事家，心学的代表人物之一，他的学说和行为对后世有着深远的影响。王阳明是谁？"
    print(get_qa_chain(question_3))

    # 有记忆
    question = "我可以学习到关于提示工程的知识吗？"
    print(get_chat_qa_chain(question))
    question = "为什么这门课需要教这方面的知识？"
    print(get_chat_qa_chain(question))
    question = "王阳明是谁"
    print(get_chat_qa_chain(question))
