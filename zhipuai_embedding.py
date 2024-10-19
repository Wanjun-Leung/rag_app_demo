from __future__ import annotations # 用于类型提示

import logging
from typing import Dict, List, Any


from langchain.embeddings.base import Embeddings
# from langchain.pydantic_v1 import BaseModel, root_validator
from pydantic.v1 import BaseModel, root_validator

logger = logging.getLogger(__name__)

'''
要实现自定义 Embeddings，需要定义一个自定义类继承自 LangChain 的 Embeddings 基类，
然后定义两个函数：
① embed_query 方法，用于对单个字符串（query）进行 embedding；
② embed_documents 方法，用于对字符串列表（documents）进行 embedding。
'''
class ZhipuAIEmbeddings(BaseModel, Embeddings):
    """`Zhipuai Embeddings` embedding models."""

    client: Any
    """`zhipuai.ZhipuAI"""

    # 用于在校验整个数据模型之前对整个数据模型进行自定义校验，以确保所有的数据都符合所期望的数据结构。
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """
        实例化ZhipuAI为values["client"]

        Args:
            values (Dict): 包含配置信息的字典，必须包含 client 的字段.

        Returns:
            values (Dict): 包含配置信息的字典。如果环境中有zhipuai库，则将返回实例化的ZhipuAI类；否则将报错 'ModuleNotFoundError: No module named 'zhipuai''.
        """
        from zhipuai import ZhipuAI
        values["client"] = ZhipuAI()
        return values
    
    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.

        Args:
            texts (str): 要生成 embedding 的文本.

        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """
        embeddings = self.client.embeddings.create(
            model="embedding-2",
            input=text
        )
        return embeddings.data[0].embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embedding.
        Args:
            texts (List[str]): 要生成 embedding 的文本列表.

        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """
        return [self.embed_query(text) for text in texts]
    
    # 本来是一个异步方法，旨在处理嵌入多个文档的异步请求
    # 直接抛出 NotImplementedError，提示你使用同步版本
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        raise NotImplementedError("Please use `embed_documents`. Official does not support asynchronous requests")

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        raise NotImplementedError("Please use `aembed_query`. Official does not support asynchronous requests")