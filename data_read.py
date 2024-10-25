## 数据读取-----------------------------------

### 读取PDF-----------------------------------
from langchain_community.document_loaders import PyMuPDFLoader

# 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdf 文档路径
loader = PyMuPDFLoader("./data_base/knowledge_db/pumkin_book/pumpkin_book.pdf")

# 调用 PyMuPDFLoader Class 的函数 load 对 pdf 文件进行加载
pdf_pages = loader.load()

print(f"载入后的变量类型为：{type(pdf_pages)}, ",  f"该 PDF 一共包含 {len(pdf_pages)} 页")

'''
page 中的每一元素为一个文档（一页），变量类型为 langchain_core.documents.base.Document,
文档变量类型包含两个属性：
    page_content 包含该文档的内容。
    meta_data 为文档相关的描述性数据。
'''
pdf_page = pdf_pages[1]
print(f"每一个元素的类型：{type(pdf_page)}.", 
    f"该文档的描述性数据：{pdf_page.metadata}", 
    f"查看该文档的内容:\n{pdf_page.page_content}", 
    sep="\n------\n")


### 读取markdown-----------------------------------
from langchain_community.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("./data_base/knowledge_db/prompt_engineering/1. 简介 Introduction.md")
md_pages = loader.load()

# 读取的对象和 PDF 文档读取出来是完全一致的：
print(f"载入后的变量类型为：{type(md_pages)}, ",  f"该 Markdown 一共包含 {len(md_pages)} 页")

md_page = md_pages[0]
print(f"每一个元素的类型：{type(md_page)}.", 
    f"该文档的描述性数据：{md_page.metadata}", 
    f"查看该文档的内容:\n{md_page.page_content[0:][:200]}", 
    sep="\n------\n")


## 数据清洗-----------------------------------
'''
我们期望知识库的数据尽量是有序的、优质的、精简的，因此我们要删除低质量的、甚至影响理解的文本数据。
可以看到上文中读取的pdf文件不仅将一句话按照原文的分行添加了换行符\n，也在原本两个符号中间插入了\n，我们可以使用正则表达式匹配并删除掉\n
'''
import re
# \u4e00-\u9fff：常见汉字  ^：取反  [^\u4e00-\u9fff]：匹配非中文字符
# re.DOTALL 使得 . 匹配任何字符，包括换行符
pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL) 
# 先用pattern匹配字符串，match为匹配到的字符串，再用lambda函数自定义替换规则
pdf_page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), pdf_page.page_content)
print(pdf_page.page_content)

# 进一步分析数据，我们发现数据中还有不少的•和空格，我们的简单实用replace方法即可。
pdf_page.page_content = pdf_page.page_content.replace('•', '')
pdf_page.page_content = pdf_page.page_content.replace(' ', '')
print(pdf_page.page_content)

# 上文中读取的md文件每一段中间隔了一个换行符，我们同样可以使用replace方法去除。
md_page.page_content = md_page.page_content.replace('\n\n', '\n')
print(md_page.page_content)


## 文档分割-----------------------------------
'''
将单个文档按长度或者按固定的规则分割成若干个 chunk，然后将每个 chunk 转化为词向量，存储到向量数据库中
Langchain 中文本分割器都根据 chunk_size (块大小)和 chunk_overlap (块与块之间的重叠大小,保持上下文的连贯性)进行分割
    RecursiveCharacterTextSplitter(): 按字符串分割文本，递归地尝试按不同的分隔符进行分割文本。
    CharacterTextSplitter(): 按字符来分割文本。
    MarkdownHeaderTextSplitter(): 基于指定的标题来分割markdown 文件。
    TokenTextSplitter(): 按token来分割文本。
    SentenceTransformersTokenTextSplitter(): 按token来分割文本
    Language(): 用于 CPP、Python、Ruby、Markdown 等。
    NLTKTextSplitter(): 使用 NLTK（自然语言工具包）按句子分割文本。
    SpacyTextSplitter(): 使用 Spacy按句子的切割文本。
'''

''' 
* RecursiveCharacterTextSplitter 递归字符文本分割
    将按不同的字符递归地分割(按照这个优先级["\n\n", "\n", " ", ""])，这样就能尽量把所有和语义相关的内容尽可能长时间地保留在同一位置
    需要关注的是4个参数：
    * separators - 分隔符字符串数组
    * chunk_size - 每个文档的字符数量限制
    * chunk_overlap - 两份文档重叠区域的长度
    * length_function - 长度计算函数
'''
#导入文本分割器
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 知识库中单段文本长度
CHUNK_SIZE = 500
# 知识库中相邻文本重合长度
OVERLAP_SIZE = 50
# 使用递归字符文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP_SIZE
)
text_splitter.split_text(pdf_page.page_content[0:1000]) # 接受一个字符串，返回一个列表，列表中的每个元素为一个切分后的文本块

split_docs = text_splitter.split_documents(pdf_pages) # 接受一个文档列表，返回一个列表，列表中的每个元素为一个切分后的文本块
print(f"切分后的文件数量：{len(split_docs)}")
# print(f"切分后的文档内容：{split_docs[2].page_content}")
print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")
print(f"切分后的类型：{type(split_docs[0])}")
