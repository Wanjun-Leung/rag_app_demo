import streamlit as st
from zhipuai_llm import ZhipuAILLM
import os
from langchain_core.output_parsers import StrOutputParser
from qa_chain import get_qa_chain, get_chat_qa_chain

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

    
# 使用用户密钥对 OpenAI API 进行身份验证、发送提示并获取 AI 生成的响应。
# 该函数接受用户的提示作为参数，并使用st.info来在蓝色框中显示 AI 生成的响应
def generate_response(input_text, api_key):
    llm = ZhipuAILLM(temperature=0.7, api_key=api_key)
    output = llm.invoke(input_text)
    output_parser = StrOutputParser()
    output = output_parser.invoke(output)
    #st.info(output) # 单轮对话
    return output # 多轮对话

def main():
    # 创建应用程序的标题
    st.title('🦜🔗 动手学大模型应用开发')
    print('title is called')

    # 添加一个文本输入框，供用户输入其 API 密钥
    zhipu_api_key = st.sidebar.text_input('请输入您的智谱AI API密钥', type='password')
    # 不输入不能进行下面的步骤
    if not zhipu_api_key:
        st.warning('Please enter your Zhipu API key!', icon='⚠')
        return
    if zhipu_api_key == 'admin':
        zhipu_api_key = os.environ['ZHIPUAI_API_KEY']

    selected_method = st.radio(
        "你想选择哪种模式进行对话？",
        ["None", "qa_chain", "chat_qa_chain"],
        captions = ["不使用检索问答的普通模式", "不带历史记录的检索问答模式", "带历史记录的检索问答模式"])


    # # 1. 单轮对话：
    # # 使用st.form()创建一个文本框（st.text_area()）供用户输入。
    # # 当用户单击Submit时，generate-response()将使用用户的输入作为参数来调用该函数
    # with st.form('my_form'): # 创建并返回一个表单，with方法可以往该表单中添加元素
    #     text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
    #     submitted = st.form_submit_button('Submit')
    #     if not zhipu_api_key:
    #         st.warning('Please enter your Zhipu API key!', icon='⚠')
    #     elif zhipu_api_key == 'admin':
    #         zhipu_api_key = os.environ['ZHIPUAI_API_KEY']
    #     if submitted:
    #         generate_response(text)
    
    # 2. 多轮对话：
    # 使用 st.session_state 来存储对话历史。
    # 用于存储应用程序的会话状态，可以是任何Python对象，例如字典、列表、字符串等
    # 可以在应用程序的不同部分共享和访问会话状态

    clear = st.form_submit_button('Clear chat history')
    
    if 'messages' not in st.session_state or clear:
        st.session_state.messages = []
    print(st.session_state)
    
    messages = st.container(height=300)
    if prompt := st.chat_input("Say something"): # :=海象运算符，在表达式中赋值
        print(prompt)
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})

        if selected_method == "None":
            # 调用 respond 函数获取回答
            answer = generate_response(prompt, zhipu_api_key)
        elif selected_method == "qa_chain":
            answer = get_qa_chain(prompt,zhipu_api_key)
        elif selected_method == "chat_qa_chain":
            answer = get_chat_qa_chain(prompt,zhipu_api_key)


        # 检查回答是否为 None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])   

if __name__ == "__main__":
    print("main() is called")
    main()
