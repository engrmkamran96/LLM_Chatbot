
#Import Libraries
import os
import time
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory


#Load env
#load_dotenv()
ENV_GROQ_API_KEY= os.getenv("GROQ_API_KEY","").strip()


#Streamlit Page Config

st.set_page_config(
    page_title="LLM Chatbot",
    page_icon="😒",
    layout="centered"
)
st.title ("🗣️ LLM Conversation AI Chatbot")
st.caption(" Bot is Built with Streamlit + Langchain + Groq Cloud API")

#Sidebar Control 

with st.sidebar:
    st.header("🎮CONTROLS")

    api_key_input= st.text_input(
        "Put Groq API Key Here",
        type="password"
    )

    GROQ_API_KEY= api_key_input.strip() if api_key_input.strip() else ENV_GROQ_API_KEY

    #Add Model
    model_name=st.selectbox(
        "Choose Model",
        [
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
            "qwen/qwen3-32b",
            "meta-llama/llama-prompt-guard-2-86m",
            "llama-3.1-8b-instant"
        ],
        index=0
    )

    temperature= st.slider(
        "Temperature (📝Creativity)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )

    max_token=st.slider(
        "Max Token (✍️Reply Length)",
        min_value=64,
        max_value=1024,
        value=256,
        step=64
    )

    system_prompt=st.text_area(
        "System Prompt (Rules for the Bot)",
        value="You are helpful AI Assistant. Be Clear, correct and concise",
        height=140
    )

    typing_effect= st.checkbox("Enable typing Effect", value=True)

    st.divider()

    # Clear Button

    if st.button("Clear Chat"):
        st.session_state.pop("history_store",None)
        st.session_state.pop("download_cache",None)
        st.rerun()
    

#API Key Guard

if not GROQ_API_KEY:
    st.error("Groq API Key is missing. Add it in .env or past it int the sidebar")
    st.stop()


#Chat History

if "history_store" not in st.session_state:
    st.session_state.history_store={}

#Build Default Session ID

SESSION_ID="default_session"

#Function for Get History

def get_history (session_id:str)-> InMemoryChatMessageHistory:
    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id]=InMemoryChatMessageHistory()
    return st.session_state.history_store[session_id]

#Build LLM + Prompt + Chain

llm= ChatGroq(
    groq_api_key= GROQ_API_KEY,
    model=model_name,
    temperature=temperature,
    max_tokens=max_token
)

prompt= ChatPromptTemplate.from_messages(
    [
        ("system","{system_prompt}"),
        MessagesPlaceholder(variable_name="history"),
        ("human","{input}"),
    ]
)

#Chaining All

chain= prompt | llm | StrOutputParser()

chat_with_history= RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)

#Render (Run) Old Message

history_obj = get_history(SESSION_ID)

for msg in history_obj.messages:
    role=getattr(msg,"type","")
    if role=="human":
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)


#User Input + Model Response

user_input= st.chat_input("Type youe Message....")

if user_input:
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        try:
            response_text = chat_with_history.invoke(
                {"input": user_input,"system_prompt": system_prompt},
                config = {"configurable":{"session_id": SESSION_ID}},
            )
        except Exception as e:
            st.error(f"Model Error: {e}")
            response_text = ""

        if typing_effect and response_text:
            typed = ""
            for ch in response_text:
                typed += ch
                placeholder.markdown(typed)
                time.sleep(0.005)
        else:
            placeholder.write(response_text)


# Download chat as JSON

st.divider()

st.subheader("⬇️ Download Chat History")

export_data = []

for m in get_history(SESSION_ID).messages:
    role = getattr(m, "type","")
    if role == "human":
        export_data.append({"role":"user","text": m.content})
    else:
        export_data.append({"role":"assistant","text":m.content})

st.download_button(
    label="Download chat_history.json",
    data=json.dumps(export_data, ensure_ascii=False, indent=2),
    file_name="chat_history.json",
    mime="application/json",
)

