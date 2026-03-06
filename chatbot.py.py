# Import Libraries
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


# Load env
load_dotenv()
ENV_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# ── Tone Presets ──────────────────────────────────────────────────────────────
TONE_PRESETS = {
    "Default": "You are a helpful AI Assistant. Be clear, correct and concise.",
    "😊 Friendly": (
        "You are a warm, cheerful, and supportive AI Assistant. "
        "Use a conversational and encouraging tone. Feel free to use light expressions "
        "like 'Great question!' or 'Happy to help!'. Be approachable and kind in every response."
    ),
    "😠 Strict": (
        "You are a precise, no-nonsense AI Assistant. "
        "Be direct, formal, and concise. Avoid pleasantries or filler phrases. "
        "Only provide factual, accurate information. Do not deviate from the question asked."
    ),
    "📚 Teacher": (
        "You are a patient and knowledgeable AI Teacher. "
        "Explain concepts step-by-step with clarity. Use examples, analogies, and structured "
        "explanations to help the user understand. Encourage learning and curiosity."
    ),
}

DEFAULT_SYSTEM_PROMPT = "You are a helpful AI Assistant. Be clear, correct and concise."

# ── Streamlit Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Chatbot",
    page_icon="🤖",
    layout="centered"
)
st.title("🗣️ LLM Conversation AI Chatbot")
st.caption("Bot is Built with Streamlit + Langchain + Groq Cloud API")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎮 CONTROLS")

    api_key_input = st.text_input("Put Groq API Key Here", type="password")
    GROQ_API_KEY = api_key_input.strip() if api_key_input.strip() else ENV_GROQ_API_KEY

    # Model selection
    model_name = st.selectbox(
        "Choose Model",
        [
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
            "qwen/qwen3-32b",
            "meta-llama/llama-prompt-guard-2-86m",
            "llama-3.1-8b-instant",
        ],
        index=0,
    )

    temperature = st.slider(
        "Temperature (📝 Creativity)",
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.1
    )

    max_token = st.slider(
        "Max Token (✍️ Reply Length)",
        min_value=64, 
        max_value=1024, 
        value=256, 
        step=64
    )

    st.divider()

    # ── Tone Preset Dropdown ──────────────────────────────────────────────────
    st.subheader("🎭 Tone Preset")
    selected_tone = st.selectbox(
        "Select Bot Tone",
        options=list(TONE_PRESETS.keys()),
        index=0,
        help="Choosing a tone will auto-fill the system prompt below.",
    )

    st.divider()

    # ── System Prompt + Reset Button ──────────────────────────────────────────
    st.subheader("🧠 System Prompt")

    # Initialise system prompt in session state
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

    # When tone changes, push its value into the text area
    if selected_tone != "Default":
        tone_value = TONE_PRESETS[selected_tone]
    else:
        tone_value = st.session_state.system_prompt  # keep whatever user typed

    # Apply tone button auto-fills text area
    if st.button("✅ Apply Selected Tone"):
        st.session_state.system_prompt = TONE_PRESETS[selected_tone]
        st.rerun()

    # Editable system prompt text area
    system_prompt = st.text_area(
        "Edit System Prompt",
        value=st.session_state.system_prompt,
        height=140,
        key="system_prompt_area",
    )
    # Keep session state in sync with manual edits
    st.session_state.system_prompt = system_prompt

    # Reset System Prompt button
    if st.button("🔄 Reset System Prompt"):
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
        st.rerun()

    st.divider()

    typing_effect = st.checkbox("Enable Typing Effect", value=True)

    st.divider()

    # ── Clear Chat ────────────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat"):
        st.session_state.pop("history_store", None)
        st.rerun()

# ── API Key Guard ─────────────────────────────────────────────────────────────
if not GROQ_API_KEY:
    st.error("Groq API Key is missing. Add it in .env or paste it in the sidebar.")
    st.stop()

# ── Active tone badge ─────────────────────────────────────────────────────────
tone_icons = {"Default": "⚪", "😊 Friendly": "🟢", "😠 Strict": "🔴", "📚 Teacher": "🔵"}
st.markdown(f"**Active Tone:** {tone_icons.get(selected_tone, '⚪')} `{selected_tone}`")

# ── Chat History ──────────────────────────────────────────────────────────────
if "history_store" not in st.session_state:
    st.session_state.history_store = {}

SESSION_ID = "default_session"


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.history_store[session_id]


# ── Build LLM + Chain ─────────────────────────────────────────────────────────
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model=model_name,
    temperature=temperature,
    max_tokens=max_token,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

chain = prompt | llm | StrOutputParser()

chat_with_history = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ── Render Old Messages ───────────────────────────────────────────────────────
history_obj = get_history(SESSION_ID)

for msg in history_obj.messages:
    role = getattr(msg, "type", "")
    if role == "human":
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

# ── User Input + Response ─────────────────────────────────────────────────────
user_input = st.chat_input("Type your message...")

if user_input:
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        try:
            response_text = chat_with_history.invoke(
                {"input": user_input, "system_prompt": st.session_state.system_prompt},
                config={"configurable": {"session_id": SESSION_ID}},
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

# ── Export Section ────────────────────────────────────────────────────────────
st.divider()
st.subheader("⬇️ Export Chat History")

messages = get_history(SESSION_ID).messages

# Build shared data structures
export_json = []
export_txt_lines = [
    f"Chat Export  |  Tone: {selected_tone}",
    "=" * 50,
    "",
]

for m in messages:
    role = getattr(m, "type", "")
    label = "User" if role == "human" else "Assistant"
    export_json.append({"role": label.lower(), "text": m.content})
    export_txt_lines.append(f"[{label}]")
    export_txt_lines.append(m.content)
    export_txt_lines.append("")

export_json_payload = {
    "tone": selected_tone,
    "system_prompt": st.session_state.system_prompt,
    "messages": export_json,
}

col1, col2 = st.columns(2)

with col1:
    st.download_button(
        label="📄 Download as JSON",
        data=json.dumps(export_json_payload, ensure_ascii=False, indent=2),
        file_name="chat_history.json",
        mime="application/json",
        use_container_width=True,
    )

with col2:
    st.download_button(
        label="📝 Export Chat as TXT",
        data="\n".join(export_txt_lines),
        file_name="chat_history.txt",
        mime="text/plain",
        use_container_width=True,
    )