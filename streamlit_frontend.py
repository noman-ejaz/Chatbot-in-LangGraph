import streamlit as st
from langgraph_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# **************************************** utility functions *************************
def heading_message():
    st.header("Hello User! How can i help you?")

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])


# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])


# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role': role, 'content': msg.text})

        st.session_state['message_history'] = temp_messages


# **************************************** Main UI ************************************

heading_message()

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_turn",
    }

    # assistant response
    with st.chat_message("assistant"):
        # Reserve slots: status box stays ABOVE the streamed answer
        status_placeholder = st.empty()
        text_placeholder = st.empty()
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # 1) Model decided to call a tool -> show it live, before it runs
                if isinstance(message_chunk, AIMessage) and message_chunk.tool_calls:
                    names = ", ".join(tc["name"] for tc in message_chunk.tool_calls)
                    box = status_placeholder.status(
                        f"🔧 Calling `{names}` …", expanded=True
                    )
                    for tc in message_chunk.tool_calls:
                        box.write(f"`{tc['name']}` args: `{tc['args']}`")
                    status_holder["box"] = box

                # 2) Tool finished -> mark the status complete
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = status_placeholder.status(
                            f"🔧 Used `{tool_name}`", expanded=True
                        )
                    status_holder["box"].update(
                        label=f"✅ `{tool_name}` finished",
                        state="complete",
                        expanded=False,
                    )

                # 3) Stream ONLY assistant text tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.text

        ai_message = text_placeholder.write_stream(ai_only_stream())

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )