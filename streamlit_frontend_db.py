import streamlit as st
import uuid

from langgraph_tool_backend import (
    chatbot,
    retrieve_all_threads,
    ingest_pdf,
    thread_document_metadata,
)

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# ============================================================
# 1. UTILITY FUNCTIONS
# ============================================================


def generate_thread_id():
    """Generate a unique ID for a conversation thread."""
    return uuid.uuid4()


def add_thread(thread_id):
    """Add a thread to the session's conversation list."""
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    """Start a new conversation and reset the message history."""
    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)

    st.session_state["message_history"] = []


def load_conversation(thread_id):
    """Load the saved messages for a conversation thread."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})

    return state.values.get("messages", [])


# ============================================================
# 2. SESSION STATE INITIALIZATION
# ============================================================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}


# Register the current conversation thread.
add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])

# Get the documents indexed for the current conversation.
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

# Reverse the list to display recent threads first.
threads = st.session_state["chat_threads"][::-1]

# Stores a conversation selected from the sidebar.
selected_thread = None


# ============================================================
# 3. SIDEBAR: CHAT MANAGEMENT AND PDF UPLOAD
# ============================================================

st.sidebar.title("LangGraph PDF Chatbot")

# ------------------------------------------------------------
# 3.1 Current Conversation
# ------------------------------------------------------------

st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()


# ------------------------------------------------------------
# 3.2 Current Document Status
# ------------------------------------------------------------

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]

    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from "
        f"{latest_doc.get('documents')} pages)"
    )

else:
    st.sidebar.info("No PDF indexed yet.")


# ------------------------------------------------------------
# 3.3 PDF Upload and Indexing
# ------------------------------------------------------------

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF for this chat",
    type=["pdf"],
)

if uploaded_pdf:

    if uploaded_pdf.name in thread_docs:

        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")

    else:

        with st.sidebar.status(
            "Indexing PDF...",
            expanded=True,
        ) as status_box:

            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )

            thread_docs[uploaded_pdf.name] = summary

            status_box.update(
                label="✅ PDF indexed",
                state="complete",
                expanded=False,
            )


# ------------------------------------------------------------
# 3.4 Previous Conversations
# ------------------------------------------------------------

st.sidebar.subheader("Past Conversations")

if not threads:

    st.sidebar.write("No past conversations yet.")

else:

    for thread_id in threads:

        if st.sidebar.button(
            str(thread_id),
            key=f"side-thread-{thread_id}",
        ):
            selected_thread = thread_id


# ============================================================
# 4. MAIN CHAT INTERFACE: DISPLAY MESSAGE HISTORY
# ============================================================

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# 5. USER INPUT AND AI RESPONSE STREAMING
# ============================================================

user_input = st.chat_input("Ask about your document or use tools")

if user_input:

    # --------------------------------------------------------
    # 5.1 Display and Save User Message
    # --------------------------------------------------------

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.text(user_input)

    # --------------------------------------------------------
    # 5.2 Configure the LangGraph Execution
    # --------------------------------------------------------

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"],
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"],
        },
        "run_name": "chat_turn",
    }

    # --------------------------------------------------------
    # 5.3 Stream AI Response and Track Tool Usage
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        # Stores the status container used to display tool activity.
        status_holder = {"box": None}

        def ai_only_stream():
            """
            Stream messages from LangGraph.

            Display tool activity in a status container and yield
            only AI-generated message content to Streamlit.
            """

            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):

                # Display tool execution status.
                if isinstance(message_chunk, ToolMessage):

                    tool_name = getattr(
                        message_chunk,
                        "name",
                        "tool",
                    )

                    if status_holder["box"] is None:

                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}`...",
                            expanded=True,
                        )

                    else:

                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}`...",
                            state="running",
                            expanded=True,
                        )

                # Yield only AI message content.
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        # Render the streamed AI response.
        ai_message = st.write_stream(ai_only_stream())

        # ----------------------------------------------------
        # 5.4 Finalize Tool Execution Status
        # ----------------------------------------------------

        if status_holder["box"] is not None:

            status_holder["box"].update(
                label="✅ Tool finished",
                state="complete",
                expanded=False,
            )

    # --------------------------------------------------------
    # 5.5 Save Assistant Response
    # --------------------------------------------------------

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_message,
        }
    )

    # --------------------------------------------------------
    # 5.6 Display Indexed Document Metadata
    # --------------------------------------------------------

    doc_meta = thread_document_metadata(thread_key)

    if doc_meta:

        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, "
            f"pages: {doc_meta.get('documents')})"
        )


# ============================================================
# 6. CONVERSATION SWITCHING
# ============================================================

st.divider()

if selected_thread:

    # Switch to the selected conversation.
    st.session_state["thread_id"] = selected_thread

    # Retrieve messages saved in the LangGraph checkpointer.
    messages = load_conversation(selected_thread)

    # Convert LangChain messages into Streamlit chat messages.
    temp_messages = []

    for msg in messages:

        role = "user" if isinstance(msg, HumanMessage) else "assistant"

        temp_messages.append(
            {
                "role": role,
                "content": msg.content,
            }
        )

    # Restore the selected conversation's message history.
    st.session_state["message_history"] = temp_messages

    # Initialize document tracking for the selected thread.
    st.session_state["ingested_docs"].setdefault(
        str(selected_thread),
        {},
    )

    # Rerun the application to render the selected conversation.
    st.rerun()
