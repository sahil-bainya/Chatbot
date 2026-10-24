# ============================================================
# 1. IMPORTS
# ============================================================

import sqlite3
from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# ============================================================
# 2. ENVIRONMENT CONFIGURATION
# ============================================================

load_dotenv()


# ============================================================
# 3. LLM INITIALIZATION
# ============================================================

llm = ChatGroq(model="openai/gpt-oss-20b")


# ============================================================
# 4. GRAPH STATE DEFINITION
# ============================================================


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# 5. CHAT NODE
# ============================================================


def chat_node(state: ChatState):
    """
    Process conversation messages using the LLM
    and return the generated response.
    """

    messages = state["messages"]

    response = llm.invoke(messages)

    return {"messages": [response]}


# ============================================================
# 6. SQLITE CHECKPOINTER CONFIGURATION
# ============================================================

connectionObject = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False,
)

checkpointer = SqliteSaver(conn=connectionObject)


# ============================================================
# 7. GRAPH CONSTRUCTION
# ============================================================

graph = StateGraph(ChatState)

# Register the chatbot node.
graph.add_node("chat_node", chat_node)

# Define the execution flow.
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)


# ============================================================
# 8. COMPILE THE CHATBOT
# ============================================================

chatbot = graph.compile(checkpointer=checkpointer)


# ============================================================
# 9. RETRIEVE ALL CONVERSATION THREADS
# ============================================================


def retrieve_all_threads():
    """
    Retrieve unique conversation thread IDs
    stored in the SQLite checkpointer.
    """

    all_threads = set()

    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        all_threads.add(thread_id)

    return list(all_threads)
