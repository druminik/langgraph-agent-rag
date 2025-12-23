import os
from typing import TypedDict, Annotated, List

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import AzureChatOpenAI
from langchain.tools import tool
from langchain_core.messages import AnyMessage, ToolMessage, HumanMessage, SystemMessage

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from state import AgentState

import requests
from bs4 import BeautifulSoup

from langgraph.graph import END, START

from langchain_huggingface import HuggingFaceEmbeddings

from IPython.display import Image, display

load_dotenv()  # Load environment variables from
api_key = os.getenv("AZURE_OPENAI_API_KEY")

# ----------------------------------------------------------
# Model Setup
# ----------------------------------------------------------

llm = AzureChatOpenAI(
    max_completion_tokens=6553,
    reasoning_effort="minimal",
    api_key=api_key,
    api_version="2025-01-01-preview",
    azure_endpoint="https://switzerlandnorth.api.cognitive.microsoft.com/openai/deployments/gpt-5-nano/chat/completions?api-version=2025-01-01-preview"
)

# ---- Global "knowledge base" (vector store) ----
INDEX_DIR = "./faiss_index"

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},       # use your RTX
    encode_kwargs={"batch_size": 64},      # tune if you want
)

vectorstore = FAISS.load_local(
    INDEX_DIR,
    embeddings,
    allow_dangerous_deserialization=True,  # required in newer LangChain versions
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# ----------------------------------------------------------
# Tooling
# ----------------------------------------------------------

@tool("reload_documents", return_direct=False)
def reload_documents_tool(folder_path: str = "./docs") -> str:
    """
    Re-index all documents in the folder and overwrite the FAISS index on disk.
    """
    import glob

    paths = glob.glob(os.path.join(folder_path, "*.pdf")) + \
            glob.glob(os.path.join(folder_path, "*.docx"))

    if not paths:
        return f"No PDF or DOCX files found in folder: {folder_path}"

    docs = []
    for path in paths:
        if path.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.lower().endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            continue
        docs.extend(loader.load())

    all_splits = []
    for d in docs:
        splits = text_splitter.split_documents([d])
        all_splits.extend(splits)

    texts = []
    metadatas = []
    for split in all_splits:
        texts.append(split.page_content)
        meta = dict(split.metadata)
        src = meta.get("source", "")
        meta["filename"] = os.path.basename(src) if src else "unknown"
        metadatas.append(meta)

    if not texts:
        return "Documents found but no text extracted."

    global vectorstore, retriever
    vectorstore = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    vectorstore.save_local(INDEX_DIR)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    return f"Rebuilt index from {len(paths)} document(s) with {len(texts)} chunks and saved to {INDEX_DIR}."



@tool("search_knowledge_base", return_direct=False)
def retrieval_tool(query: str) -> str:
    """
    Perform semantic search over all loaded documents and fetched web pages.
    Returns the most relevant chunks as context.
    """
    results = retriever.invoke(query)
    if not results:
        return "No relevant results found."

    out_lines = []
    for i, doc in enumerate(results, start=1):
        src = doc.metadata.get("source", "local")
        filename = doc.metadata.get("filename") or os.path.basename(src_path)
        snippet = doc.page_content[:500].replace("\n", " ")

        out_lines.append(
            f"[{i}] Source file: {filename}\n"
            f"Full path: {src}\n"
            f"Snippet: {snippet}\n"
        )

    return "\n\n".join(out_lines)


tools = [reload_documents_tool, retrieval_tool]
llm_with_tools = llm.bind_tools(tools)


# ----------------------------------------------------------------------
# Graph nodes
# ----------------------------------------------------------------------
def call_model(state: AgentState) -> AgentState:
    """Call the LLM (which can decide to invoke tools)."""
    messages = state["messages"]
    response = llm_with_tools.invoke(
        [
            SystemMessage(
                content=(
                    "You are a helpful assistant tasked with answering questions based only on the "
                    "retrieved documents from your knowledge base. "
                    "Do NOT answer questions that are not supported by the documents. "
                    "The tool output will show snippets with 'Source file: <filename>'. "
                    "In every answer you give, add a final line like:\n"
                    "'Quelle(n): <filename1>, <filename2>' listing the filenames of the documents "
                    "you used to answer."
                )
            )
        ]
        + state["messages"]
    )
    return {"messages": messages + [response]}


def call_tools(state: AgentState) -> AgentState:
    """Execute all tool calls from the last message."""
    messages = state["messages"]
    last_msg = messages[-1]

    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        # No tools to call
        return state

    new_messages: List[AnyMessage] = []
    for tc in last_msg.tool_calls:
        name = tc["name"]
        args = tc["args"]
        tool_id = tc["id"]

        # Find the tool by name
        tool_obj = {t.name: t for t in tools}.get(name)
        if tool_obj is None:
            tool_output = f"Tool {name} not found."
        else:
            try:
                tool_output = tool_obj.invoke(args)
            except Exception as e:
                tool_output = f"Error while running tool {name}: {e}"

        new_messages.append(
            ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_id,
            )
        )

    return {"messages": messages + new_messages}


# Helper to see if model wants to call tools
def route_after_model(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    else:
        return "end"


# ----------------------------------------------------------------------
# Build graph
# ----------------------------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("model", call_model)
builder.add_node("tools", call_tools)

builder.add_edge(START, "model")
builder.add_conditional_edges("model", route_after_model, {
    "tools": "tools",
    "end": END,
})
builder.add_edge("tools", "model")  # After tools, go back to model

memory = MemorySaver()
config = {"configurable": {"thread_id": "approval-123"}}
graph = builder.compile(checkpointer=memory)


# Show the agent
display(Image(graph.get_graph(xray=True).draw_mermaid_png()))



# ----------------------------------------------------------------------
# Interactive prompt loop
# ----------------------------------------------------------------------

def main():
    while True:
        user_input = input("Enter a prompt (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        messages = [HumanMessage(content=user_input)]
        state = graph.invoke({"messages": messages}, config=config)
        if state.get("__interrupt__", None):
            print(state["__interrupt__"][0].value)    
            user_answer = input("> ")

            # 🔍 Use LLM to interpret the natural language
            resume_value = interpret_user_confirmation(user_answer)

            state = graph.invoke(Command(resume=resume_value),config=config)

        for m in state["messages"]:
            m.pretty_print()

# actually run the async main
if __name__ == "__main__":
    main()