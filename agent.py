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
    temperature=0,
    api_key=api_key,
    api_version="2025-01-01-preview",
    azure_endpoint="https://switzerlandnorth.api.cognitive.microsoft.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview"
)

# ---- Global "knowledge base" (vector store) ----
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},       # use your RTX
    encode_kwargs={"batch_size": 64},      # tune if you want
)

vectorstore = FAISS.from_texts(["Initial empty store"], embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# ----------------------------------------------------------
# Tooling
# ----------------------------------------------------------

@tool("load_documents", return_direct=False)
def load_documents_tool(folder_path: str) -> str:
    """
    Load and index all PDF and Word (docx) files in the given folder path.
    The content is added to the shared vector store. Use this before retrieval
    if new documents have been added.
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

    # Split and add to vectorstore
    all_splits = []
    for d in docs:
        splits = text_splitter.split_documents([d])
        all_splits.extend(splits)

    texts = []
    metadatas = []
    for split in all_splits:
        texts.append(split.page_content)

        # start from loader metadata (usually has "source": full path)
        meta = dict(split.metadata)
        source_path = meta.get("source", "")
        # add a clean filename field for convenience
        if source_path:
            meta["filename"] = os.path.basename(source_path)
        else:
            meta["filename"] = "unknown"

        metadatas.append(meta)

    if texts:
        vectorstore.add_texts(texts, metadatas=metadatas)
        return f"Loaded and indexed {len(paths)} document(s) with {len(texts)} chunks."
    else:
        return "Documents found but no text extracted."


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


tools = [load_documents_tool, retrieval_tool]
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

    # events = graph.stream(
    #     {"messages": [HumanMessage(content="Index all documents in ./docs")]},
    #     config=config
    # )
    
    graph.invoke({"messages": [HumanMessage(content="Index all documents in ./docs")]}, config=config)

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