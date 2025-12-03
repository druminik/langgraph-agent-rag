from langchain_core.messages import AnyMessage, ToolMessage, HumanMessage, SystemMessage
from typing_extensions import TypedDict, Annotated, List
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], "The conversation history"]
