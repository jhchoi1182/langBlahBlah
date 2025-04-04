from typing import TypedDict, Annotated

from langchain_community.tools import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()

class State(TypedDict):
    # 메시지 정의(list type 이며 add_messages 함수를 사용하여 메시지를 추가)
    messages: Annotated[list, add_messages]

memory = MemorySaver()
config = RunnableConfig(
    recursion_limit=10,  # 최대 10개의 노드까지 방문. 그 이상은 RecursionError 발생
    configurable={"thread_id": "1"},  # 스레드 ID 설정
)

# llm = GoogleGenerativeAI(model="gemini-2.0-flash")
llm = ChatOpenAI(model="gpt-4o-mini")

tavily_search_tool = TavilySearchResults(k=3)
tools = [tavily_search_tool]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools=[tavily_search_tool])

def chatbot(state: State):
    # 메시지 호출 및 반환
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def route_tools(
    state: State,
):
    if messages := state.get("messages", []):
        # 가장 최근 AI 메시지 추출
        ai_message = messages[-1]
    else:
        # 입력 상태에 메시지가 없는 경우 예외 발생
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    # AI 메시지에 도구 호출이 있는 경우 "tools" 반환
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        # 도구 호출이 있는 경우 "tools" 반환
        return "tools"
    # 도구 호출이 없는 경우 "END" 반환
    return END

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph_builder.add_conditional_edges(
    source="chatbot",
    path=route_tools,
    path_map={"tools": "tools", END: END},
)

graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile(checkpointer=memory)

question = "윤석열의 탄핵 심판 결과 어떻게 됐지?"

# msg = tavily_search_tool.invoke({"query": question})
# print(msg['results'])

for event in graph.stream({"messages": [("user", question)]}, config=config):
    for value in event.values():
        value["messages"][-1].pretty_print()

# question = "내 이름이 뭐라고 했지?"

# for event in graph.stream({"messages": [("user", question)]}, config=config):
#     for value in event.values():
#         value["messages"][-1].pretty_print()