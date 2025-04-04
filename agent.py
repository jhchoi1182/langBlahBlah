from langchain.agents import create_tool_calling_agent, AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.tools import Tool
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_teddynote.messages import AgentStreamParser
from vector import search_vector_db

load_dotenv()
# import tavily_search

search = TavilySearchResults(k=3)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. "
            "Make sure to use the `pdf_search` tool for searching information information from PDF documents. "
            "If you can't find the information from the PDF documents, use the `search` tool for searching information from the web.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# tools = [
#     Tool(
#         name="tavily_search_tool",
#         func=search,
#         description="Use this tool to search on the web.",
#     ),
#     Tool(
#         name="pdf_search",
#         func=search_vector_db,
#         description="Search the vector database pdf documents using the provided query.",
#     )
# ]

tools = [search, search_vector_db]

# llm = GoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25")
llm = ChatOpenAI(model="gpt-4o-mini")

agent = create_tool_calling_agent(llm, tools, prompt)
# agent = create_react_agent(
#     llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_stream_parser = AgentStreamParser()

result = agent_executor.stream({"input": "삼성 전자의 매출액을 문서에서 찾아줘."})

# print(result)

for step in result:
    # 중간 단계를 parser 를 사용하여 단계별로 출력
    agent_stream_parser.process_agent_steps(step)