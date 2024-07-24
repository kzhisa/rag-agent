import os
import uuid

import chromadb
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# OpenAI embedding model
EMBEDDING_MODEL = "text-embedding-3-small"

# ChromaDB
CHROMA_PERSIST_DIRECTORY = os.environ.get("CHROMA_PERSIST_DIRECTORY")
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME")

# Retriever settings
TOP_K_VECTOR = 10
DEFAULT_MAX_MESSAGES = 4

# Langchain LangSmith
unique_id = uuid.uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Tracing RAG agent - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"


# 既存のChromaDBを読み込みVector Retrieverを作成
def vector_retriever(top_k: int = TOP_K_VECTOR):
    """Create base vector retriever from ChromaDB

    Returns:
        Vector Retriever
    """

    # chroma db
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
    vectordb = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        client=client,
    )

    # base retriever (vector retriever)
    vector_retriever = vectordb.as_retriever(
        search_kwargs={"k": top_k},
    )

    return vector_retriever



# 会話履歴数をmax_lengthに制限するLimitedChatMessageHistoryクラス
class LimitedChatMessageHistory(ChatMessageHistory):

    # 会話履歴の保持数
    max_messages: int = DEFAULT_MAX_MESSAGES

    def __init__(self, max_messages=DEFAULT_MAX_MESSAGES):
        super().__init__()
        self.max_messages = max_messages

    def add_message(self, message):
        super().add_message(message)
        # 会話履歴数を制限
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_messages(self):
        return self.messages


# 会話履歴のストア
store = {}

# セッションIDごとの会話履歴の取得
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = LimitedChatMessageHistory()
    return store[session_id]


# プロンプトテンプレート
system_prompt = (
    "あなたは青春18きっぷとその関連商品に関する有能なアシスタントです。"
    "青春18きっぷの情報を取得するseishun18_retrieverを活用して、ユーザーの質問に答えてください。"
    "青春18きっぷに関する質問に回答する場合には、必ずseishun18_retrieverを活用して取得した情報をもとに回答してください。"
    "質問に回答するための情報が含まれない場合には、無理に質問に答えないでください。"
)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


# 実際の応答生成の例
def chat_with_bot(session_id: str):

    # LLM (OpenAI GPT-4o-mini)
    chat_model = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0.0)

    # Vector Retriever
    retriever = vector_retriever()

    # Retriever Tool
    tool = create_retriever_tool(
        retriever,
        "seishun18_retriever",
        "Searches and returns excerpts from the Autonomous Agents seishun18 knowledge",
    )
    tools = [tool]

    # Agent
    agent = create_tool_calling_agent(chat_model, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    # 会話履歴付きAgentExecutor
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    count = 0
    while True:
        print("---")
        input_message = input(f"[{count}]あなた: ")
        if input_message.lower() == "終了":
            break

        # Agentを実行
        response = agent_with_chat_history.invoke(
            {"input": input_message},
            config={"configurable": {"session_id": session_id}}
        )

        print(f"AI: {response['output']}")
        count += 1


if __name__ == "__main__":

    # チャットセッションの開始
    session_id = "example_session"
    chat_with_bot(session_id)

