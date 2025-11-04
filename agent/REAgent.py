from __future__ import annotations

from typing import Any, List, Optional
import sqlite3

from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools.file_management.read import ReadFileTool

from .State import State


class REAgent:
    """Runtime Execution Agent encapsulating LangGraph setup and execution.

    This class builds the conversational graph with tools and exposes simple
    methods to invoke it programmatically or via a chat handler compatible with
    Gradio's ChatInterface.
    """

    def __init__(self, model: str, db_path: str = "memory.db") -> None:
        self.model = model
        self.db_path = db_path
        self._graph = self._build_graph()

    @staticmethod
    def _get_file_tools():
        """Return the list of file tools scoped to the raw/ directory."""
        toolkit = FileManagementToolkit(root_dir="raw", selected_tools=[ReadFileTool().name])
        return toolkit.get_tools()

    def _build_graph(self):
        """Build and compile the LangGraph graph for the chatbot."""
        tools = self._get_file_tools()

        llm = ChatOpenAI(model=self.model)
        llm_with_tools = llm.bind_tools(tools=tools)

        def chatbot(state: State) -> dict[str, List[Any]]:
            # Single turn of conversation using the bound tools.
            return {"messages": [llm_with_tools.invoke(state["messages"]) ]}

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        sql_memory = SqliteSaver(conn)

        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_node("tools", ToolNode(tools=tools))

        graph_builder.add_conditional_edges("chatbot", tools_condition, "tools")
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge(START, "chatbot")
        return graph_builder.compile(checkpointer=sql_memory)

    @property
    def graph(self):
        return self._graph

    def invoke_messages(self, messages: List[Any], config: Optional[dict] = None):
        """Invoke the compiled graph with a list of LC messages.

        Parameters:
            messages: List of LangChain message objects or dicts compatible with the graph state.
            config: Optional invocation config (e.g., thread_id).
        Returns:
            The graph execution result (state dict with messages).
        """
        state = {"messages": messages}
        if config is not None:
            return self._graph.invoke(state, config=config)
        return self._graph.invoke(state)

    def chat(self, user_input: str, history, config: Optional[dict] = None) -> str:
        """Gradio-compatible chat handler.

        Parameters:
            user_input: Incoming message content from user.
            history: Prior chat history (unused here; memory handled by LangGraph).
            config: Optional invocation config.
        Returns:
            Assistant's last message content as string.
        """
        result = self.invoke_messages([{"role": "user", "content": user_input}], config=config)
        return result["messages"][ -1 ].content
