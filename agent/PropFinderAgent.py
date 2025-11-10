from pathlib import Path
from typing import Literal
from uuid import uuid4

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, StateGraph, END
from typing import Any
from langchain_core.messages import AIMessage, ToolMessage
from agent.SubmitFinalAnswer import SubmitFinalAnswer
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.tools import tool
from agent.State import State


class PropFinderAgent:

    def __init__(self, config: RunnableConfig = None):
        self.model = ChatOpenAI(model="gpt-5-mini")
        db_path = Path(__file__).resolve().parents[1] / "properties.db"
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_path.as_posix()}")
        print(f"Database Type: {self.db.dialect}")
        print(f"Tables: {self.db.get_usable_table_names()}")
        print(f"Artist: {self.db.run('SELECT * FROM properties LIMIT 5;')}")
        self.config = config or RunnableConfig()

        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.model)
        self.tools = toolkit.get_tools()

        self.db_query_tool = next(next_tool for next_tool in self.tools if next_tool.name == "sql_db_query")
        self.db_schema_tool = next(next_tool for next_tool in self.tools if next_tool.name == "sql_db_schema")
        self.db_list_tables_tool = next(next_tool for next_tool in self.tools if next_tool.name == "sql_db_list_tables")
        self.db_query_checker_tool = next(next_tool for next_tool in self.tools if next_tool.name == "sql_db_query_checker")

        self.llm_with_schema_tool = self.model.bind_tools(tools=self.tools)

        self.config = config or RunnableConfig(recursion_limit=20)
        self.loop_counter = 15

        # Define the query generation system prompt
        query_gen_system = """You are a SQLite expert with a strong attention to detail.
        
        Given an input question, output a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
        
        DO NOT call any tool besides SubmitFinalAnswer to submit the final answer.
        
        When generating the query:
        
        Output the SQLite query that answers the input question without a tool call.
        
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        
        If you get an error while executing a query, rewrite the query and try again.
        
        If you get an empty result set, you should try to rewrite the query to get a non-empty result set.
        NEVER make stuff up if you don't have enough information to answer the query... just say you don't have enough information.
        
        If you have enough information to answer the input question, simply invoke the appropriate tool to submit the final answer to the user.
        
        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database."""
        query_gen_prompt = ChatPromptTemplate.from_messages(
            [("system", query_gen_system), ("placeholder", "{messages}")]
        )
        self.query_gen_chain = (query_gen_prompt | ChatOpenAI(model="gpt-5-mini", temperature=0)
                                .bind_tools([SubmitFinalAnswer]))

        # Define the query check system prompt
        query_check_system = """You are a SQLite expert with a strong attention to detail.
        Double check the SQLite query for common mistakes, including:
        - Using NOT IN with NULL values
        - Using UNION when UNION ALL should have been used
        - Using BETWEEN for exclusive ranges
        - Data type mismatch in predicates
        - Properly quoting identifiers
        - Using the correct number of arguments for functions
        - Casting to the correct data type
        - Using the proper columns for joins
        
        If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.
        
        You will call the appropriate tool to execute the query after running this check."""

        query_check_prompt = ChatPromptTemplate.from_messages(
            [("system", query_check_system), ("placeholder", "{messages}")]
        )

        db_stmt_exec_tool = tool(self.db_stmt_exec_tool)
        self.query_check_chain = (query_check_prompt | ChatOpenAI(model="gpt-5-mini", temperature=0)
                                  .bind_tools([db_stmt_exec_tool], tool_choice="required"))

        # Define the workflow
        workflow = StateGraph(State)

        # Add nodes with redesigned names
        workflow.add_node("initial_tool_node", self.initial_tool_node)
        workflow.add_node("list_tables_node", self.list_tables_node)
        workflow.add_node("model_get_schema_node", self.model_get_schema_node)
        workflow.add_node("retrieve_schema_node", self.retrieve_schema_node)
        workflow.add_node("query_gen_node", self.query_gen_node)
        workflow.add_node("correct_query_node", self.correct_query_node)
        workflow.add_node("execute_query_node", self.execute_query_node)

        # Add edges with updated node names
        workflow.add_edge(START, "initial_tool_node")
        workflow.add_edge("initial_tool_node", "list_tables_node")
        workflow.add_edge("list_tables_node", "model_get_schema_node")
        workflow.add_edge("model_get_schema_node", "retrieve_schema_node")
        workflow.add_edge("retrieve_schema_node", "query_gen_node")
        workflow.add_conditional_edges("query_gen_node", self.should_continue, [END, "correct_query_node", "query_gen_node"])
        workflow.add_edge("correct_query_node", "execute_query_node")
        workflow.add_edge("execute_query_node", "query_gen_node")

        self.app = workflow.compile()


    # Define the initial tool node
    def initial_tool_node(self, state: State) -> dict[str, list[AIMessage]]:
        """
        Initializes the workflow by creating a tool call to list tables in the database.
        """
        tool_call_id = "initial_tool_call_id_123" # hardcoded for debug
        print(f"--- First Tool Call Node ---\nTool Call ID: {tool_call_id}")
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "sql_db_list_tables",
                            "args": {},
                            "id": tool_call_id,
                        }
                    ],
                )
            ]
        }

    # Define the list tables node
    def list_tables_node(self, state: State) -> dict[str, list[AIMessage]]:
        """
        Lists all tables in the database and returns the result as a ToolMessage.
        """
        print("--- List Tables Node ---")
        result = self.db_list_tables_tool.invoke({})
        print("Tables in the database:", result)

        # Get the tool_call_id from the previous message
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        print(f"Tool Call ID: {tool_call_id}")
        return {"messages": [ToolMessage(content=result, tool_call_id=tool_call_id)]}

    # Define the model get schema node
    def model_get_schema_node(self, state: State) -> dict[str, list[AIMessage]]:
        """
        Uses the model to generate a schema request based on the current state.
        """
        print("--- Model Get Schema Node ---")
        return {"messages": [self.llm_with_schema_tool.invoke(state["messages"])]}

    # Define the retrieve schema node
    def retrieve_schema_node(self, state: State) -> dict[str, list[AIMessage]]:
        """
        Retrieves the schema for a specific table and returns it as a ToolMessage.
        """
        print("--- Retrieve Schema Node ---")
        table_name = state["messages"][-1].tool_calls[0]["args"]["table_names"]
        result = self.db_schema_tool.invoke(table_name)
        print(f"Schema for table '{table_name}':\n{result}")

        # Get the tool_call_id from the previous message
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        print(f"Tool Call ID: {tool_call_id}")

        # Return a ToolMessage with the same tool_call_id
        return {"messages": [ToolMessage(content=result, tool_call_id=tool_call_id)]}

    # Define the query generation node
    def query_gen_node(self, state: State):
        """
        Generates a SQLite query based on the current state and returns the result.
        """
        print("--- Query Gen Node ---")
        # Ensure the last message is a ToolMessage
        if isinstance(state["messages"][-1], ToolMessage):
            tool_call_id = state["messages"][-1].tool_call_id
            print(f"Tool Call ID from previous message: {tool_call_id}")
        else:
            raise ValueError("Expected a ToolMessage as the last message.")

        # Generate the query
        message = self.query_gen_chain.invoke(state)
        tool_messages = []
        if message.tool_calls:
            for tc in message.tool_calls:
                if tc["name"] != "SubmitFinalAnswer":
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                            tool_call_id=tc["id"],
                        )
                    )
        else:
            tool_messages = []
        return {"messages": [message] + tool_messages}

    # Define the should_continue function
    def should_continue(self, state: State) -> Literal[END, "correct_query", "query_gen"]:
        """
        Determines the next step in the workflow based on the current state.
        """
        messages = state["messages"]
        last_message = messages[-1]
        if getattr(last_message, "tool_calls", None):
            return END
        if last_message.content.startswith("Error:"):
            return "query_gen"
        else:
            return "correct_query"


    def db_stmt_exec_tool(self, query: str) -> str:
        """
        Execute an SQLite query against the database and return the result.
        If the query fails, return an error message.
        """
        result = self.db.run_no_throw(query)
        if not result:
            return "Error: Query failed. Please rewrite your query and try again."
        return result

    # Define the correct query node
    def correct_query_node(self, state: State) -> dict[str, list[AIMessage]]:
        """
        Corrects the SQLite query if necessary and returns the corrected query.
        """
        print("--- Correct Query Node ---")
        return {"messages": [self.query_check_chain.invoke({"messages": [state["messages"][-1]]})]}

    # Define the execute query node
    def execute_query_node(self, state: State) -> dict[str, list[Any]]:
        """
        Executes the SQLite query and returns the result as a ToolMessage.
        """
        print("--- Execute Query Node ---")
        try:
            query = state["messages"][-1].tool_calls[0]["args"]["query"]
            result = self.db_stmt_exec_tool.invoke(query)
            print(f"Query Results:\n{result}")
        except Exception as e:
            result = f"Error: {str(e)}"
            print(result)

        # Get the tool_call_id from the previous message
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        print(f"Tool Call ID: {tool_call_id}")
        return {"messages": [ToolMessage(content=result, tool_call_id=tool_call_id)]}



def main():
    load_dotenv(override=True)

    wrapper = PropFinderAgent()
    query_en = "Find me a property in Madrid with 1 bedrooms and Parking"
    query_es = "Encuéntrame una propiedad en Madrid con 1 dormitorio y aparcamiento"
    query_fr = "French: Trouve-moi une propriété à Madrid avec 1 chambre et un parking"
    query_ru = "Найди мне недвижимость в Валенсии под аренду"

    for step in wrapper.app.stream(
            {"messages": [{"role": "user", "content": query_ru}]},
        config=wrapper.config,
    ):
        for update in step.values():
            for message in update.get("messages", []):
                message.pretty_print()


if __name__ == "__main__":
    raise SystemExit(main())