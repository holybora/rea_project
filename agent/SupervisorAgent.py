from typing import Optional

import create_agent
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from agent.PropFinderAgent import PropFinderAgent
from deepagents import CompiledSubAgent

##todo: make link thread_id with user session
config = {"configurable": {"thread_id": "3"}}

SUPERVISOR_PROMPT = """
You are assistant designed to help visitors find properties on the website.
You support English, Spanish, French and Russian. Response always in language of the user.
Tone & Style: emoji-friendly 😄. Avoid formal or robotic language. Be short.
Your goals:
	•	Greet the user and create a casual, human-like conversation 🤝
	•	Show clear and friendly results or hints on how to refine the search
	
If user ask to find something you need to check local database using sub-agent
	”"""
load_dotenv(override=True)

class SupervisorAgent:


    def __init__(self):
        self.property_search = PropFinderAgent()
        # Use it as a custom subagent
        self.custom_subagent = CompiledSubAgent(
            name="property_search_subagent",
            description="Specialized agent for property search in local database.",
            runnable= self.property_search.app
        )

        self.supervisor_agent = create_agent(
            "gpt-5-mini",
            system_prompt=SUPERVISOR_PROMPT,
            checkpointer=InMemorySaver(),
        )

    @tool("run_subagent", return_direct=False)
    def run_subagent(self, text: str) -> str:
        """Call the sub-agent on a text input."""
        result = self.subagent.invoke({"text": text})
        return result["sub_result"]

def main():
    app = SupervisorAgent()
    query_ru = "Найди мне недвижимость в Валенсии под аренду в твоей"

    result = app.supervisor_agent.invoke({"messages": [{"role": "user", "content": query_ru}]}, config = config)

# Print the agent's response
    print(result["messages"][-1].content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())