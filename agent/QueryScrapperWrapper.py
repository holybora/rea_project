from __future__ import annotations
from langchain.agents import create_agent
from dotenv import load_dotenv
"""
QueryScrapperAgent: A simple LangChain agent that extracts real-estate query filters
from free-form user text and returns strictly JSON according to the given schema.
"""

SYSTEM_PROMPT = ("""
You are an agent responsible for extracting real estate search parameters from the user’s text.
Your goal is to analyze natural language and return a structured JSON object with filters for querying the property database.
Be aware that database fully in English. So if user ask about property in Russian, you should return JSON in English.
Database structure (JSON keys):
	•	title: string
	•	description: string
	•	property_type: string (“residential” | “rental”)
	•	price: float (in euros)
	•	address: string
	•	city: string
	•	state: string
	•	postal_code: string
	•	bedrooms: int
	•	bathrooms: int
	•	square_meters: int
	•	amenities: [string]
	•	nearby_schools: [{“name”: string, “distance_km”: float}]
Your task:
	1.	Understand what exactly the user is looking for (type, city, budget, size, etc.).
	2.	Extract all parameters mentioned explicitly or implicitly.
	3.	Return the result strictly in JSON format, without explanations or additional text.
	4.	If the user did not specify a parameter, do not include it in the JSON.
Examples:
User:
Looking for an apartment in Valencia near the university, up to 800 euros, furnished and with Wi-Fi.
Response:
{
“property_type”: “rental”,
“city”: “Valencia”,
“price”: {“max”: 800},
“amenities”: [“furnished”, “wifi”],
“nearby_schools”: [{“name”: “Universitat de València”}]
}
User:
Need a house for sale with two bedrooms and a garden.
Response:
{
“property_type”: “residential”,
“bedrooms”: 2,
“amenities”: [“garden”]
}
User:
Apartment under 1000 euros in the Ramon Llull area.
Response:
{
“property_type”: “rental”,
“price”: {“max”: 1000},
“address”: “Calle Ramon Llull”
}
Always respond only with a JSON object in the specified structure."""
)


class QueryScrapperWrapper:
    """Simple agent wrapper to extract real-estate filters as JSON using LangChain."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.agent = create_agent(
            model,
            tools=[],
            system_prompt=SYSTEM_PROMPT,
        )
        # No tools are needed; we only transform text to structured JSON.

    @property
    def graph(self):
        return self.agent


def main():
    load_dotenv(override=True)

    wrapper = QueryScrapperWrapper()
    query_en = "Find me a property in Madrid with 1 bedrooms and Parking"
    query_es = "Encuéntrame una propiedad en Madrid con 1 dormitorio y aparcamiento"
    query_fr = "French: Trouve-moi une propriété à Madrid avec 1 chambre et un parking"
    query_ru = "Найди мне недвижимость в Мадриде с одной спальней и парковкой"

    for step in wrapper.agent.stream(
            {"messages": [{"role": "user", "content": query_ru}]}
    ):
        for update in step.values():
            for message in update.get("messages", []):
                message.pretty_print()


if __name__ == "__main__":
    raise SystemExit(main())