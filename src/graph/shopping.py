import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from config import llm
from src.graph.state import TravelPlannerState, ShoppingOption
from src.tools.search_tool import search_web

logger = logging.getLogger(__name__)


class ShoppingOptionItem(BaseModel):
    """
    Structured details for a recommended shopping spot or district.
    """
    name: str = Field(..., description="Name of the shopping mall, market, or district.")
    location: str = Field(..., description="Street location, neighborhood, or general district name.")
    type: str = Field(..., description="Type of shopping (e.g. Flea Market, Boutique Mall, Souvenirs, Luxury).")
    specialty: str = Field(..., description="Unique items to buy or specialty highlight of this venue.")


class ShoppingRecommendations(BaseModel):
    """
    List of shopping spot recommendations.
    """
    shops: List[ShoppingOptionItem] = Field(..., description="List of recommended shopping spots.")


# Configure structured output LLM
structured_shopping_llm = llm.with_structured_output(ShoppingRecommendations)

SHOPPING_SYSTEM_PROMPT = (
    "You are a shopping and retail travel guide.\n"
    "Generate 3 to 5 recommended shopping spots, markets, or districts in {destination}.\n"
    "Match shopping recommendations with the traveler's interests, budget, and trip type.\n\n"
    "Traveler Profile:\n"
    "- Budget: {budget}\n"
    "- Interests: {interests}\n"
    "- Trip Type: {trip_type}\n\n"
    "Search Context:\n"
    "{search_context}\n\n"
    "Provide a structured list of shopping recommendations."
)

SHOPPING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SHOPPING_SYSTEM_PROMPT),
    ("placeholder", "{messages}")
])

shopping_chain = SHOPPING_PROMPT | structured_shopping_llm


def shopping_planning(state: TravelPlannerState) -> Dict[str, Any]:
    """
    LangGraph node that gathers recommended shopping locations.
    
    This node searches the web for best shopping spots and local markets in the destination, 
    processes the results through ChatGroq, maps them to shopping_options in the state, 
    and handles errors.
    
    Args:
        state (TravelPlannerState): The current LangGraph state.
        
    Returns:
        Dict[str, Any]: A state update dictionary containing 'shopping_options'.
    """
    trip_details = state.get("trip_details") or {}
    destination = trip_details.get("destination")
    
    if not destination or not destination.strip():
        logger.warning("No destination specified in trip details. Skipping shopping planning.")
        return {"shopping_options": []}
        
    budget = trip_details.get("budget")
    interests = trip_details.get("interests", [])
    trip_type = trip_details.get("trip_type")
    
    # 1. Search the web for shopping districts and markets
    search_context = "No external search context available."
    search_query = f"best shopping districts local markets malls to buy things in {destination}"
    try:
        search_res = search_web(search_query, max_results=4)
        if search_res.success and search_res.results:
            search_context = "\n\n".join([
                f"Title: {r.title}\nContent: {r.content}\nURL: {r.url}"
                for r in search_res.results
            ])
    except Exception as e:
        logger.warning(f"Tavily search failed during shopping planning: {e}")
        
    try:
        # 2. Invoke structured LLM chain
        response: ShoppingRecommendations = shopping_chain.invoke({
            "destination": destination,
            "budget": f"${budget} USD" if budget else "Flexible",
            "interests": ", ".join(interests) if interests else "Shopping",
            "trip_type": trip_type or "Flexible",
            "search_context": search_context,
            "messages": state.get("messages", [])
        })
        
        # 3. Format as state's ShoppingOption TypedDict
        shopping_options: List[ShoppingOption] = [
            {
                "name": item.name,
                "location": item.location,
                "type": item.type,
                "specialty": item.specialty
            }
            for item in response.shops
        ]
        
        return {"shopping_options": shopping_options}
        
    except Exception as e:
        logger.exception(f"Error during shopping node execution: {e}")
        errors = list(state.get("validation_errors", []))
        errors.append(f"Shopping planning failed: {str(e)}")
        return {
            "shopping_options": [],
            "validation_errors": errors
        }


# Aliases for naming preferences
shopping_node = shopping_planning
shopping_planning_node = shopping_planning
