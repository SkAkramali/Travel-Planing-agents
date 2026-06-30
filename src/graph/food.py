import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from config import llm
from src.graph.state import TravelPlannerState, FoodOption
from src.tools.search_tool import search_web

logger = logging.getLogger(__name__)


class FoodOptionItem(BaseModel):
    """
    Structured details for a recommended restaurant or food place.
    """
    name: str = Field(..., description="Name of the restaurant or eatery.")
    cuisine: str = Field(..., description="Type of cuisine (e.g., Italian, Japanese, Street Food).")
    price_range: str = Field(..., description="Price tier indicator ($, $$, or $$$).")
    rating: float = Field(..., description="Review rating score out of 5.0.")
    description: str = Field(..., description="Short summary/highlight of this eatery.")


class FoodRecommendations(BaseModel):
    """
    List of food recommendations.
    """
    places: List[FoodOptionItem] = Field(..., description="List of recommended eating spots.")


# Configure structured output LLM
structured_food_llm = llm.with_structured_output(FoodRecommendations)

FOOD_SYSTEM_PROMPT = (
    "You are a culinary travel expert.\n"
    "Generate 3 to 5 recommended dining locations or street food places in {destination}.\n"
    "Align the options with the traveler's budget and interests.\n\n"
    "Traveler Profile:\n"
    "- Budget: {budget}\n"
    "- Interests: {interests}\n"
    "- Trip Type: {trip_type}\n\n"
    "Search Context:\n"
    "{search_context}\n\n"
    "Provide a structured list of restaurant recommendations."
)

FOOD_PROMPT = ChatPromptTemplate.from_messages([
    ("system", FOOD_SYSTEM_PROMPT),
    ("placeholder", "{messages}")
])

food_chain = FOOD_PROMPT | structured_food_llm


def food_planning(state: TravelPlannerState) -> Dict[str, Any]:
    """
    LangGraph node that gathers recommended food and dining options.
    
    This node searches the web for top restaurants in the destination, processes the results
    through ChatGroq, maps them to food_options in the state, and handles errors.
    
    Args:
        state (TravelPlannerState): The current LangGraph state.
        
    Returns:
        Dict[str, Any]: A state update dictionary containing 'food_options'.
    """
    trip_details = state.get("trip_details") or {}
    destination = trip_details.get("destination")
    
    if not destination or not destination.strip():
        logger.warning("No destination specified in trip details. Skipping food planning.")
        return {"food_options": []}
        
    budget = trip_details.get("budget")
    interests = trip_details.get("interests", [])
    trip_type = trip_details.get("trip_type")
    
    # 1. Search the web for food/restaurants
    search_context = "No external search context available."
    search_query = f"best restaurants street food places to eat in {destination}"
    try:
        search_res = search_web(search_query, max_results=4)
        if search_res.success and search_res.results:
            search_context = "\n\n".join([
                f"Title: {r.title}\nContent: {r.content}\nURL: {r.url}"
                for r in search_res.results
            ])
    except Exception as e:
        logger.warning(f"Tavily search failed during food planning: {e}")
        
    try:
        # 2. Invoke structured LLM chain
        response: FoodRecommendations = food_chain.invoke({
            "destination": destination,
            "budget": f"${budget} USD" if budget else "Flexible",
            "interests": ", ".join(interests) if interests else "General food",
            "trip_type": trip_type or "Flexible",
            "search_context": search_context,
            "messages": state.get("messages", [])
        })
        
        # 3. Format as state's FoodOption TypedDict
        food_options: List[FoodOption] = [
            {
                "name": item.name,
                "cuisine": item.cuisine,
                "price_range": item.price_range,
                "rating": item.rating,
                "description": item.description
            }
            for item in response.places
        ]
        
        return {"food_options": food_options}
        
    except Exception as e:
        logger.exception(f"Error during food node execution: {e}")
        errors = list(state.get("validation_errors", []))
        errors.append(f"Food planning failed: {str(e)}")
        return {
            "food_options": [],
            "validation_errors": errors
        }


# Aliases for naming preferences
food_node = food_planning
food_planning_node = food_planning
