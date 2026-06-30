import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from config import llm
from src.graph.state import TravelPlannerState, ActivityOption
from src.tools.search_tool import search_web

logger = logging.getLogger(__name__)


class ActivityOptionItem(BaseModel):
    """
    Structured details for a recommended activity or attraction.
    """
    activity_id: str = Field(..., description="Unique slug identifier (e.g., 'eiffel-tower').")
    name: str = Field(..., description="Name of the activity or attraction.")
    description: str = Field(..., description="Short summary/description of the attraction.")
    cost: float = Field(..., description="Estimated ticket fee or cost in USD (0.0 if free).")
    duration_hours: float = Field(..., description="Estimated duration of the activity in hours.")
    category: str = Field(..., description="Category (e.g., museum, nature, historical, adventure).")


class ActivityRecommendations(BaseModel):
    """
    List of activity recommendations.
    """
    activities: List[ActivityOptionItem] = Field(..., description="List of recommended activities.")


# Configure structured output LLM
structured_activities_llm = llm.with_structured_output(ActivityRecommendations)

ATTRACTIONS_SYSTEM_PROMPT = (
    "You are a travel planning expert specializing in sightseeing and activities.\n"
    "Generate 3 to 5 recommended activities or attractions for a trip to {destination}.\n"
    "Align the options with the traveler's interests, budget, and trip type.\n\n"
    "Traveler Profile:\n"
    "- Budget: {budget}\n"
    "- Interests: {interests}\n"
    "- Trip Type: {trip_type}\n\n"
    "Search Context:\n"
    "{search_context}\n\n"
    "Provide a structured list of recommendations matching the traveler's context."
)

ATTRACTIONS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ATTRACTIONS_SYSTEM_PROMPT),
    ("placeholder", "{messages}")
])

activities_chain = ATTRACTIONS_PROMPT | structured_activities_llm


def attractions_planning(state: TravelPlannerState) -> Dict[str, Any]:
    """
    LangGraph node that gathers recommended attractions/activities.
    
    This node searches the web for attractions in the destination, processes the results
    through ChatGroq, maps them to activity_options in the state, and handles errors.
    
    Args:
        state (TravelPlannerState): The current LangGraph state.
        
    Returns:
        Dict[str, Any]: A state update dictionary containing 'activity_options'.
    """
    trip_details = state.get("trip_details") or {}
    destination = trip_details.get("destination")
    
    if not destination or not destination.strip():
        logger.warning("No destination specified in trip details. Skipping attractions planning.")
        return {"activity_options": []}
        
    budget = trip_details.get("budget")
    interests = trip_details.get("interests", [])
    trip_type = trip_details.get("trip_type")
    
    # 1. Search the web for attractions
    search_context = "No external search context available."
    search_query = f"top tourist attractions activities points of interest in {destination}"
    try:
        search_res = search_web(search_query, max_results=4)
        if search_res.success and search_res.results:
            search_context = "\n\n".join([
                f"Title: {r.title}\nContent: {r.content}\nURL: {r.url}"
                for r in search_res.results
            ])
    except Exception as e:
        logger.warning(f"Tavily search failed during attractions planning: {e}")
        
    try:
        # 2. Invoke structured LLM chain
        response: ActivityRecommendations = activities_chain.invoke({
            "destination": destination,
            "budget": f"${budget} USD" if budget else "Flexible",
            "interests": ", ".join(interests) if interests else "General sightseeing",
            "trip_type": trip_type or "Flexible",
            "search_context": search_context,
            "messages": state.get("messages", [])
        })
        
        # 3. Format as state's ActivityOption TypedDict
        activity_options: List[ActivityOption] = [
            {
                "activity_id": item.activity_id,
                "name": item.name,
                "description": item.description,
                "cost": item.cost,
                "duration_hours": item.duration_hours,
                "category": item.category
            }
            for item in response.activities
        ]
        
        return {"activity_options": activity_options}
        
    except Exception as e:
        logger.exception(f"Error during attractions node execution: {e}")
        errors = list(state.get("validation_errors", []))
        errors.append(f"Attractions planning failed: {str(e)}")
        return {
            "activity_options": [],
            "validation_errors": errors
        }


# Aliases for naming preferences
attractions_node = attractions_planning
attractions_planning_node = attractions_planning
