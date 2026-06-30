import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from config import llm
from src.graph.state import TravelPlannerState, RecommendedDestination

logger = logging.getLogger(__name__)


class DestinationRecommendationItem(BaseModel):
    """
    Structured recommendation details for a single destination.
    """
    name: str = Field(..., description="Name of the recommended city/country (e.g., 'Kyoto, Japan').")
    reason: str = Field(..., description="Reason why this destination matches the traveler's budget, trip type, and season.")


class DestinationRecommendationOutput(BaseModel):
    """
    List of recommended travel destinations.
    """
    recommendations: List[DestinationRecommendationItem] = Field(..., description="List of recommended destinations.")


# Configure structured output LLM
structured_rec_llm = llm.with_structured_output(DestinationRecommendationOutput)

# System prompt for destination suggestions
RECOMMENDATION_SYSTEM_PROMPT = (
    "You are an expert travel consultant.\n"
    "The traveler has not selected a destination yet. Based on the details provided and the conversation history, "
    "recommend 3 suitable travel destinations that fit their profile.\n\n"
    "Travel profile details:\n"
    "- Maximum Budget: {budget}\n"
    "- Trip Type/Style: {trip_type}\n"
    "- Season/Timing: {season_info}\n\n"
    "For each recommendation, provide the name of the destination and a clear, brief reason explaining "
    "why it matches their budget, style, and season/timing preferences."
)

RECOMMENDATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RECOMMENDATION_SYSTEM_PROMPT),
    ("placeholder", "{messages}")
])

# Recommendation chain
recommendation_chain = RECOMMENDATION_PROMPT | structured_rec_llm


def destination_recommendation(state: TravelPlannerState) -> Dict[str, Any]:
    """
    LangGraph node that recommends destinations if the user has not specified one.
    
    If state['trip_details']['destination'] is missing or empty, this node suggests 
    3 destinations matching their budget, trip type, and season, and updates the 
    'recommended_destinations' state field. Otherwise, it makes no changes.
    
    Args:
        state (TravelPlannerState): The current LangGraph state.
        
    Returns:
        Dict[str, Any]: A state update dictionary containing 'recommended_destinations'.
    """
    trip_details = state.get("trip_details") or {}
    destination = trip_details.get("destination")
    
    # 1. If destination is already set, do nothing
    if destination and destination.strip():
        logger.info(f"Destination '{destination}' is already set. Skipping destination recommendations.")
        return {}
        
    budget = trip_details.get("budget")
    trip_type = trip_details.get("trip_type")
    start_date = trip_details.get("start_date")
    
    # Determine season from start date if available
    season_info = "Flexible / Not specified"
    if start_date:
        try:
            dt = datetime.strptime(start_date, "%Y-%m-%d")
            month = dt.strftime("%B")
            season_info = f"Starts in {month}"
        except Exception:
            season_info = f"Timing set to {start_date}"
            
    try:
        # Run recommendation model
        response: DestinationRecommendationOutput = recommendation_chain.invoke({
            "budget": f"${budget} USD" if budget else "Flexible / Not specified",
            "trip_type": trip_type or "Flexible / Not specified",
            "season_info": season_info,
            "messages": state.get("messages", [])
        })
        
        # Convert response items to state format (RecommendedDestination TypedDict)
        recommended_list: List[RecommendedDestination] = [
            {
                "name": item.name,
                "reason": item.reason
            }
            for item in response.recommendations
        ]
        
        return {"recommended_destinations": recommended_list}
        
    except Exception as e:
        logger.exception(f"Error during destination recommendation execution: {e}")
        errors = list(state.get("validation_errors", []))
        errors.append(f"Destination Recommendation failed: {str(e)}")
        return {
            "recommended_destinations": [],
            "validation_errors": errors
        }


# Alias for backward compatibility
destination_recommendation_node = destination_recommendation
