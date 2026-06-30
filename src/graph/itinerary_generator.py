import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from config import llm
from src.graph.state import TravelPlannerState, FinalItinerary

logger = logging.getLogger(__name__)


class DailyScheduleItem(BaseModel):
    """
    Morning, Afternoon, Evening, and Night schedule for a single day of the trip.
    """
    day_number: int = Field(..., description="Day index number (e.g. 1, 2, 3).")
    morning: str = Field(..., description="Activities and recommendations for the morning.")
    afternoon: str = Field(..., description="Activities and recommendations for the afternoon.")
    evening: str = Field(..., description="Activities and dining details for the evening.")
    night: str = Field(..., description="Leisure, nightlife, or evening recommendations for the night.")


class ItineraryOutput(BaseModel):
    """
    Structured day-by-day travel itinerary with pricing and presentation format.
    """
    daily_schedule: List[DailyScheduleItem] = Field(..., description="Structured daily schedule list.")
    total_cost: float = Field(..., description="Calculated sum total cost of flights, lodging, dining, and activities in USD.")
    formatted_markdown: str = Field(
        ..., 
        description="Premium formatted markdown report of the itinerary including tables, headers, and highlights."
    )


# Configure structured output LLM
structured_itinerary_llm = llm.with_structured_output(ItineraryOutput)

ITINERARY_SYSTEM_PROMPT = (
    "You are an elite travel concierge planner.\n"
    "Your task is to generate a comprehensive day-by-day travel itinerary for a trip to {destination}.\n\n"
    "Trip Parameters:\n"
    "- Total Days: {days} days\n"
    "- Total Budget Limit: {budget}\n"
    "- Weather Forecast Info:\n{weather_context}\n"
    "- Curated Attractions/Activities to include:\n{attractions_context}\n\n"
    "Instructions:\n"
    "1. For each day (from Day 1 to Day {days}), allocate specific, highly curated plans for:\n"
    "   - Morning\n"
    "   - Afternoon\n"
    "   - Evening\n"
    "   - Night\n"
    "2. Calculate a realistic total cost (total_cost) in USD, keeping the sum of flights, lodging, dining, "
    "and sightseeing within the traveler's target budget limit.\n"
    "3. Build a beautiful, rich formatted_markdown presentation of the trip. Ensure it looks highly professional "
    "with clear markdown tables, bold headings, check boxes or bullets, daily breakdowns, and a cost summary section."
)

ITINERARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ITINERARY_SYSTEM_PROMPT),
    ("placeholder", "{messages}")
])

itinerary_generator_chain = ITINERARY_PROMPT | structured_itinerary_llm


def generate_itinerary(state: TravelPlannerState) -> Dict[str, Any]:
    """
    LangGraph node that compiles the final travel itinerary.
    
    This node takes the destination, duration, budget, weather forecast, and pre-selected
    attractions, and runs a structured ChatGroq chain to generate a day-by-day (Morning, 
    Afternoon, Evening, Night) itinerary, updating state['final_itinerary'].
    
    Args:
        state (TravelPlannerState): The current LangGraph state.
        
    Returns:
        Dict[str, Any]: A state update dictionary containing 'final_itinerary'.
    """
    trip_details = state.get("trip_details") or {}
    destination = trip_details.get("destination")
    
    if not destination or not destination.strip():
        logger.warning("No destination specified in trip details. Skipping itinerary generation.")
        return {"final_itinerary": None}
        
    days = trip_details.get("number_of_days") or 1
    budget = trip_details.get("budget")
    
    # 1. Format weather forecast details from state
    weather_list = state.get("weather_forecast") or []
    weather_context = "\n".join([
        f"- {w.get('date')}: {w.get('temp_celsius')}°C, {w.get('condition')}"
        for w in weather_list
    ]) if weather_list else "Flexible / Not specified"
    
    # 2. Format activities/attractions options from state
    activity_list = state.get("activity_options") or []
    attractions_context = "\n".join([
        f"- {a.get('name')} (Cost: ${a.get('cost')}, Category: {a.get('category')}): {a.get('description')}"
        for a in activity_list
    ]) if activity_list else "Flexible / No pre-selected attractions"
    
    try:
        # 3. Invoke structured LLM chain
        response: ItineraryOutput = itinerary_generator_chain.invoke({
            "destination": destination,
            "days": days,
            "budget": f"${budget} USD" if budget else "Flexible / Not specified",
            "weather_context": weather_context,
            "attractions_context": attractions_context,
            "messages": state.get("messages", [])
        })
        
        # 4. Map Pydantic structures to FinalItinerary TypedDict format
        daily_schedule = [
            {
                "day_number": item.day_number,
                "morning": item.morning,
                "afternoon": item.afternoon,
                "evening": item.evening,
                "night": item.night
            }
            for item in response.daily_schedule
        ]
        
        final_itinerary: FinalItinerary = {
            "daily_schedule": daily_schedule,
            "total_cost": response.total_cost,
            "formatted_markdown": response.formatted_markdown
        }
        
        return {"final_itinerary": final_itinerary}
        
    except Exception as e:
        logger.exception(f"Error during itinerary generation execution: {e}")
        errors = list(state.get("validation_errors", []))
        errors.append(f"Itinerary generation failed: {str(e)}")
        return {
            "final_itinerary": None,
            "validation_errors": errors
        }


# Aliases for naming preferences
itinerary_generator_node = generate_itinerary
itinerary_generator = generate_itinerary
