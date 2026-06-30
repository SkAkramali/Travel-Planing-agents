import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from config import llm
from src.graph.state import TravelPlannerState, BudgetEstimate

logger = logging.getLogger(__name__)


class BudgetEstimatorOutput(BaseModel):
    """
    Structured budget estimation breakdown.
    """
    hotel_cost: float = Field(..., description="Estimated lodging/hotel cost in USD.")
    food_cost: float = Field(..., description="Estimated food and dining cost in USD.")
    transport_cost: float = Field(..., description="Estimated transport cost in USD.")
    sightseeing_cost: float = Field(..., description="Estimated sightseeing/activities cost in USD.")
    shopping_cost: float = Field(..., description="Estimated shopping and souvenir cost in USD.")
    total_estimated_cost: float = Field(..., description="Sum total of all estimated costs in USD.")
    breakdown_reasoning: str = Field(
        ..., 
        description="Detail assumptions and reasoning behind the estimation, showing how search results were utilized."
    )


# Configure structured LLM instance
structured_budget_llm = llm.with_structured_output(BudgetEstimatorOutput)

BUDGET_ESTIMATOR_SYSTEM_PROMPT = (
    "You are an expert travel budget analyst.\n"
    "Calculate estimated costs for a trip to {destination} based on the traveler's profile, "
    "duration, and discovered travel options from search results.\n\n"
    "Traveler Profile:\n"
    "- Target budget limit: {budget_limit}\n"
    "- Trip duration: {duration_days} days\n"
    "- Number of travelers: {num_travelers}\n\n"
    "Discovered Travel Options (Use these to compute estimates if populated):\n"
    "1. Hotels/Lodging Options:\n{hotel_context}\n\n"
    "2. Food/Dining Options:\n{food_context}\n\n"
    "3. Transport/Flight Options:\n{transport_context}\n\n"
    "4. Sightseeing/Activities:\n{activity_context}\n\n"
    "5. Shopping Options:\n{shopping_context}\n\n"
    "Compute individual estimates in USD for:\n"
    "- Hotel: Lodging cost based on duration, rooms, or rates.\n"
    "- Food: Meals and dining costs for the trip duration.\n"
    "- Transport: Flights, train ticket, public transit, or car rentals.\n"
    "- Sightseeing: Attraction entries or tour costs.\n"
    "- Shopping: Gift or market expenses.\n\n"
    "Return a structured breakdown with your calculations and reasoning. Do not generate an itinerary."
)

BUDGET_ESTIMATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", BUDGET_ESTIMATOR_SYSTEM_PROMPT),
    ("placeholder", "{messages}")
])

budget_estimator_chain = BUDGET_ESTIMATOR_PROMPT | structured_budget_llm


def estimate_budget(state: TravelPlannerState) -> Dict[str, Any]:
    """
    LangGraph node that estimates the trip's cost breakdown.
    
    This node checks the state for discovered travel options (hotels, dining,
    transport, activities, and shopping), processes them with ChatGroq to compute
    cost estimates, updates 'budget_estimate', and does not create an itinerary.
    
    Args:
        state (TravelPlannerState): The current LangGraph state.
        
    Returns:
        Dict[str, Any]: A state update dictionary containing 'budget_estimate'.
    """
    trip_details = state.get("trip_details") or {}
    destination = trip_details.get("destination")
    
    if not destination or not destination.strip():
        logger.warning("No destination specified in trip details. Skipping budget estimation.")
        return {"budget_estimate": None}
        
    # Extract trip parameters
    budget_limit = trip_details.get("budget")
    duration_days = trip_details.get("number_of_days") or 1
    num_travelers = trip_details.get("num_travelers") or 1
    
    # 1. Format context from search results/options currently in state
    hotel_options = state.get("hotel_options", [])
    hotel_context = "\n".join([
        f"- {h.get('name')}: ${h.get('price_per_night')}/night. Address: {h.get('address')}"
        for h in hotel_options
    ]) if hotel_options else "No hotels found in search results yet."
    
    food_options = state.get("food_options", [])
    food_context = "\n".join([
        f"- {f.get('name')} ({f.get('cuisine')} cuisine): Price range {f.get('price_range')}. Rating: {f.get('rating')}"
        for f in food_options
    ]) if food_options else "No food options found in search results yet."
    
    transport_options = state.get("transport_options", [])
    transport_context = "\n".join([
        f"- {t.get('provider')} ({t.get('type')}): Route: {t.get('route')}. Price: ${t.get('price')}"
        for t in transport_options
    ]) if transport_options else ""
    # Include flight_options as fallback/extra context
    flight_options = state.get("flight_options", [])
    if flight_options:
        flight_context = "\n".join([
            f"- Flight with {f.get('airline')}: Price: ${f.get('price')}"
            for f in flight_options
        ])
        transport_context = f"{transport_context}\n{flight_context}".strip()
    if not transport_context:
        transport_context = "No transport options found in search results yet."
        
    activity_options = state.get("activity_options", [])
    activity_context = "\n".join([
        f"- {a.get('name')}: Cost: ${a.get('cost')}. Duration: {a.get('duration_hours')} hrs."
        for a in activity_options
    ]) if activity_options else "No activity options found in search results yet."
    
    shopping_options = state.get("shopping_options", [])
    shopping_context = "\n".join([
        f"- {s.get('name')} ({s.get('type')}): Specialty: {s.get('specialty')}. Location: {s.get('location')}"
        for s in shopping_options
    ]) if shopping_options else "No shopping options found in search results yet."
    
    try:
        # 2. Invoke structured LLM chain to estimate cost
        response: BudgetEstimatorOutput = budget_estimator_chain.invoke({
            "destination": destination,
            "budget_limit": f"${budget_limit} USD" if budget_limit else "Flexible",
            "duration_days": duration_days,
            "num_travelers": num_travelers,
            "hotel_context": hotel_context,
            "food_context": food_context,
            "transport_context": transport_context,
            "activity_context": activity_context,
            "shopping_context": shopping_context,
            "messages": state.get("messages", [])
        })
        
        # 3. Map as state's BudgetEstimate TypedDict
        estimate: BudgetEstimate = {
            "hotel_cost": response.hotel_cost,
            "food_cost": response.food_cost,
            "transport_cost": response.transport_cost,
            "sightseeing_cost": response.sightseeing_cost,
            "shopping_cost": response.shopping_cost,
            "total_estimated_cost": response.total_estimated_cost,
            "breakdown_reasoning": response.breakdown_reasoning
        }
        
        return {"budget_estimate": estimate}
        
    except Exception as e:
        logger.exception(f"Error during budget estimation node execution: {e}")
        errors = list(state.get("validation_errors", []))
        errors.append(f"Budget estimation failed: {str(e)}")
        return {
            "budget_estimate": None,
            "validation_errors": errors
        }


# Aliases for naming preferences
budget_estimator_node = estimate_budget
budget_estimator = estimate_budget
cost_estimator_node = estimate_budget
