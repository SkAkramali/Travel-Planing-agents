import logging
from typing import Dict, Any

from langgraph.graph import StateGraph, END

# Import the state schema
from src.graph.state import TravelPlannerState

# Import node functions
from src.graph.query_understanding import query_understanding
from src.graph.destination_recommendation import destination_recommendation
from src.graph.weather import weather_forecast
from src.graph.attractions import attractions_planning
from src.graph.food import food_planning
from src.graph.hotels import hotels_planning
from src.graph.transport import transport_planning
from src.graph.mobile_networks import mobile_networks_planning
from src.graph.shopping import shopping_planning
from src.graph.budget_estimator import estimate_budget
from src.graph.itinerary_generator import generate_itinerary

logger = logging.getLogger(__name__)


def info_router_node(state: TravelPlannerState) -> Dict[str, Any]:
    """
    Helper joiner/router node that facilitates fanning out to parallel information nodes.
    
    Args:
        state (TravelPlannerState): The current LangGraph state.
        
    Returns:
        Dict[str, Any]: Empty update dict (performs no state modifications).
    """
    return {}


def final_response_node(state: TravelPlannerState) -> Dict[str, Any]:
    """
    Compiles the final text or markdown response and appends it to state['messages'].
    
    Args:
        state (TravelPlannerState): The current LangGraph state.
        
    Returns:
        Dict[str, Any]: State update dictionary adding the final AIMessage to 'messages'.
    """
    from langchain_core.messages import AIMessage
    
    recommended = state.get("recommended_destinations")
    itinerary = state.get("final_itinerary")
    
    if recommended:
        content = "I've suggested some travel destinations based on your preferences:\n\n"
        for idx, rec in enumerate(recommended, 1):
            content += f"{idx}. **{rec['name']}**: {rec['reason']}\n"
        content += "\nPlease let me know which destination you'd like to plan for!"
    elif itinerary:
        content = itinerary.get("formatted_markdown", "Itinerary generated successfully.")
    else:
        content = "I gathered your preferences but couldn't plan an itinerary. Please make sure to specify a destination."
        
    return {"messages": [AIMessage(content=content)]}


def route_after_query(state: TravelPlannerState) -> str:
    """
    Conditional routing logic checking if destination is missing.
    
    Args:
        state (TravelPlannerState): The current LangGraph state.
        
    Returns:
        str: 'recommendation' if destination is missing, otherwise 'info_planning'.
    """
    trip_details = state.get("trip_details") or {}
    destination = trip_details.get("destination")
    
    if not destination or not destination.strip():
        logger.info("Destination is missing. Routing to destination recommendation.")
        return "recommendation"
    else:
        logger.info("Destination specified. Routing to information planning nodes.")
        return "info_planning"


# Initialize the StateGraph with the custom State structure
workflow = StateGraph(TravelPlannerState)

# Add nodes
workflow.add_node("query_understanding", query_understanding)
workflow.add_node("destination_recommendation", destination_recommendation)
workflow.add_node("info_router", info_router_node)

# Add information gathering nodes
workflow.add_node("weather_forecast", weather_forecast)
workflow.add_node("attractions_planning", attractions_planning)
workflow.add_node("food_planning", food_planning)
workflow.add_node("hotels_planning", hotels_planning)
workflow.add_node("transport_planning", transport_planning)
workflow.add_node("mobile_networks_planning", mobile_networks_planning)
workflow.add_node("shopping_planning", shopping_planning)

# Add cost estimator, itinerary generator, and final response nodes
workflow.add_node("estimate_budget", estimate_budget)
workflow.add_node("generate_itinerary", generate_itinerary)
workflow.add_node("final_response", final_response_node)

# Set Entry Point
workflow.set_entry_point("query_understanding")

# Add Conditional Edge from Query Understanding
workflow.add_conditional_edges(
    "query_understanding",
    route_after_query,
    {
        "recommendation": "destination_recommendation",
        "info_planning": "info_router"
    }
)

# If destination recommendation was needed, go straight to Final Response
workflow.add_edge("destination_recommendation", "final_response")

# Parallel Fan-out from Info Router to all Info Nodes
workflow.add_edge("info_router", "weather_forecast")
workflow.add_edge("info_router", "attractions_planning")
workflow.add_edge("info_router", "food_planning")
workflow.add_edge("info_router", "hotels_planning")
workflow.add_edge("info_router", "transport_planning")
workflow.add_edge("info_router", "mobile_networks_planning")
workflow.add_edge("info_router", "shopping_planning")

# Parallel Fan-in (Join) from all Info Nodes to Budget Estimator
workflow.add_edge("weather_forecast", "estimate_budget")
workflow.add_edge("attractions_planning", "estimate_budget")
workflow.add_edge("food_planning", "estimate_budget")
workflow.add_edge("hotels_planning", "estimate_budget")
workflow.add_edge("transport_planning", "estimate_budget")
workflow.add_edge("mobile_networks_planning", "estimate_budget")
workflow.add_edge("shopping_planning", "estimate_budget")

# Budget Estimator to Itinerary Generator
workflow.add_edge("estimate_budget", "generate_itinerary")

# Itinerary Generator to Final Response
workflow.add_edge("generate_itinerary", "final_response")

# Final Response to END
workflow.add_edge("final_response", END)

# Compile the workflow into a runnable graph
app = workflow.compile()
