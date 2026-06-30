import logging
from typing import Dict, Any, List

from src.graph.state import TravelPlannerState, HotelOption
from src.tools.hotel_tool import get_hotels

logger = logging.getLogger(__name__)


def hotels_planning(state: TravelPlannerState) -> Dict[str, Any]:
    """
    LangGraph node that gathers recommended hotel options.
    
    This node calls the hotel search tool for the destination, maps the structured
    output to hotel_options in the state, and handles any errors.
    
    Args:
        state (TravelPlannerState): The current LangGraph state.
        
    Returns:
        Dict[str, Any]: A state update dictionary containing 'hotel_options'.
    """
    trip_details = state.get("trip_details") or {}
    destination = trip_details.get("destination")
    
    if not destination or not destination.strip():
        logger.warning("No destination specified in trip details. Skipping hotels planning.")
        return {"hotel_options": []}
        
    try:
        # Call the hotel tool
        response = get_hotels(destination)
        
        if not response.success:
            logger.error(f"Hotel search tool failed: {response.message}")
            errors = list(state.get("validation_errors", []))
            errors.append(f"Hotels search tool failed: {response.message}")
            return {
                "hotel_options": [],
                "validation_errors": errors
            }
            
        # Format as state's HotelOption TypedDict
        hotel_options: List[HotelOption] = []
        for idx, item in enumerate(response.hotels):
            hotel_options.append({
                "hotel_id": f"hotel-{idx+1}-{item.name.lower().replace(' ', '-')}",
                "name": item.name,
                "price_per_night": item.price or 0.0,
                "rating": item.rating or 0.0,
                "address": item.location,
                "booking_link": "https://www.booking.com"
            })
            
        return {"hotel_options": hotel_options}
        
    except Exception as e:
        logger.exception(f"Error during hotels node execution: {e}")
        errors = list(state.get("validation_errors", []))
        errors.append(f"Hotels planning failed: {str(e)}")
        return {
            "hotel_options": [],
            "validation_errors": errors
        }


# Aliases for naming preferences
hotels_node = hotels_planning
hotels_planning_node = hotels_planning
