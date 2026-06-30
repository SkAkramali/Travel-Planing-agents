import logging
from typing import Dict, Any, List

from src.graph.state import TravelPlannerState, TransportOption, FlightOption
from src.tools.transport_tool import get_transport
from src.tools.flight_tool import get_flights

logger = logging.getLogger(__name__)


def transport_planning(state: TravelPlannerState) -> Dict[str, Any]:
    """
    LangGraph node that gathers recommended transportation and flight options.
    
    This node calls the transport search tool and flight search tool, 
    mapping them to transport_options and flight_options in the state.
    
    Args:
        state (TravelPlannerState): The current LangGraph state.
        
    Returns:
        Dict[str, Any]: A state update dictionary containing 'transport_options' and 'flight_options'.
    """
    trip_details = state.get("trip_details") or {}
    destination = trip_details.get("destination")
    
    if not destination or not destination.strip():
        logger.warning("No destination specified in trip details. Skipping transport planning.")
        return {"transport_options": [], "flight_options": []}
        
    try:
        # 1. Fetch general local transit (trains, buses, metro, taxis, rentals)
        transport_res = get_transport(destination)
        transport_options: List[TransportOption] = []
        if transport_res.success:
            for item in transport_res.transport:
                transport_options.append({
                    "type": item.type,
                    "provider": item.provider,
                    "route": item.route,
                    "price": item.price or 0.0,
                    "booking_link": item.booking_link or "https://www.google.com/travel"
                })
        else:
            logger.error(f"Transport search tool failed: {transport_res.message}")
            
        # 2. Fetch flights from origin to destination if specified
        origin = trip_details.get("origin")
        start_date = trip_details.get("start_date")
        flight_options: List[FlightOption] = []
        
        if origin and start_date:
            try:
                flight_res = get_flights(origin, destination, start_date)
                if flight_res.success:
                    for item in flight_res.flights:
                        flight_options.append({
                            "flight_id": item.flight_id,
                            "airline": item.airline,
                            "departure_time": item.departure_time,
                            "arrival_time": item.arrival_time,
                            "price": item.price,
                            "booking_link": item.booking_link
                        })
                else:
                    logger.info(f"Flight search returned status: {flight_res.message}")
            except Exception as fe:
                logger.warning(f"Failed to fetch flights: {fe}")
                
        return {
            "transport_options": transport_options,
            "flight_options": flight_options
        }
        
    except Exception as e:
        logger.exception(f"Error during transport planning node: {e}")
        errors = list(state.get("validation_errors", []))
        errors.append(f"Transport planning failed: {str(e)}")
        return {
            "transport_options": [],
            "flight_options": [],
            "validation_errors": errors
        }


# Aliases for naming preferences
transport_node = transport_planning
transport_planning_node = transport_planning
