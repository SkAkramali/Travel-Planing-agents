import logging
from typing import Dict, Any, List

from src.graph.state import TravelPlannerState, MobileNetworkOption
from src.tools.network_tool import get_mobile_networks

logger = logging.getLogger(__name__)


def mobile_networks_planning(state: TravelPlannerState) -> Dict[str, Any]:
    """
    LangGraph node that gathers recommended mobile networks and eSIM options.
    
    This node calls the network search tool for the destination, maps the structured
    output to mobile_network_options in the state, and handles any errors.
    
    Args:
        state (TravelPlannerState): The current LangGraph state.
        
    Returns:
        Dict[str, Any]: A state update dictionary containing 'mobile_network_options'.
    """
    trip_details = state.get("trip_details") or {}
    destination = trip_details.get("destination")
    
    if not destination or not destination.strip():
        logger.warning("No destination specified in trip details. Skipping mobile networks planning.")
        return {"mobile_network_options": []}
        
    try:
        # Call the networks tool
        response = get_mobile_networks(destination)
        
        if not response.success:
            logger.error(f"Mobile networks search tool failed: {response.message}")
            errors = list(state.get("validation_errors", []))
            errors.append(f"Mobile networks search tool failed: {response.message}")
            return {
                "mobile_network_options": [],
                "validation_errors": errors
            }
            
        # Format as state's MobileNetworkOption TypedDict
        mobile_network_options: List[MobileNetworkOption] = []
        for item in response.operators:
            mobile_network_options.append({
                "provider": item.operator_name,
                "plan_name": item.plan_name or "Tourist plan",
                "data_limit": item.data_limit or "Not specified",
                "price": item.price or 0.0,
                "validity_days": item.validity_days or 0
            })
            
        return {"mobile_network_options": mobile_network_options}
        
    except Exception as e:
        logger.exception(f"Error during mobile networks node execution: {e}")
        errors = list(state.get("validation_errors", []))
        errors.append(f"Mobile networks planning failed: {str(e)}")
        return {
            "mobile_network_options": [],
            "validation_errors": errors
        }


# Aliases for naming preferences
mobile_networks_node = mobile_networks_planning
mobile_networks_planning_node = mobile_networks_planning
