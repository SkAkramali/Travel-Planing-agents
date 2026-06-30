import logging
from typing import Dict, Any, List
from src.graph.state import TravelPlannerState, WeatherForecast
from src.tools.weather_tool import get_weather_forecast

logger = logging.getLogger(__name__)


def weather_forecast(state: TravelPlannerState) -> Dict[str, Any]:
    """
    LangGraph node that fetches the weather forecast for the trip destination.
    
    This node invokes the weather tool using the destination in state['trip_details'],
    converts the response forecast to the state's WeatherForecast format, and handles
    any network or API key errors gracefully.
    
    Args:
        state (TravelPlannerState): The current LangGraph state.
        
    Returns:
        Dict[str, Any]: A state update dictionary containing 'weather_forecast'.
    """
    trip_details = state.get("trip_details") or {}
    destination = trip_details.get("destination")
    
    # 1. Check if destination is specified
    if not destination or not destination.strip():
        logger.warning("No destination specified in trip details. Skipping weather forecast retrieval.")
        return {"weather_forecast": []}
        
    try:
        # 2. Call the weather tool
        response = get_weather_forecast(destination)
        
        # 3. Handle API/Tool failure
        if not response.success:
            logger.error(f"Weather tool failed for location '{destination}': {response.error_message}")
            errors = list(state.get("validation_errors", []))
            errors.append(f"Failed to fetch weather: {response.error_message}")
            return {
                "weather_forecast": [],
                "validation_errors": errors
            }
            
        # 4. Map Pydantic model attributes to WeatherForecast TypedDict list
        forecast_list: List[WeatherForecast] = [
            {
                "date": item.date,
                "temp_celsius": item.temp_celsius,
                "condition": item.condition
            }
            for item in response.forecast
        ]
        
        return {"weather_forecast": forecast_list}
        
    except Exception as e:
        logger.exception(f"Unexpected error in weather node: {e}")
        errors = list(state.get("validation_errors", []))
        errors.append(f"Weather retrieval encountered an unexpected error: {str(e)}")
        return {
            "weather_forecast": [],
            "validation_errors": errors
        }


# Aliases for backward compatibility and naming preferences
weather_node = weather_forecast
weather_forecast_node = weather_forecast
