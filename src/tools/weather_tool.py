import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from collections import Counter
import httpx
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class WeatherForecastItem(BaseModel):
    """
    Represents weather condition summaries for a specific date.
    """
    date: str = Field(..., description="Target date in YYYY-MM-DD format.")
    temp_celsius: float = Field(..., description="Average expected temperature in Celsius for the day.")
    condition: str = Field(..., description="Text summary of weather conditions (e.g., Sunny, Rainy, Cloudy).")


class WeatherForecastResponse(BaseModel):
    """
    Structured response for the weather forecast query.
    """
    location: str = Field(..., description="Target city/location query.")
    success: bool = Field(..., description="Whether the weather query was successful.")
    forecast: List[WeatherForecastItem] = Field(default_factory=list, description="List of daily weather forecasts.")
    error_message: Optional[str] = Field(None, description="Error details if the query failed.")


def get_weather_forecast(location: str) -> WeatherForecastResponse:
    """
    Fetches the daily weather forecast for a given location using the Open-Meteo API.
    
    This function first resolves the location coordinates (lat/lon) via Open-Meteo's
    free geocoding API, and then queries the daily forecast data using those coordinates.
    
    Args:
        location (str): The city or location name (e.g., "Paris" or "Tokyo, JP").
        
    Returns:
        WeatherForecastResponse: Structured Pydantic model with success status, location, forecast list, or error details.
    """
    try:
        # 1. Geocode location using Open-Meteo's free geocoding service
        geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
        geocoding_params = {
            "name": location,
            "count": 1,
            "language": "en",
            "format": "json"
        }
        
        with httpx.Client(timeout=10.0) as client:
            geo_resp = client.get(geocoding_url, params=geocoding_params)
            
        if geo_resp.status_code != 200:
            return WeatherForecastResponse(
                location=location,
                success=False,
                error_message=f"Geocoding API request failed with status code {geo_resp.status_code}."
            )
            
        geo_data = geo_resp.json()
        results = geo_data.get("results")
        if not results:
            return WeatherForecastResponse(
                location=location,
                success=False,
                error_message=f"Location '{location}' not resolved by geocoding API."
            )
            
        lat = results[0]["latitude"]
        lon = results[0]["longitude"]
        timezone = results[0].get("timezone", "UTC")
        resolved_location = f"{results[0]['name']}, {results[0].get('country', '')}"
        
        # 2. Retrieve forecast from Open-Meteo
        base_url = os.getenv("OPENMETEO_BASE_URL", "https://api.open-meteo.com")
        forecast_url = f"{base_url}/v1/forecast"
        
        forecast_params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,weathercode",
            "timezone": timezone
        }
        
        with httpx.Client(timeout=10.0) as client:
            fore_resp = client.get(forecast_url, params=forecast_params)
            
        if fore_resp.status_code != 200:
            return WeatherForecastResponse(
                location=resolved_location,
                success=False,
                error_message=f"Weather forecast request failed with status code {fore_resp.status_code}."
            )
            
        fore_data = fore_resp.json()
        daily = fore_data.get("daily", {})
        
        time_list = daily.get("time", [])
        temp_max_list = daily.get("temperature_2m_max", [])
        temp_min_list = daily.get("temperature_2m_min", [])
        weathercodes = daily.get("weathercode", [])
        
        def map_wmo_code(code: int) -> str:
            # Standard WMO Weather interpretation codes
            if code == 0:
                return "Sunny"
            elif code in [1, 2, 3]:
                return "Partly Cloudy"
            elif code in [45, 48]:
                return "Foggy"
            elif code in [51, 53, 55]:
                return "Drizzle"
            elif code in [61, 63, 65]:
                return "Rainy"
            elif code in [66, 67]:
                return "Freezing Rain"
            elif code in [71, 73, 75]:
                return "Snowy"
            elif code in [77]:
                return "Snow Grains"
            elif code in [80, 81, 82]:
                return "Rain Showers"
            elif code in [85, 86]:
                return "Snow Showers"
            elif code in [95]:
                return "Thunderstorm"
            elif code in [96, 99]:
                return "Thunderstorm with Hail"
            return "Unknown"
            
        forecasts = []
        for i in range(len(time_list)):
            date_str = time_list[i]
            max_t = temp_max_list[i] if i < len(temp_max_list) else 0.0
            min_t = temp_min_list[i] if i < len(temp_min_list) else 0.0
            code = weathercodes[i] if i < len(weathercodes) else 0
            
            avg_t = round((max_t + min_t) / 2, 1)
            condition = map_wmo_code(code)
            
            forecasts.append(WeatherForecastItem(
                date=date_str,
                temp_celsius=avg_t,
                condition=condition
            ))
            
        return WeatherForecastResponse(
            location=resolved_location,
            success=True,
            forecast=forecasts
        )
        
    except httpx.RequestError as e:
        error_msg = f"Network communication error during weather fetch: {e}"
        logger.exception(error_msg)
        return WeatherForecastResponse(
            location=location,
            success=False,
            error_message=error_msg
        )
    except Exception as e:
        error_msg = f"Unexpected error during weather fetch: {e}"
        logger.exception(error_msg)
        return WeatherForecastResponse(
            location=location,
            success=False,
            error_message=error_msg
        )


try:
    from langchain_core.tools import tool
    
    @tool
    def weather_forecast_tool(location: str) -> str:
        """
        Fetches the 5-day weather forecast for a given location.
        
        Args:
            location: The location or city to get the weather forecast for (e.g. 'Paris, France').
            
        Returns:
            str: A structured, readable string containing the 5-day weather forecast.
        """
        response = get_weather_forecast(location)
        if not response.success:
            return f"Failed to get weather forecast for '{location}': {response.error_message}"
        
        lines = [f"Weather forecast for {response.location}:"]
        for f in response.forecast:
            lines.append(f"- {f.date}: {f.temp_celsius}°C, {f.condition}")
        return "\n".join(lines)
except ImportError:
    pass
