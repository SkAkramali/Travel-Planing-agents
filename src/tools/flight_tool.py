import os
import logging
from typing import List, Optional, Tuple
import httpx
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class FlightOptionItem(BaseModel):
    """
    Structured schema representing a flight recommendation option.
    """
    flight_id: str = Field(..., description="Unique identifier for tracking the flight option.")
    airline: str = Field(..., description="Name of the airline carrier.")
    departure_time: str = Field(..., description="ISO format departure datetime (e.g. YYYY-MM-DDTHH:MM:SS).")
    arrival_time: str = Field(..., description="ISO format arrival datetime.")
    price: float = Field(..., description="Total flight ticket cost in USD.")
    booking_link: Optional[str] = Field(None, description="Direct URL to purchase/book the flight.")


class FlightSearchResponse(BaseModel):
    """
    Structured response for the flight search query.
    """
    success: bool = Field(..., description="Whether the flight search query was successful.")
    message: str = Field(..., description="Status message or error explanation.")
    flights: List[FlightOptionItem] = Field(default_factory=list, description="Discovered flight options.")


# Static fallback database of major cities to Skyscanner SkyId and EntityId
FALLBACK_AIRPORTS = {
    "london": ("LON", "27544008"),
    "paris": ("PARI", "27539733"),
    "tokyo": ("TYO", "27539486"),
    "new york": ("NYC", "27537542"),
    "rome": ("ROM", "27539665"),
    "barcelona": ("BCN", "27544093"),
    "singapore": ("SIN", "27540209"),
    "dubai": ("DXB", "27539655")
}


def resolve_airport(location: str, rapidapi_key: str) -> Optional[Tuple[str, str]]:
    """
    Resolves a location query (e.g. 'Paris') to Skyscanner SkyId and EntityId.
    
    Args:
        location (str): Name of the city/location.
        rapidapi_key (str): RapidAPI secret token.
        
    Returns:
        Optional[Tuple[str, str]]: Tuple of (SkyId, EntityId) if successfully resolved, else None.
    """
    # Check fallback map first (case-insensitive)
    loc_clean = location.strip().lower()
    for key, val in FALLBACK_AIRPORTS.items():
        if key in loc_clean:
            return val
            
    # Attempt geocoding using RapidAPI SkyScraper endpoint
    url = "https://sky-scrapper.p.rapidapi.com/api/v1/flights/searchAirport"
    params = {"query": location, "locale": "en-US"}
    headers = {
        "x-rapidapi-key": rapidapi_key,
        "x-rapidapi-host": "sky-scrapper.p.rapidapi.com"
    }
    
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, params=params, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("data", [])
            if results:
                sky_id = results[0].get("skyId")
                entity_id = results[0].get("entityId")
                if sky_id and entity_id:
                    return sky_id, entity_id
    except Exception as e:
        logger.warning(f"Failed to geocode location '{location}' via SkyScraper Airport API: {e}")
        
    return None


def get_flights(origin: str, destination: str, date: str) -> FlightSearchResponse:
    """
    Retrieves flights between origin and destination for a given date using SkyScraper API on RapidAPI.
    
    If RAPIDAPI_KEY is not configured or queries fail, it returns a structured response 
    declaring flight search is currently unavailable, ensuring execution does not crash.
    
    Args:
        origin (str): Origin city/airport name.
        destination (str): Destination city/airport name.
        date (str): Departure date in YYYY-MM-DD format.
        
    Returns:
        FlightSearchResponse: Structured response container.
    """
    rapidapi_key = os.getenv("RAPIDAPI_KEY")
    if not rapidapi_key or not rapidapi_key.strip():
        logger.warning("RAPIDAPI_KEY is missing. Returning fallback flight search unavailable message.")
        return FlightSearchResponse(
            success=False,
            message="Flight search is currently unavailable."
        )
        
    # Resolve origin and destination airports
    origin_resolved = resolve_airport(origin, rapidapi_key)
    dest_resolved = resolve_airport(destination, rapidapi_key)
    
    if not origin_resolved or not dest_resolved:
        logger.warning(f"Could not resolve airport coordinates for '{origin}' -> '{destination}' using Skyscanner API.")
        return FlightSearchResponse(
            success=False,
            message="Flight search is currently unavailable (could not resolve airport codes)."
        )
        
    origin_sky_id, origin_entity_id = origin_resolved
    dest_sky_id, dest_entity_id = dest_resolved
    
    # Query Skyscanner search flights
    url = "https://sky-scrapper.p.rapidapi.com/api/v1/flights/searchFlights"
    params = {
        "originSkyId": origin_sky_id,
        "destinationSkyId": dest_sky_id,
        "originEntityId": origin_entity_id,
        "destinationEntityId": dest_entity_id,
        "date": date,
        "cabinClass": "economy",
        "adults": 1,
        "sortBy": "best"
    }
    headers = {
        "x-rapidapi-key": rapidapi_key,
        "x-rapidapi-host": "sky-scrapper.p.rapidapi.com"
    }
    
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url, params=params, headers=headers)
            
        if resp.status_code != 200:
            logger.error(f"SkyScraper flight search returned status code {resp.status_code}.")
            return FlightSearchResponse(
                success=False,
                message="Flight search is currently unavailable (API communication error)."
            )
            
        data = resp.json()
        itineraries = data.get("data", {}).get("itineraries", [])
        
        flights = []
        for idx, item in enumerate(itineraries[:5]):  # Return top 5 flights
            price_raw = item.get("price", {}).get("raw", 0.0)
            legs = item.get("legs", [])
            
            airline = "Unknown Airline"
            departure_time = f"{date}T00:00:00"
            arrival_time = f"{date}T00:00:00"
            
            if legs:
                leg = legs[0]
                departure_time = leg.get("departure", departure_time)
                arrival_time = leg.get("arrival", arrival_time)
                carriers = leg.get("carriers", {}).get("marketing", [])
                if carriers:
                    airline = carriers[0].get("name", airline)
                    
            flight_id = item.get("id", f"fl-mock-{idx}")
            
            flights.append(FlightOptionItem(
                flight_id=flight_id,
                airline=airline,
                departure_time=departure_time,
                arrival_time=arrival_time,
                price=float(price_raw),
                booking_link="https://www.skyscanner.net"
            ))
            
        return FlightSearchResponse(
            success=True,
            message="Flight search completed successfully.",
            flights=flights
        )
        
    except Exception as e:
        logger.exception(f"Error querying Skyscanner SkyScraper flight search: {e}")
        return FlightSearchResponse(
            success=False,
            message="Flight search is currently unavailable."
        )


try:
    from langchain_core.tools import tool
    
    @tool
    def flight_search_tool(origin: str, destination: str, date: str) -> str:
        """
        Searches flights between an origin and destination for a given date.
        
        Args:
            origin: The origin city or airport code (e.g. 'London' or 'LHR').
            destination: The destination city or airport code (e.g. 'Paris' or 'CDG').
            date: The departure date in YYYY-MM-DD format (e.g. '2027-04-10').
            
        Returns:
            str: Readable format of flight search recommendations.
        """
        response = get_flights(origin, destination, date)
        if not response.success or not response.flights:
            return f"Failed to fetch flight recommendations: {response.message}"
            
        lines = [f"Flight options from {origin} to {destination} on {date}:"]
        for idx, fl in enumerate(response.flights, 1):
            lines.append(f"{idx}. {fl.airline} | Dep: {fl.departure_time} | Arr: {fl.arrival_time} | Price: ${fl.price:.2f} USD")
        return "\n".join(lines)
except ImportError:
    pass
