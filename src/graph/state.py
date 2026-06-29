from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class TripDetails(TypedDict):
    """
    Structured schema representing the core travel parameters gathered from the user.
    """
    destination: str
    """Target city/country for the trip (e.g., 'Paris, France')."""
    
    origin: str
    """Departure city/airport (e.g., 'New York, JFK')."""
    
    start_date: str
    """Trip start date in YYYY-MM-DD format."""
    
    end_date: str
    """Trip end date in YYYY-MM-DD format."""
    
    budget: float
    """Total maximum budget allocated for the trip in USD."""
    
    interests: List[str]
    """List of travel interests/hobbies (e.g., ['art', 'adventure', 'gastronomy'])."""
    
    num_travelers: int
    """Number of people traveling."""


class FlightOption(TypedDict):
    """
    Represents a flight recommendation retrieved from flight tools/APIs.
    """
    flight_id: str
    """Unique identifier for tracking the flight option."""
    
    airline: str
    """Name of the airline carrier."""
    
    departure_time: str
    """ISO format departure datetime."""
    
    arrival_time: str
    """ISO format arrival datetime."""
    
    price: float
    """Total flight ticket cost in USD."""
    
    booking_link: Optional[str]
    """Direct URL to purchase/book the flight (optional)."""


class HotelOption(TypedDict):
    """
    Represents an accommodation option retrieved from hotel search tools.
    """
    hotel_id: str
    """Unique identifier for tracking the hotel option."""
    
    name: str
    """Name of the hotel or lodging establishment."""
    
    price_per_night: float
    """Daily cost of lodging in USD."""
    
    rating: float
    """Average guest score/rating (typically on a 0-5 scale)."""
    
    address: str
    """Physical address of the accommodation."""
    
    booking_link: Optional[str]
    """Direct URL to book the lodging (optional)."""


class ActivityOption(TypedDict):
    """
    Represents a curated activity, tour, or attraction recommendation.
    """
    activity_id: str
    """Unique identifier for tracking the activity."""
    
    name: str
    """Name of the activity or point of interest."""
    
    description: str
    """Short description of the activity/excursion."""
    
    cost: float
    """Entrance fee or cost in USD (0.0 for free attractions)."""
    
    duration_hours: float
    """Estimated duration of the activity in hours."""
    
    category: str
    """Category classification (e.g., 'museum', 'nature', 'food')."""


class WeatherForecast(TypedDict):
    """
    Represents weather condition summaries for a specific date.
    """
    date: str
    """Target date in YYYY-MM-DD format."""
    
    temp_celsius: float
    """Average expected temperature in Celsius."""
    
    condition: str
    """Text summary of weather conditions (e.g., 'Sunny', 'Rainy', 'Cloudy')."""


class FinalItinerary(TypedDict):
    """
    Consolidated final travel itinerary.
    """
    daily_schedule: List[Dict[str, Any]]
    """Structured list containing sequential activities mapped to dates and times."""
    
    total_cost: float
    """Sum total cost of all flights, hotels, and selected activities in USD."""
    
    formatted_markdown: str
    """Pre-formatted Markdown presentation showing the summary, maps, flights, and agenda."""


class TravelPlannerState(TypedDict):
    """
    Global state schema passed between LangGraph nodes during execution.
    """
    
    # ----------------------------------------------------
    # Core Conversational and Input State
    # ----------------------------------------------------
    messages: Annotated[List[AnyMessage], add_messages]
    """Accumulates message history between the traveler and the planner agent."""
    
    trip_details: TripDetails
    """Extracted core travel specifications (destination, budget, dates, etc.)."""
    
    # ----------------------------------------------------
    # Intermediary API Search Results
    # ----------------------------------------------------
    flight_options: List[FlightOption]
    """Discovered flight options parsed from external search API calls."""
    
    hotel_options: List[HotelOption]
    """Discovered lodging options matching travel criteria and budget bounds."""
    
    activity_options: List[ActivityOption]
    """Sightseeing and attraction choices tailored to user interests."""
    
    weather_forecast: List[WeatherForecast]
    """Weather forecast snapshots mapped to travel dates at the destination."""
    
    # ----------------------------------------------------
    # Output and Orchestration State
    # ----------------------------------------------------
    final_itinerary: Optional[FinalItinerary]
    """The compiled final travel agenda. Defaults to None until constructed."""
    
    validation_errors: List[str]
    """Tracks constraint violations (e.g., budget exceeded, schedule clashes)."""
    
    # ----------------------------------------------------
    # Future-proofing and Extensibility
    # ----------------------------------------------------
    metadata: Dict[str, Any]
    """
    Flexible dictionary hook for future integrations, telemetry tracing, 
    analytics, or partner API variables.
    """
