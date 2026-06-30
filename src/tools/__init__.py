"""
Tools package providing weather forecasting, Wikipedia searching/summarization, and web search utilities.

This package exposes both direct reusable Python functions (which return structured Pydantic objects) 
and LangChain tool-compatible wrapper instances.
"""

from .weather_tool import (
    get_weather_forecast,
    WeatherForecastResponse,
    WeatherForecastItem,
)
from .wiki_tool import (
    search_wikipedia,
    get_wikipedia_summary,
    WikipediaSearchResponse,
    WikipediaSummaryResponse,
    WikipediaSearchResult,
)
from .search_tool import (
    search_web,
    SearchResponse,
    SearchResultItem,
)
from .flight_tool import (
    get_flights,
    FlightSearchResponse,
    FlightOptionItem,
)
from .hotel_tool import (
    get_hotels,
    HotelSearchResponse,
    HotelItem,
)
from .transport_tool import (
    get_transport,
    TransportSearchResponse,
    TransportItem,
)
from .network_tool import (
    get_mobile_networks,
    NetworkSearchResponse,
    NetworkOperatorItem,
)

# Safely expose LangChain tools if langchain_core is installed
try:
    from .weather_tool import weather_forecast_tool
    from .wiki_tool import wikipedia_search_tool, wikipedia_summary_tool
    from .search_tool import web_search_tool
    from .flight_tool import flight_search_tool
    from .hotel_tool import hotel_search_tool
    from .transport_tool import transport_search_tool
    from .network_tool import network_search_tool
    
    langchain_tools = [
        weather_forecast_tool,
        wikipedia_search_tool,
        wikipedia_summary_tool,
        web_search_tool,
        flight_search_tool,
        hotel_search_tool,
        transport_search_tool,
        network_search_tool,
    ]
except ImportError:
    weather_forecast_tool = None
    wikipedia_search_tool = None
    wikipedia_summary_tool = None
    web_search_tool = None
    flight_search_tool = None
    hotel_search_tool = None
    transport_search_tool = None
    network_search_tool = None
    langchain_tools = []

__all__ = [
    # Weather
    "get_weather_forecast",
    "WeatherForecastResponse",
    "WeatherForecastItem",
    "weather_forecast_tool",
    
    # Wikipedia
    "search_wikipedia",
    "get_wikipedia_summary",
    "WikipediaSearchResponse",
    "WikipediaSummaryResponse",
    "WikipediaSearchResult",
    "wikipedia_search_tool",
    "wikipedia_summary_tool",
    
    # Web Search
    "search_web",
    "SearchResponse",
    "SearchResultItem",
    "web_search_tool",

    # Flights
    "get_flights",
    "FlightSearchResponse",
    "FlightOptionItem",
    "flight_search_tool",

    # Hotels
    "get_hotels",
    "HotelSearchResponse",
    "HotelItem",
    "hotel_search_tool",

    # Transport
    "get_transport",
    "TransportSearchResponse",
    "TransportItem",
    "transport_search_tool",

    # Network operator
    "get_mobile_networks",
    "NetworkSearchResponse",
    "NetworkOperatorItem",
    "network_search_tool",
    
    # Collection of langchain tools
    "langchain_tools",
]
