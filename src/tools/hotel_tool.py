import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from config import llm
from src.tools.search_tool import search_web

logger = logging.getLogger(__name__)


class HotelItem(BaseModel):
    """
    Structured details for a single hotel recommendation.
    """
    name: str = Field(..., description="Name of the hotel or lodging.")
    price: Optional[float] = Field(None, description="Approximate price per night in USD if available.")
    rating: Optional[float] = Field(None, description="Guest rating score (typically on a 0-5 scale) if available.")
    location: str = Field(..., description="Location, neighborhood, or general address of the hotel.")


class HotelSearchOutput(BaseModel):
    """
    List of structured hotel recommendations extracted by the LLM.
    """
    hotels: List[HotelItem] = Field(..., description="List of recommended hotels.")


# Configure structured output LLM
structured_hotels_llm = llm.with_structured_output(HotelSearchOutput)

HOTELS_SYSTEM_PROMPT = (
    "You are a hospitality data extraction specialist.\n"
    "Based on the following search results for hotels in {location}, extract 3 to 5 recommended hotels.\n"
    "For each hotel, extract the name, approximate price per night in USD (if available), guest rating (if available), "
    "and general location/neighborhood. Do not invent any pricing or ratings; leave them as null if not specified.\n\n"
    "Search Context:\n"
    "{search_context}\n"
)

HOTELS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", HOTELS_SYSTEM_PROMPT)
])

hotels_extraction_chain = HOTELS_PROMPT | structured_hotels_llm


class HotelSearchResponse(BaseModel):
    """
    Final structured response container returned by the hotel tool.
    """
    success: bool = Field(..., description="Whether the hotel search and extraction was successful.")
    message: str = Field(..., description="Status message or error explanation.")
    hotels: List[HotelItem] = Field(default_factory=list, description="Discovered hotel options.")


def get_hotels(location: str) -> HotelSearchResponse:
    """
    Gathers hotel recommendations for a target location by executing a web search 
    and extracting structured details (name, price, rating, location) via ChatGroq.
    
    Args:
        location (str): The destination city or region.
        
    Returns:
        HotelSearchResponse: Structured response containing hotel recommendations or error details.
    """
    search_query = f"best hotels accommodations lodging prices ratings in {location}"
    
    try:
        # 1. Search the web for hotels
        search_res = search_web(search_query, max_results=5)
        
        if not search_res.success or not search_res.results:
            logger.warning(f"Web search returned no results for query '{search_query}'. Using fallback LLM retrieval.")
            search_context = "No search results available."
        else:
            search_context = "\n\n".join([
                f"Title: {r.title}\nContent: {r.content}\nURL: {r.url}"
                for r in search_res.results
            ])
            
        # 2. Invoke structured LLM to parse recommendations
        extraction: HotelSearchOutput = hotels_extraction_chain.invoke({
            "location": location,
            "search_context": search_context
        })
        
        return HotelSearchResponse(
            success=True,
            message="Hotel recommendations retrieved successfully.",
            hotels=extraction.hotels
        )
        
    except Exception as e:
        logger.exception(f"Error during hotel search and extraction: {e}")
        return HotelSearchResponse(
            success=False,
            message=f"Hotel search failed: {str(e)}"
        )


try:
    from langchain_core.tools import tool
    
    @tool
    def hotel_search_tool(location: str) -> str:
        """
        Gathers hotel recommendations for a target location.
        
        Args:
            location: The city or region name (e.g. 'Paris' or 'Tokyo').
            
        Returns:
            str: Readable format of hotel recommendations.
        """
        response = get_hotels(location)
        if not response.success or not response.hotels:
            return f"Failed to retrieve hotel recommendations: {response.message}"
            
        lines = [f"Hotel Recommendations in {location}:"]
        for idx, h in enumerate(response.hotels, 1):
            price_str = f"${h.price:.2f}/night" if h.price else "Price not available"
            rating_str = f"★ {h.rating}" if h.rating else "Rating not available"
            lines.append(f"{idx}. {h.name} ({location})\n   Price: {price_str} | Rating: {rating_str} | Location: {h.location}")
        return "\n".join(lines)
except ImportError:
    pass
