import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from config import llm
from src.tools.search_tool import search_web

logger = logging.getLogger(__name__)


class TransportItem(BaseModel):
    """
    Structured details for a single transportation option.
    """
    type: str = Field(..., description="Type of transit (e.g., Bus, Train, Taxi, Metro, Rental Vehicle).")
    provider: str = Field(..., description="Name of the carrier or service provider (e.g., Eurostar, Local Metro, Hertz).")
    route: str = Field(..., description="Route details or airport/station connection coverage.")
    price: Optional[float] = Field(None, description="Estimated price or average fare in USD if available.")
    booking_link: Optional[str] = Field(None, description="URL link to purchase or book if available.")


class TransportSearchOutput(BaseModel):
    """
    List of structured transport recommendations extracted by the LLM.
    """
    transport: List[TransportItem] = Field(..., description="List of transit options.")


# Configure structured output LLM
structured_transport_llm = llm.with_structured_output(TransportSearchOutput)

TRANSPORT_SYSTEM_PROMPT = (
    "You are a transportation logistics analyst.\n"
    "Based on the following search results for public and private transit in {location}, "
    "extract 3 to 5 options including buses, trains, taxis, metro, and rental vehicles.\n"
    "For each option, extract the type of transport, service provider, route/coverage, "
    "approximate cost in USD (if available), and booking link (if available). Do not invent "
    "any data; leave fields as null if not specified.\n\n"
    "Search Context:\n"
    "{search_context}\n"
)

TRANSPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", TRANSPORT_SYSTEM_PROMPT)
])

transport_extraction_chain = TRANSPORT_PROMPT | structured_transport_llm


class TransportSearchResponse(BaseModel):
    """
    Final structured response container returned by the transport tool.
    """
    success: bool = Field(..., description="Whether the transport search and extraction was successful.")
    message: str = Field(..., description="Status message or error explanation.")
    transport: List[TransportItem] = Field(default_factory=list, description="Discovered transport options.")


def get_transport(location: str) -> TransportSearchResponse:
    """
    Gathers local transportation recommendations (buses, trains, taxis, metro, rentals) 
    for a location by executing web searches and extracting structured options via ChatGroq.
    
    Args:
        location (str): The target destination name.
        
    Returns:
        TransportSearchResponse: Structured response containing transit recommendations.
    """
    search_query = f"public transport trains buses taxis metro rental vehicles in {location}"
    
    try:
        # 1. Search the web for transit
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
        extraction: TransportSearchOutput = transport_extraction_chain.invoke({
            "location": location,
            "search_context": search_context
        })
        
        return TransportSearchResponse(
            success=True,
            message="Transportation recommendations retrieved successfully.",
            transport=extraction.transport
        )
        
    except Exception as e:
        logger.exception(f"Error during transport search and extraction: {e}")
        return TransportSearchResponse(
            success=False,
            message=f"Transport search failed: {str(e)}"
        )


try:
    from langchain_core.tools import tool
    
    @tool
    def transport_search_tool(location: str) -> str:
        """
        Gathers local transportation recommendations (buses, trains, taxis, metro, rentals) for a location.
        
        Args:
            location: The city or region name (e.g. 'Paris' or 'Tokyo').
            
        Returns:
            str: Readable format of transportation recommendations.
        """
        response = get_transport(location)
        if not response.success or not response.transport:
            return f"Failed to retrieve transport recommendations: {response.message}"
            
        lines = [f"Transit Recommendations in {location}:"]
        for idx, t in enumerate(response.transport, 1):
            price_str = f"${t.price:.2f}" if t.price else "Price not available"
            lines.append(f"{idx}. {t.provider} ({t.type})\n   Route: {t.route} | Cost: {price_str}")
        return "\n".join(lines)
except ImportError:
    pass
