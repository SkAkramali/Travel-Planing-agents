import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from config import llm
from src.tools.search_tool import search_web

logger = logging.getLogger(__name__)


class NetworkOperatorItem(BaseModel):
    """
    Structured details for a single mobile network operator or travel eSIM plan.
    """
    operator_name: str = Field(..., description="Name of the mobile operator or eSIM provider (e.g., Orange, Softbank, Airalo).")
    plan_name: Optional[str] = Field(None, description="Name of the specific data plan or package.")
    data_limit: Optional[str] = Field(None, description="Included data limit (e.g., 10GB, Unlimited).")
    price: Optional[float] = Field(None, description="Price of the plan in USD if available.")
    validity_days: Optional[int] = Field(None, description="Validity duration of the plan in days if available.")
    quality: str = Field(..., description="Estimated network quality and coverage (e.g., Excellent 5G, Good 4G, Spotty).")


class NetworkSearchOutput(BaseModel):
    """
    List of structured network operators and connectivity statements extracted by the LLM.
    """
    operators: List[NetworkOperatorItem] = Field(..., description="List of mobile network options.")
    internet_availability: str = Field(..., description="Statement summarizing overall public internet, wifi availability, and travel connectivity advice.")


# Configure structured output LLM
structured_network_llm = llm.with_structured_output(NetworkSearchOutput)

NETWORK_SYSTEM_PROMPT = (
    "You are a global connectivity and telecommunication specialist.\n"
    "Based on the following search results for mobile networks, eSIMs, and tourist SIM cards in {location}, "
    "extract 3 to 5 options including available operators, travel plans, data limits, prices, and validity.\n"
    "Additionally, summarize the overall tourist internet availability (e.g., public wifi availability, coverage quality) "
    "as a concise advice statement.\n\n"
    "Search Context:\n"
    "{search_context}\n"
)

NETWORK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", NETWORK_SYSTEM_PROMPT)
])

network_extraction_chain = NETWORK_PROMPT | structured_network_llm


class NetworkSearchResponse(BaseModel):
    """
    Final structured response container returned by the network tool.
    """
    success: bool = Field(..., description="Whether the network search and extraction was successful.")
    message: str = Field(..., description="Status message or error explanation.")
    operators: List[NetworkOperatorItem] = Field(default_factory=list, description="Discovered mobile operators.")
    internet_availability: str = Field(default="Internet is widely available via eSIM or local SIMs.", description="Summary statement of connectivity advice.")


def get_mobile_networks(location: str) -> NetworkSearchResponse:
    """
    Gathers local mobile network, SIM, and eSIM recommendations for a target location by executing 
    web searches and extracting structured data via ChatGroq.
    
    Args:
        location (str): The target destination name.
        
    Returns:
        NetworkSearchResponse: Structured response containing eSIM and operator recommendations.
    """
    search_query = f"tourist sim card esim mobile operators coverage internet quality in {location}"
    
    try:
        # 1. Search the web for connectivity
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
        extraction: NetworkSearchOutput = network_extraction_chain.invoke({
            "location": location,
            "search_context": search_context
        })
        
        return NetworkSearchResponse(
            success=True,
            message="Mobile network options retrieved successfully.",
            operators=extraction.operators,
            internet_availability=extraction.internet_availability
        )
        
    except Exception as e:
        logger.exception(f"Error during network search and extraction: {e}")
        return NetworkSearchResponse(
            success=False,
            message=f"Mobile networks search failed: {str(e)}"
        )


try:
    from langchain_core.tools import tool
    
    @tool
    def network_search_tool(location: str) -> str:
        """
        Gathers local mobile network, SIM, and eSIM recommendations for a location.
        
        Args:
            location: The city or region name (e.g. 'Paris' or 'Tokyo').
            
        Returns:
            str: Readable format of mobile network and connectivity suggestions.
        """
        response = get_mobile_networks(location)
        if not response.success or not response.operators:
            return f"Failed to retrieve mobile network options: {response.message}"
            
        lines = [f"Connectivity in {location}:", f"Advice: {response.internet_availability}", "\nOperators/eSIM plans:"]
        for idx, op in enumerate(response.operators, 1):
            price_str = f"${op.price:.2f}" if op.price else "Price not available"
            valid_str = f"{op.validity_days} days" if op.validity_days else "validity not specified"
            limit_str = op.data_limit if op.data_limit else "data not specified"
            lines.append(f"{idx}. {op.operator_name} ({op.plan_name or 'Travel plan'}) | Price: {price_str} | Data: {limit_str} | Validity: {valid_str} | Coverage: {op.quality}")
        return "\n".join(lines)
except ImportError:
    pass
