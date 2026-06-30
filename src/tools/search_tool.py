import os
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class SearchResultItem(BaseModel):
    """
    Represents a single search result from Tavily.
    """
    title: str = Field(..., description="Title of the search result page.")
    url: str = Field(..., description="URL link to the search result.")
    content: str = Field(..., description="Snippet of content from the page.")
    score: float = Field(..., description="Relevance score of the result.")


class SearchResponse(BaseModel):
    """
    Structured response for the web search query.
    """
    query: str = Field(..., description="The search query.")
    success: bool = Field(..., description="Whether the search query was successful.")
    results: List[SearchResultItem] = Field(default_factory=list, description="List of search results.")
    error_message: Optional[str] = Field(None, description="Error details if the search failed.")


def search_web(query: str, max_results: int = 5) -> SearchResponse:
    """
    Performs a web search using the Tavily API, falling back to DuckDuckGo if Tavily is unavailable.
    
    Args:
        query (str): The search query.
        max_results (int): Maximum number of search results to return (default: 5).
        
    Returns:
        SearchResponse: Structured response containing search results or error details.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    
    # If Tavily key is present, attempt Tavily search
    if api_key and api_key.strip():
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth="basic"
            )
            
            results = []
            raw_results = response.get("results", [])
            for item in raw_results:
                results.append(SearchResultItem(
                    title=item.get("title", "No Title"),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    score=item.get("score", 0.0)
                ))
                
            return SearchResponse(
                query=query,
                success=True,
                results=results
            )
        except Exception as e:
            logger.warning(f"Tavily search failed, falling back to DuckDuckGo. Error: {e}")
            
    # Fallback to DuckDuckGo search
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            ddg_results = ddgs.text(query, max_results=max_results)
            if ddg_results:
                for idx, r in enumerate(ddg_results):
                    results.append(SearchResultItem(
                        title=r.get("title", "No Title"),
                        url=r.get("href", ""),
                        content=r.get("body", ""),
                        score=round(1.0 - (idx * 0.1), 2)
                    ))
        return SearchResponse(
            query=query,
            success=True,
            results=results
        )
    except Exception as e:
        error_msg = f"Web search failed (both Tavily and DuckDuckGo failed): {e}"
        logger.exception(error_msg)
        return SearchResponse(
            query=query,
            success=False,
            error_message=error_msg
        )



try:
    from langchain_core.tools import tool
    
    @tool
    def web_search_tool(query: str) -> str:
        """
        Searches the web for information matching the query.
        
        Args:
            query: The search query to lookup on the web.
            
        Returns:
            str: Readable format of web search results.
        """
        response = search_web(query, max_results=5)
        if not response.success:
            return f"Failed to perform search: {response.error_message}"
        if not response.results:
            return f"No search results found for query: {query}"
            
        lines = [f"Web Search Results for '{query}':"]
        for idx, res in enumerate(response.results, 1):
            lines.append(f"{idx}. {res.title}\n   Content: {res.content}\n   URL: {res.url}")
        return "\n\n".join(lines)
except ImportError:
    pass
