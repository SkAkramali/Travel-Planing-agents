import logging
import re
from typing import List, Optional
import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WikipediaSearchResult(BaseModel):
    """
    Represents a single Wikipedia search result snippet.
    """
    title: str = Field(..., description="Title of the Wikipedia page.")
    snippet: str = Field(..., description="Short text snippet matching the search query.")
    page_id: int = Field(..., description="Wikipedia page ID.")
    url: str = Field(..., description="Direct link to the Wikipedia page.")


class WikipediaSearchResponse(BaseModel):
    """
    Structured response for the Wikipedia search query.
    """
    query: str = Field(..., description="The search query.")
    success: bool = Field(..., description="Whether the search was successful.")
    results: List[WikipediaSearchResult] = Field(default_factory=list, description="List of search results.")
    error_message: Optional[str] = Field(None, description="Error details if the search failed.")


class WikipediaSummaryResponse(BaseModel):
    """
    Structured response for a Wikipedia page summary.
    """
    title: str = Field(..., description="Title of the page.")
    description: Optional[str] = Field(None, description="Short description of the topic.")
    extract: Optional[str] = Field(None, description="Text extract/summary of the page.")
    url: Optional[str] = Field(None, description="Desktop URL of the Wikipedia page.")
    success: bool = Field(..., description="Whether the query was successful.")
    error_message: Optional[str] = Field(None, description="Error details if the query failed.")


def search_wikipedia(query: str, limit: int = 5) -> WikipediaSearchResponse:
    """
    Searches Wikipedia for articles matching the query.
    
    Args:
        query (str): The search query.
        limit (int): Maximum number of search results to return (default: 5).
        
    Returns:
        WikipediaSearchResponse: Structured response containing search results or error details.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json"
    }
    headers = {
        "User-Agent": "TravelPlannerAgent/1.0 (contact@example.com)"
    }
    
    try:
        with httpx.Client(timeout=10.0, headers=headers) as client:
            response = client.get(url, params=params)
            
        if response.status_code != 200:
            return WikipediaSearchResponse(
                query=query,
                success=False,
                error_message=f"Wikipedia search returned status code {response.status_code}"
            )
            
        data = response.json()
        search_data = data.get("query", {}).get("search", [])
        
        results = []
        for item in search_data:
            title = item.get("title", "")
            page_id = item.get("pageid", 0)
            snippet = item.get("snippet", "")
            
            # Clean HTML tags from Wikipedia snippet using regex
            clean_snippet = re.sub(r'<[^>]*>', '', snippet)
            
            # Format desktop URL
            formatted_title = title.replace(" ", "_")
            url_str = f"https://en.wikipedia.org/wiki/{formatted_title}"
            
            results.append(WikipediaSearchResult(
                title=title,
                snippet=clean_snippet,
                page_id=page_id,
                url=url_str
            ))
            
        return WikipediaSearchResponse(
            query=query,
            success=True,
            results=results
        )
        
    except httpx.RequestError as e:
        error_msg = f"Network communication error during Wikipedia search: {e}"
        logger.exception(error_msg)
        return WikipediaSearchResponse(
            query=query,
            success=False,
            error_message=error_msg
        )
    except Exception as e:
        error_msg = f"Unexpected error during Wikipedia search: {e}"
        logger.exception(error_msg)
        return WikipediaSearchResponse(
            query=query,
            success=False,
            error_message=error_msg
        )


def get_wikipedia_summary(title: str) -> WikipediaSummaryResponse:
    """
    Fetches the summary paragraph and description of a specific Wikipedia page by title.
    
    Args:
        title (str): The exact title of the Wikipedia page.
        
    Returns:
        WikipediaSummaryResponse: Structured response containing summary content or error details.
    """
    # URL encode title for REST API
    formatted_title = title.replace(" ", "_")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{formatted_title}"
    headers = {
        "User-Agent": "TravelPlannerAgent/1.0 (contact@example.com)"
    }
    
    try:
        with httpx.Client(timeout=10.0, headers=headers, follow_redirects=True) as client:
            response = client.get(url)
            
        if response.status_code == 404:
            return WikipediaSummaryResponse(
                title=title,
                success=False,
                error_message=f"Wikipedia page '{title}' not found."
            )
        elif response.status_code != 200:
            return WikipediaSummaryResponse(
                title=title,
                success=False,
                error_message=f"Wikipedia summary API returned status code {response.status_code}"
            )
            
        data = response.json()
        return WikipediaSummaryResponse(
            title=data.get("title", title),
            description=data.get("description"),
            extract=data.get("extract"),
            url=data.get("content_urls", {}).get("desktop", {}).get("page"),
            success=True
        )
        
    except httpx.RequestError as e:
        error_msg = f"Network communication error during Wikipedia summary fetch: {e}"
        logger.exception(error_msg)
        return WikipediaSummaryResponse(
            title=title,
            success=False,
            error_message=error_msg
        )
    except Exception as e:
        error_msg = f"Unexpected error during Wikipedia summary fetch: {e}"
        logger.exception(error_msg)
        return WikipediaSummaryResponse(
            title=title,
            success=False,
            error_message=error_msg
        )


try:
    from langchain_core.tools import tool
    
    @tool
    def wikipedia_search_tool(query: str) -> str:
        """
        Searches Wikipedia for articles matching the query.
        
        Args:
            query: The search query to look up on Wikipedia.
            
        Returns:
            str: Readable format of Wikipedia search results.
        """
        response = search_wikipedia(query, limit=3)
        if not response.success:
            return f"Failed to search Wikipedia for '{query}': {response.error_message}"
        if not response.results:
            return f"No Wikipedia articles found matching query: {query}"
        
        lines = [f"Wikipedia Search Results for '{query}':"]
        for idx, res in enumerate(response.results, 1):
            lines.append(f"{idx}. {res.title}\n   Summary: {res.snippet}\n   URL: {res.url}")
        return "\n\n".join(lines)
        
    @tool
    def wikipedia_summary_tool(title: str) -> str:
        """
        Fetches the summary paragraph and details of a specific Wikipedia page by title.
        
        Args:
            title: The exact title of the Wikipedia page.
            
        Returns:
            str: Readable format of Wikipedia page summary.
        """
        response = get_wikipedia_summary(title)
        if not response.success:
            return f"Failed to fetch Wikipedia summary for '{title}': {response.error_message}"
        
        output = f"Wikipedia Title: {response.title}\n"
        if response.description:
            output += f"Description: {response.description}\n"
        if response.extract:
            output += f"Summary:\n{response.extract}\n"
        if response.url:
            output += f"URL: {response.url}\n"
        return output
except ImportError:
    pass
