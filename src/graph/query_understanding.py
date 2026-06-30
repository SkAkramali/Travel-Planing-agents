import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from config import llm
from src.graph.state import TravelPlannerState

logger = logging.getLogger(__name__)


class QueryUnderstandingOutput(BaseModel):
    """
    Structured extraction representing query understanding.
    """
    destination: Optional[str] = Field(
        None, description="The destination city or country (e.g., 'Paris, France' or 'Japan')."
    )
    budget: Optional[float] = Field(
        None, description="Total maximum budget allocated for the trip in USD."
    )
    number_of_days: Optional[int] = Field(
        None, description="Total number of days for the trip."
    )
    trip_type: Optional[str] = Field(
        None, description="The style or type of trip (e.g., solo, family, romantic, business, adventure, relaxation)."
    )


# Configure the structured LLM instance
structured_llm = llm.with_structured_output(QueryUnderstandingOutput)

# Prompt template to parse travel details from conversation messages
QUERY_UNDERSTANDING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an AI Travel Assistant specialized in query understanding.\n"
        "Your task is to analyze the conversation history and extract the following travel preferences:\n"
        "1. Destination: The city, country, or region the traveler wants to visit.\n"
        "2. Budget: The total maximum budget in USD (extract or convert to numeric float, e.g. 5000).\n"
        "3. Number of days: The duration of the trip as an integer (e.g. 7).\n"
        "4. Trip type: The style/category of the trip (e.g., solo, family, romantic, business, adventure, relaxation).\n\n"
        "Only extract values that are explicitly mentioned or clearly implied in the messages. "
        "Leave fields as null if they are not specified or cannot be inferred."
    ),
    ("placeholder", "{messages}")
])

# Reusable extraction chain
query_understanding_chain = QUERY_UNDERSTANDING_PROMPT | structured_llm


def query_understanding(state: TravelPlannerState) -> Dict[str, Any]:
    """
    LangGraph node that parses travel preferences from conversation messages.
    
    This node extracts destination, budget, number of days, and trip type from
    the state messages, updates the state's `trip_details`, and returns the updates.
    
    Args:
        state (TravelPlannerState): The current LangGraph state containing conversation messages.
        
    Returns:
        Dict[str, Any]: A state update dictionary containing the modified 'trip_details'.
    """
    messages = state.get("messages", [])
    current_details = state.get("trip_details") or {}
    
    # Initialize updated details with existing state values
    updated_details = dict(current_details)
    
    if not messages:
        logger.warning("No messages found in the state. Skipping query understanding extraction.")
        return {"trip_details": updated_details}
        
    try:
        # Run extraction chain
        extraction: QueryUnderstandingOutput = query_understanding_chain.invoke(
            {"messages": messages}
        )
        
        # Update details with non-null extracted fields
        if extraction.destination:
            updated_details["destination"] = extraction.destination
        if extraction.budget is not None:
            updated_details["budget"] = extraction.budget
        if extraction.number_of_days is not None:
            updated_details["number_of_days"] = extraction.number_of_days
        if extraction.trip_type:
            updated_details["trip_type"] = extraction.trip_type
            
    except Exception as e:
        logger.exception(f"Error during query understanding node execution: {e}")
        # In case of error, log it under validation_errors in the state
        errors = list(state.get("validation_errors", []))
        errors.append(f"Query Understanding extraction failed: {str(e)}")
        return {
            "trip_details": updated_details,
            "validation_errors": errors
        }
        
    return {"trip_details": updated_details}


# Alias for backward compatibility
query_understanding_node = query_understanding

