import os
from google.adk.agents import Agent
from abac_demo.tools.datastore import get_datastore_content
from google.adk.tools import ToolContext

# Agent Definitions
marketing_agent = Agent(
    name="marketing_agent",
    model="gemini-1.5-flash",
    instruction="You are a marketing assistant. You can only access the marketing datastore. Use the 'get_datastore_content' tool to retrieve marketing data.",
    description="An agent that can access marketing data.",
)

sales_agent = Agent(
    name="sales_agent",
    model="gemini-1.5-flash",
    instruction="You are a sales assistant. You can only access the sales datastore. Use the 'get_datastore_content' tool to retrieve sales data.",
    description="An agent that can access sales data.",
)

# Get agent type and access level from environment variables
agent_type = os.getenv("AGENT_TYPE", "marketing")
access_level = os.getenv("ACCESS_LEVEL", "employee")

# Select the agent based on the agent type
if agent_type == "marketing":
    root_agent = marketing_agent
    datastore_name = "marketing"
else:
    root_agent = sales_agent
    datastore_name = "sales"

def tool_wrapper(tool_context: ToolContext) -> dict:
    """Fetches content from the datastore based on the user's access level.
    
    Args:
        tool_context (ToolContext): The tool context.
        
    Returns:
        dict: The datastore content or an error message.
    """
    return get_datastore_content(datastore_name, access_level, tool_context)

root_agent.tools = [tool_wrapper]
