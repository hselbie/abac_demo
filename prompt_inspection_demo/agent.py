
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types as genai_types
from model_armor_demo.tools.sensitive_tool import handle_sensitive_data

# Worker Agent
worker_agent = Agent(
    name="worker_agent",
    model="gemini-2.5-flash",
    instruction="You are a worker agent. You can handle sensitive data.",
    description="An agent that can handle sensitive data.",
    tools=[handle_sensitive_data],
)

# Model Armor Callback
def model_armor_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """Intercepts and blocks requests containing sensitive information."""
    if "sensitive" in llm_request.contents[-1].parts[0].text:
        return LlmResponse(
            content=genai_types.Content(
                parts=[
                    genai_types.Part(
                        text="Blocked by Model Armor: Request contains sensitive information."
                    )
                ]
            )
        )
    return None

# Supervisor Agent
supervisor_agent = Agent(
    name="supervisor_agent",
    model="gemini-2.5-flash",
    instruction="You are a supervisor agent. You delegate tasks to the worker agent.",
    description="An agent that delegates tasks.",
    sub_agents=[worker_agent],
    before_model_callback=model_armor_callback,
)

root_agent = supervisor_agent
