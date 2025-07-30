import os
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.cloud import modelarmor_v1
from google.genai import types as genai_types
from model_armor_demo_full.tools.sensitive_tool import handle_sensitive_data

# Set the GOOGLE_CLOUD_PROJECT environment variable
GOOGLE_CLOUD_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "zinc-forge-302418")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
TEMPLATE_ID = "ma-all-low"

# Worker Agent
worker_agent = Agent(
    name="worker_agent",
    model="gemini-2.5-flash",
    instruction="You are a worker agent. You can handle sensitive data.",
    description="An agent that can handle sensitive data.",
    tools=[handle_sensitive_data],
)

# Model Armor Callback
async def model_armor_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """Intercepts and blocks requests containing sensitive information."""
    client = modelarmor_v1.ModelArmorAsyncClient(
        client_options={"api_endpoint": "modelarmor.us-central1.rep.googleapis.com"}
    )
    template_name = client.template_path(GOOGLE_CLOUD_PROJECT_ID, GOOGLE_CLOUD_LOCATION, TEMPLATE_ID)
    prompt_data = modelarmor_v1.DataItem(text=llm_request.contents[-1].parts[0].text)
    request = modelarmor_v1.SanitizeUserPromptRequest(
        name=template_name,
        user_prompt_data=prompt_data,
    )
    response = await client.sanitize_user_prompt(request=request)
    if response.sanitization_result.filter_match_state == 2:
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
