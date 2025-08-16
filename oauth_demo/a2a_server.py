from a2a.server.apps.rest.fastapi_app import A2ARESTFastAPIApplication
from a2a.types import AgentCard, AgentCapabilities, AgentSkill
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.request_handlers import DefaultRequestHandler
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
import os
import logging
from dotenv import load_dotenv
from agent_executor import OAuthAgentExecutor
import uvicorn
import agent


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

host="0.0.0.0"
port=int(os.environ.get("PORT",10003))
PUBLIC_URL=os.environ.get("PUBLIC_URL", f"http://{host}:{port}")

runner = Runner(
    agent=agent.root_agent,
    session_service=InMemorySessionService(),
    artifact_service=InMemoryArtifactService(),
    memory_service=InMemoryMemoryService(),
    app_name="oauth_demo",
)

agent_card = AgentCard(
    agent_id="oauth_agent",
    name="OAuth Agent",
    description="An agent that can access BigQuery using OAuth.",
    capabilities=AgentCapabilities(),
    skills=[AgentSkill(name="query", id="query", description="Perform a query", tags=[])],
    public_url=PUBLIC_URL,
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    url=PUBLIC_URL,
    version="1.0",
    securitySchemes={
        "google_oauth": {
            "type": "openIdConnect",
            "openIdConnectUrl": "https://accounts.google.com/.well-known/openid-configuration",
            "description": "Google OAuth 2.0"
        }
    },
    security=[
        {
            "google_oauth": [
                "openid",
                "profile",
                "email",
                "https://www.googleapis.com/auth/bigquery"
            ]
        }
    ]
)

http_handler = DefaultRequestHandler(
    agent_executor=OAuthAgentExecutor(runner, agent_card),
    task_store=InMemoryTaskStore(),
)

app = A2ARESTFastAPIApplication(
    agent_card=agent_card,
    http_handler=http_handler,
).build()


if __name__ == "__main__":
    print("Available routes:")
    for route in app.routes:
        print(f"  {route.path}")
    uvicorn.run(app, host=host, port=port)