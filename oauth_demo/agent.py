import os
from enum import Enum

from google.adk.agents import llm_agent
from google.adk.auth import AuthCredentialTypes
from google.adk.tools.bigquery import BigQueryCredentialsConfig
from google.adk.tools.bigquery import BigQueryToolset
from google.adk.tools.bigquery.config import BigQueryToolConfig
from google.adk.tools.bigquery.config import WriteMode
from google.adk.agents.callback_context import CallbackContext
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()
class EnvVars(str, Enum):
    """Environment variables for the agent."""

    CLIENT_ID = os.getenv("OAUTH_CLIENT_ID")
    CLIENT_SECRET = os.getenv("OAUTH_CLIENT_SECRET")
    MODEL = "gemini-2.5-flash"
    AUTH_ID = os.getenv("AUTHORIZATION_ID")


CREDENTIALS_TYPE = AuthCredentialTypes.OAUTH2

def whoami(callback_context: CallbackContext):
  """Prints the user's email address."""
  print("**** PREREQ SETUP ****")
  access_token = callback_context.state[f"temp:{EnvVars.AUTH_ID}"]
  creds = Credentials(token=access_token)
  user_info_service = build("oauth2", "v2", credentials=creds)
  user_info = user_info_service.userinfo().get().execute()
  user_email = user_info.get("email")
  print(f"The person who is just authorized is {user_email}")


# Define BigQuery tool config
tool_config = BigQueryToolConfig(write_mode=WriteMode.ALLOWED)

credentials_config = BigQueryCredentialsConfig(
    client_id=EnvVars.CLIENT_ID,
    client_secret=EnvVars.CLIENT_SECRET,
    # scopes=["https://www.googleapis.com/auth/drive",
    #         "https://www.googleapis.com/auth/bigquery"],
)

bigquery_toolset = BigQueryToolset(
    credentials_config=credentials_config, bigquery_tool_config=tool_config
)

root_agent = llm_agent.Agent(
    model=EnvVars.MODEL,
    name="hello_agent",
    description=(
        "Agent to answer questions about BigQuery data and models and"
        " execute SQL queries."
    ),
    instruction="""\
        You are a data science agent with access to several BigQuery tools.
        Make use of those tools to answer the user's questions.
    """,
    tools=[bigquery_toolset],
)
