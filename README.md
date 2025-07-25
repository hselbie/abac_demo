# Attribute-Based Access Control (ABAC) and OAuth Demos with ADK

This project demonstrates practical implementations of Attribute-Based Access Control (ABAC) and OAuth 2.0 within a multi-agent system using the Google Agent Development Kit (ADK).

## ABAC Demo

The ABAC demo showcases how to dynamically control an agent's access to data based on simulated user attributes (e.g., department and role) that are configured via environment variables.

### How It Works

This demo uses a layered approach to enforce access control:

1.  **Attribute Simulation**: The user's attributes (their department and role) are simulated using environment variables in the `.env` file (`AGENT_TYPE` and `ACCESS_LEVEL`).

2.  **Agent Orchestration**: The main `abac_demo/agent.py` script reads these environment variables at startup. It selects the appropriate agent (e.g., `marketing_agent`) and creates a `tool_wrapper` function. This wrapper captures the user's attributes for the current session.

3.  **Tool-Based Enforcement**: The `get_datastore_content` tool acts as the final gatekeeper. It receives the user's attributes from the `tool_wrapper` and uses them to fetch the correct data (`marketing` or `sales`) and the correct version of that data (`limited` or `full`).

This architecture ensures that the agent's capabilities are dynamically configured at the beginning of each session, providing a flexible and secure way to manage data access.

## OAuth 2.0 Demo with BigQuery

The OAuth demo shows how to use the ADK's built-in OAuth 2.0 support to authenticate a user and authorize access to a Google Cloud service (BigQuery) on their behalf.

### How It Works

1.  **Tool Configuration**: The `oauth_demo/agent.py` script defines a `BigQueryToolset`. This toolset is configured with `BigQueryCredentialsConfig`, which includes the OAuth `client_id` and `client_secret`.

2.  **OAuth Flow**: When the agent needs to use a BigQuery tool for the first time in a session, the ADK automatically initiates the OAuth 2.0 authorization code flow. The user is prompted to grant consent.

3.  **Token Storage**: Once the user grants consent, the ADK securely receives and stores the access token in the session's temporary state (`temp:`).

4.  **Authenticated Tool Calls**: Subsequent calls to the BigQuery tools in the same session will use the stored access token for authentication.

## Project Structure

```
/
├── .env
├── abac_demo/
│   ├── __init__.py
│   ├── agent.py         # Main agent logic and orchestration
│   ├── data/
│   │   ├── marketing_data.json
│   │   └── sales_data.json
│   └── tools/
│       ├── __init__.py
│       └── datastore.py   # Data access tool (the "gatekeeper")
├── oauth_demo/
│   ├── __init__.py
│   └── agent.py         # OAuth demo agent and tool setup
└── README.md
```

## Setup

1.  **Install Dependencies**:

    ```bash
    pip install google-adk google-api-python-client google-auth-httplib2 google-auth-oauthlib python-dotenv
    ```

2.  **Configure Environment**:

    Create a `.env` file in the project's root directory. You can use the `.env_example` file as a template.

    **For the ABAC Demo**:
    ```env
    # Defines which agent to use ('marketing' or 'sales')
    AGENT_TYPE=marketing

    # Defines the user's access level ('employee' or 'manager')
    ACCESS_LEVEL=employee
    ```

    **For the OAuth Demo**:
    ```env
    # --- OAuth 2.0 Configuration ---
    OAUTH_CLIENT_ID=your-oauth-client-id.apps.googleusercontent.com
    OAUTH_CLIENT_SECRET=your-oauth-client-secret
    AUTHORIZATION_ID=your-chosen-authorization-id # e.g., "my_bq_auth"
    ```

    **For the LLM**:
    ```env
    # --- Google Cloud Vertex AI Configuration ---
    # Uncomment and configure if using Vertex AI
    # GOOGLE_GENAI_USE_VERTEXAI=True
    # GOOGLE_CLOUD_PROJECT=your-gcp-project-id
    # GOOGLE_CLOUD_LOCATION=your-gcp-region

    # --- Google AI (AI Studio) Configuration ---
    # Uncomment and configure if using Google AI Studio
    # GOOGLE_API_KEY=your-google-ai-api-key
    ```

3.  **Authenticate (if using Vertex AI)**:

    If you are using Vertex AI, make sure you have authenticated with the Google Cloud CLI:

    ```bash
    gcloud auth application-default login
    ```

## Running the Demos

1.  **Start the ADK Web Server**:

    From the root of the project directory, run:

    ```bash
    adk web
    ```

2.  **Open the Web Interface**:

    Navigate to the URL provided by the `adk web` command in your browser.

3.  **Interact with the Agents**:

    You can now select either the `abac_demo` or `oauth_demo` from the agent dropdown in the web UI.

    ### ABAC Demo Scenarios

    -   **Marketing Employee** (`AGENT_TYPE=marketing`, `ACCESS_LEVEL=employee`):
        -   *Your Prompt*: `"Get the latest marketing data."`
        -   *Expected Result*: The agent returns the *limited* marketing dataset (without budget information).

    -   **Marketing Manager** (`AGENT_TYPE=marketing`, `ACCESS_LEVEL=manager`):
        -   *Your Prompt*: `"Get the latest marketing data."`
        -   *Expected Result*: The agent returns the *full* marketing dataset (including budget information).

    -   **Attempting Unauthorized Access**:
        -   If you are running the `marketing_agent`, try asking: `"Get the sales data."`
        -   *Expected Result*: The agent should refuse the request, as its instructions limit it to the marketing datastore.

    ### OAuth Demo Scenario

    -   **Select the `oauth_demo` agent.**
    -   *Your Prompt*: `"List the datasets in my BigQuery project."`
    -   *Expected Result*: The system will present you with a URL to begin the OAuth consent flow. After you approve access, the agent will list the BigQuery datasets it can access on your behalf.