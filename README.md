# Attribute-Based Access Control (ABAC) Demo with ADK

This project demonstrates a practical implementation of Attribute-Based Access Control (ABAC) within a multi-agent system using the Google Agent Development Kit (ADK). The demo showcases how to dynamically control an agent's access to data based on simulated user attributes (e.g., department and role) that are configured via environment variables.

## How It Works

This demo uses a layered approach to enforce access control:

1.  **Attribute Simulation**: The user's attributes (their department and role) are simulated using environment variables in the `.env` file (`AGENT_TYPE` and `ACCESS_LEVEL`).

2.  **Agent Orchestration**: The main `abac_demo/agent.py` script reads these environment variables at startup. It selects the appropriate agent (e.g., `marketing_agent`) and creates a `tool_wrapper` function. This wrapper captures the user's attributes for the current session.

3.  **Tool-Based Enforcement**: The `get_datastore_content` tool acts as the final gatekeeper. It receives the user's attributes from the `tool_wrapper` and uses them to fetch the correct data (`marketing` or `sales`) and the correct version of that data (`limited` or `full`).

This architecture ensures that the agent's capabilities are dynamically configured at the beginning of each session, providing a flexible and secure way to manage data access.

## Project Structure

```
/abac_demo
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
└── README.md
```

## Setup

1.  **Install Dependencies**:

    ```bash
    pip install google-adk
    ```

2.  **Configure Environment**:

    Create a `.env` file in the project's root directory. You can use the following as a template:

    ```env
    # Defines which agent to use ('marketing' or 'sales')
    AGENT_TYPE=marketing

    # Defines the user's access level ('employee' or 'manager')
    ACCESS_LEVEL=employee

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

## Running the Demo

1.  **Start the ADK Web Server**:

    From the root of the `abac_demo` directory, run:

    ```bash
    adk web
    ```

2.  **Open the Web Interface**:

    Navigate to the URL provided by the `adk web` command in your browser.

3.  **Interact with the Agent**:

    You can now chat with the agent and test the access control by changing the values in your `.env` file and restarting the server.

    **Example Scenarios**:

    -   **Marketing Employee** (`AGENT_TYPE=marketing`, `ACCESS_LEVEL=employee`):
        -   *Your Prompt*: `"Get the latest marketing data."`
        -   *Expected Result*: The agent returns the *limited* marketing dataset (without budget information).

    -   **Marketing Manager** (`AGENT_TYPE=marketing`, `ACCESS_LEVEL=manager`):
        -   *Your Prompt*: `"Get the latest marketing data."`
        -   *Expected Result*: The agent returns the *full* marketing dataset (including budget information).

    -   **Sales Employee** (`AGENT_TYPE=sales`, `ACCESS_LEVEL=employee`):
        -   *Your Prompt*: `"Show me the current sales pipeline."`
        -   *Expected Result*: The agent returns the *limited* sales dataset (without deal values).

    -   **Sales Manager** (`AGENT_TYPE=sales`, `ACCESS_LEVEL=manager`):
        -   *Your Prompt*: `"Show me the current sales pipeline."`
        -   *Expected Result*: The agent returns the *full* sales dataset (including deal values).

    -   **Attempting Unauthorized Access**:
        -   If you are running the `marketing_agent`, try asking: `"Get the sales data."`
        -   *Expected Result*: The agent should refuse the request, as its instructions limit it to the marketing datastore.