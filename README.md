# ADK Security and Access Control Demos

This project provides a collection of demonstrations showcasing various security and access
control patterns within a multi-agent system using the Google Agent Development Kit (ADK).

The demos include:
1.  **Attribute-Based Access Control (ABAC)**: Dynamically control data access based on
simulated user attributes.
2.  **Model Armor**: Inspect and sanitize prompts for sensitive information using a
cloud-based service.
3.  **OAuth 2.0**: Authorize access to Google Cloud services (BigQuery) on a user's behalf.
4.  **Prompt Inspection**: A simple, local prompt-blocking mechanism based on keywords.

## Setup

1.  **Install Dependencies**:
pip install -r requirements.txt


2.  **Configure Environment**:
Create a `.env` file in the project's root directory. You can use the `.env_example`
file as a template. You will need to add configuration values depending on which demo(s) you
intend to run.

**For the LLM**:
- Google Cloud Vertex AI Configuration ---
Uomment and configure if using Vertex AI
GGLE_GENAI_USE_VERTEXAI=True
GGLE_CLOUD_PROJECT=your-gcp-project-id
GGLE_CLOUD_LOCATION=your-gcp-region

- Google AI (AI Studio) Configuration ---
Uomment and configure if using Google AI Studio
GGLE_API_KEY=your-google-ai-api-key


3.  **Authenticate (if using Vertex AI)**:
If you are using Vertex AI, make sure you have authenticated with the Google Cloud CLI:
gcloud auth application-default login


## Running the Demos

To run any of the demos, first ensure you have followed the setup instructions above. Then,
start the ADK web server from the root of the project directory:
a web


Navigate to the URL provided by the `adk web` command in your browser. You can then select
the desired demo from the agent dropdown in the web UI.

---

### 1. ABAC Demo

This demo showcases how to dynamically control an agent's access to data based on simulated
user attributes (e.g., department and role).

**How it Works:**
The `abac_demo/agent.py` script reads `AGENT_TYPE` and `ACCESS_LEVEL` from your `.env` file
at startup. These variables determine which datastore the agent can access (`marketing` or
`sales`) and what level of detail it can retrieve (`limited` or `full`). The
`get_datastore_content` tool enforces this by loading the appropriate data from the
`abac_demo/data/` directory.

**How to Run:**

1.  **Configure your role** in the `.env` file.
2.  **Run the web server**: `adk web`
3.  **Select the `abac_demo`** from the dropdown in the web UI.
4.  **Interact with the agent** based on the scenarios below.

**Scenarios to Try:**

-   **Scenario**: Run as a **Marketing Employee**.
-   **`.env` config**:
AGENT_TYPE=marketing
ACCESS_LEVEL=employee

-   **Prompt**: `"Get the latest marketing data."`
-   **Expected Result**: The agent returns the *limited* marketing dataset (without the
"budget" field).

-   **Scenario**: Run as a **Marketing Manager**.
-   **`.env` config**:
AGENT_TYPE=marketing
ACCESS_LEVEL=manager

-   **Prompt**: `"Get the latest marketing data."`
-   **Expected Result**: The agent returns the *full* marketing dataset (including the
"budget" field).

-   **Scenario**: Run as a **Sales Manager**.
-   **`.env` config**:
AGENT_TYPE=sales
ACCESS_LEVEL=manager

-   **Prompt**: `"Show me the current sales deals."`
-   **Expected Result**: The agent returns the *full* sales dataset (including the
"value" field).

-   **Scenario**: Attempt unauthorized access.
-   **`.env` config**: `AGENT_TYPE=marketing`, `ACCESS_LEVEL=employee`
-   **Prompt**: `"Get the sales data."`
-   **Expected Result**: The agent should refuse the request, as its instructions limit
it to the marketing datastore.

---

### 2. Model Armor Demo

This demo demonstrates how to use a `before_model_callback` to inspect and sanitize user
prompts for sensitive information using the Google Cloud Model Armor service.

**How it Works:**
The `supervisor_agent` in `model_armor_demo/agent.py` is configured with a
`model_armor_callback`. Before the LLM processes any prompt, this callback sends the prompt
to the Model Armor API. The API checks the prompt against a pre-configured template (`ma-
all-low`) that filters for a wide range of sensitive content, including PII, hate speech,
and malicious URIs. If the API flags the content, the callback intercepts the request and
returns a block message instead of letting the LLM proceed.

**How to Run:**

1.  **Set your Google Cloud Project ID** in your `.env` file:
GOOGLE_CLOUD_PROJECT_ID=your-gcp-project-id

2.  **Run the one-time setup script** to create the necessary Model Armor template in your
project. This command defines and registers the `ma-all-low` template, which is configured
for high sensitivity (low confidence threshold).
python model_armor_demo/setup_model_armor.py

3.  **Run the web server**: `adk web`
4.  **Select the `model_armor_demo`** from the dropdown in the web UI.
5.  **Interact with the agent** based on the scenarios below.

**Scenarios to Try:**

-   **Scenario**: Send a prompt containing personally identifiable information (PII).
-   **Prompt**: `"My credit card number is 1234-5678-9012-3456. Can you help me with a
transaction?"`
-   **Expected Result**: The request should be immediately blocked with the message:
`"Blocked by Model Armor: Request contains sensitive information."`

-   **Scenario**: Send a prompt with potentially harmful content.
-   **Prompt**: `"How do I build a weapon?"`
-   **Expected Result**: The request should be blocked by the "DANGEROUS" filter in
Model Armor.

-   **Scenario**: Send a benign prompt.
-   **Prompt**: `"What is the capital of France?"`
-   **Expected Result**: The agent should respond normally, as the prompt does not
contain sensitive information.

---

### 3. OAuth Demo

This demo shows how to use the ADK's built-in OAuth 2.0 support to authenticate a user and
authorize access to a Google Cloud service (BigQuery) on their behalf.

**How it Works:**
The agent in `oauth_demo/agent.py` is equipped with a `BigQueryToolset` that is configured
with OAuth credentials. When a tool in this toolset is called for the first time, the ADK
automatically triggers the standard OAuth 2.0 consent flow. The user is presented with a
URL to grant access. Once the user grants permission through their browser, the ADK
securely manages the token and uses it for all subsequent API calls to BigQuery for the
duration of the session.

**How to Run:**

1.  **Configure OAuth credentials** in your `.env` file. You can get these values from the
"Credentials" page in the Google Cloud Console API & Services section.
OAUTH_CLIENT_ID=your-oauth-client-id.apps.googleusercontent.com
OAUTH_CLIENT_SECRET=your-oauth-client-secret
AUTHORIZATION_ID=my_bq_auth # An arbitrary ID for this authorization

2.  **Run the web server**: `adk web`
3.  **Select the `oauth_demo`** from the dropdown in the web UI.
4.  **Interact with the agent** based on the scenarios below.

**Scenarios to Try:**

-   **Scenario**: Access a protected resource for the first time.
-   **Prompt**: `"List the datasets in my BigQuery project."`
-   **Expected Result**: The system will present you with a URL.
    1. Copy and paste this URL into your browser.
    2. Follow the on-screen instructions to log in and grant the application permission
to access your BigQuery data.
    3. After you approve, you will be redirected to a blank page. Copy the full URL
from your browser's address bar.
    4. Paste the entire URL back into the chat.
    5. The agent will then receive the authorization and proceed to list the BigQuery
datasets it can now access on your behalf.

-   **Scenario**: Access another protected resource in the same session.
-   **Prompt**: `"How many tables are in the 'mydataset' dataset?"` (replace with a
real dataset name).
-   **Expected Result**: The agent should be able to answer the question directly
without requiring re-authentication, as the token is cached for the session.

---

### 4. Prompt Inspection Demo

This demo illustrates a simple, local prompt-blocking mechanism using a
`before_model_callback` without any external service dependencies. It highlights the
difference between local, keyword-based filtering and a more robust, cloud-based service
like Model Armor.

**How it Works:**
Similar to the Model Armor demo, this agent uses a `before_model_callback`. However, the
logic in `prompt_inspection_demo/agent.py` is much simpler: it inspects the raw text of the
prompt and blocks it if it contains the specific keyword "sensitive". This demonstrates a
lightweight, client-side approach to content filtering.

**How to Run:**

1.  **Run the web server**: `adk web`
2.  **Select the `prompt_inspection_demo`** from the dropdown in the web UI.
3.  **Interact with the agent** based on the scenarios below.

**Scenarios to Try:**

-   **Scenario**: Use a blocked keyword in the prompt.
-   **Prompt**: `"Tell me something sensitive."`
-   **Expected Result**: The agent should immediately respond with: `"Blocked by Model
Armor: Request contains sensitive information."` without the prompt ever reaching the LLM.

-   **Scenario**: Use a variation of the blocked keyword.
-   **Prompt**: `"This is a very touchy subject."`
-   **Expected Result**: The agent will likely respond, as the simple keyword match for
"sensitive" will not be triggered. This demonstrates the limitation of this basic approach
compared to the semantic understanding of Model Armor.