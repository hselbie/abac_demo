# Google Agent Development Kit (ADK) Python Cheatsheet

This document serves as a long-form, comprehensive reference for building, orchestrating, and deploying AI agents using the Python Agent Development Kit (ADK). It aims to cover every significant aspect with greater detail, more code examples, and in-depth best practices.

## Table of Contents

1.  [Core Concepts & Project Structure](#1-core-concepts--project-structure)
    *   1.1 ADK's Foundational Principles
    *   1.2 Essential Primitives
    *   1.3 Standard Project Layout
2.  [Agent Definitions (`LlmAgent`)](#2-agent-definitions-llmagent)
    *   2.1 Basic `LlmAgent` Setup
    *   2.2 Advanced `LlmAgent` Configuration
    *   2.3 LLM Instruction Crafting
3.  [Orchestration with Workflow Agents](#3-orchestration-with-workflow-agents)
    *   3.1 `SequentialAgent`: Linear Execution
    *   3.2 `ParallelAgent`: Concurrent Execution
    *   3.3 `LoopAgent`: Iterative Processes
4.  [Multi-Agent Systems & Communication](#4-multi-agent-systems--communication)
    *   4.1 Agent Hierarchy
    *   4.2 Inter-Agent Communication Mechanisms
    *   4.3 Common Multi-Agent Patterns
5.  [Building Custom Agents (`BaseAgent`)](#5-building-custom-agents-baseagent)
    *   5.1 When to Use Custom Agents
    *   5.2 Implementing `_run_async_impl`
6.  [Models: Gemini, LiteLLM, and Vertex AI](#6-models-gemini-litellm-and-vertex-ai)
    *   6.1 Google Gemini Models (AI Studio & Vertex AI)
    *   6.2 Other Cloud & Proprietary Models via LiteLLM
    *   6.3 Open & Local Models via LiteLLM (Ollama, vLLM)
    *   6.4 Customizing LLM API Clients
7.  [Tools: The Agent's Capabilities](#7-tools-the-agents-capabilities)
    *   7.1 Defining Function Tools: Principles & Best Practices
    *   7.2 The `ToolContext` Object: Accessing Runtime Information
    *   7.3 All Tool Types & Their Usage
8.  [Context, State, and Memory Management](#8-context-state-and-memory-management)
    *   8.1 The `Session` Object & `SessionService`
    *   8.2 `State`: The Conversational Scratchpad
    *   8.3 `Memory`: Long-Term Knowledge & Retrieval
    *   8.4 `Artifacts`: Binary Data Management
9.  [Runtime, Events, and Execution Flow](#9-runtime-events-and-execution-flow)
    *   9.1 The `Runner`: The Orchestrator
    *   9.2 The Event Loop: Core Execution Flow
    *   9.3 `Event` Object: The Communication Backbone
    *   9.4 Asynchronous Programming (Python Specific)
10. [Control Flow with Callbacks](#10-control-flow-with-callbacks)
    *   10.1 Callback Mechanism: Interception & Control
    *   10.2 Types of Callbacks
    *   10.3 Callback Best Practices
11. [Authentication for Tools](#11-authentication-for-tools)
    *   11.1 Core Concepts: `AuthScheme` & `AuthCredential`
    *   11.2 Interactive OAuth/OIDC Flows
    *   11.3 Custom Tool Authentication
12. [Deployment Strategies](#12-deployment-strategies)
    *   12.1 Local Development & Testing (`adk web`, `adk run`, `adk api_server`)
    *   12.2 Vertex AI Agent Engine
    *   12.3 Cloud Run
    *   12.4 Google Kubernetes Engine (GKE)
    *   12.5 CI/CD Integration
13. [Evaluation and Safety](#13-evaluation-and-safety)
    *   13.1 Agent Evaluation (`adk eval`)
    *   13.2 Safety & Guardrails
14. [Debugging, Logging & Observability](#14-debugging-logging--observability)
15. [Advanced I/O Modalities](#15-advanced-io-modalities)
16. [Performance Optimization](#16-performance-optimization)
17. [General Best Practices & Common Pitfalls](#17-general-best-practices--common-pitfalls)

---

## 1. Core Concepts & Project Structure

### 1.1 ADK's Foundational Principles

*   **Modularity**: Break down complex problems into smaller, manageable agents and tools.
*   **Composability**: Combine simple agents and tools to build sophisticated systems.
*   **Observability**: Detailed event logging and tracing capabilities to understand agent behavior.
*   **Extensibility**: Easily integrate with external services, models, and frameworks.
*   **Deployment-Agnostic**: Design agents once, deploy anywhere.

### 1.2 Essential Primitives

*   **`Agent`**: The core intelligent unit. Can be `LlmAgent` (LLM-driven) or `BaseAgent` (custom/workflow).
*   **`Tool`**: Callable function/class providing external capabilities (`FunctionTool`, `OpenAPIToolset`, etc.).
*   **`Session`**: A unique, stateful conversation thread with history (`events`) and short-term memory (`state`).
*   **`State`**: Key-value dictionary within a `Session` for transient conversation data.
*   **`Memory`**: Long-term, searchable knowledge base beyond a single session (`MemoryService`).
*   **`Artifact`**: Named, versioned binary data (files, images) associated with a session or user.
*   **`Runner`**: The execution engine; orchestrates agent activity and event flow.
*   **`Event`**: Atomic unit of communication and history; carries content and side-effect `actions`.
*   **`InvocationContext`**: The comprehensive root context object holding all runtime information for a single `run_async` call.

### 1.3 Standard Project Layout

A well-structured ADK project is crucial for maintainability and leveraging `adk` CLI tools.

```
your_project_root/
├── my_first_agent/             # Each folder is a distinct agent app
│   ├── __init__.py             # Makes `my_first_agent` a Python package (`from . import agent`)
│   ├── agent.py                # Contains `root_agent` definition and `LlmAgent`/WorkflowAgent instances
│   ├── tools.py                # Custom tool function definitions
│   ├── data/                   # Optional: static data, templates
│   └── .env                    # Environment variables (API keys, project IDs)
├── my_second_agent/
│   ├── __init__.py
│   └── agent.py
├── requirements.txt            # Project's Python dependencies (e.g., google-adk, litellm)
├── tests/                      # Unit and integration tests
│   ├── unit/
│   │   └── test_tools.py
│   └── integration/
│       └── test_my_first_agent.py
│       └── my_first_agent.evalset.json # Evaluation dataset for `adk eval`
└── main.py                     # Optional: Entry point for custom FastAPI server deployment
```
*   `adk web` and `adk run` automatically discover agents in subdirectories with `__init__.py` and `agent.py`.
*   `.env` files are automatically loaded by `adk` tools when run from the root or agent directory.

---

## 2. Agent Definitions (`LlmAgent`)

The `LlmAgent` is the cornerstone of intelligent behavior, leveraging an LLM for reasoning and decision-making.

### 2.1 Basic `LlmAgent` Setup

```python
from google.adk.agents import Agent

def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city."""
    # Mock implementation
    if city.lower() == "new york":
        return {"status": "success", "time": "10:30 AM EST"}
    return {"status": "error", "message": f"Time for {city} not available."}

my_first_llm_agent = Agent(
    name="time_teller_agent",
    model="gemini-2.5-flash", # Essential: The LLM powering the agent
    instruction="You are a helpful assistant that tells the current time in cities. Use the 'get_current_time' tool for this purpose.",
    description="Tells the current time in a specified city.", # Crucial for multi-agent delegation
    tools=[get_current_time] # List of callable functions/tool instances
)
```

### 2.2 Advanced `LlmAgent` Configuration

*   **`generate_content_config`**: Controls LLM generation parameters (temperature, token limits, safety).
    ```python
    from google.genai import types as genai_types
    from google.adk.agents import Agent

    gen_config = genai_types.GenerateContentConfig(
        temperature=0.2,            # Controls randomness (0.0-1.0), lower for more deterministic.
        top_p=0.9,                  # Nucleus sampling: sample from top_p probability mass.
        top_k=40,                   # Top-k sampling: sample from top_k most likely tokens.
        max_output_tokens=1024,     # Max tokens in LLM's response.
        stop_sequences=["## END"]   # LLM will stop generating if these sequences appear.
    )
    agent = Agent(
        # ... basic config ...
        generate_content_config=gen_config
    )
    ```

*   **`output_key`**: Automatically saves the agent's final text or structured (if `output_schema` is used) response to the `session.state` under this key. Facilitates data flow between agents.
    ```python
    agent = Agent(
        # ... basic config ...
        output_key="llm_final_response_text"
    )
    # After agent runs, session.state['llm_final_response_text'] will contain its output.
    ```

*   **`input_schema` & `output_schema`**: Define strict JSON input/output formats using Pydantic models.
    > **Warning**: Using `output_schema` forces the LLM to generate JSON and **disables** its ability to use tools or delegate to other agents.

#### **Example: Defining and Using Structured Output**

This is the most reliable way to make an LLM produce predictable, parseable JSON, which is essential for multi-agent workflows.

1.  **Define the Schema with Pydantic:**
    ```python
    from pydantic import BaseModel, Field
    from typing import Literal

    class SearchQuery(BaseModel):
        """Model representing a specific search query for web search."""
        search_query: str = Field(
            description="A highly specific and targeted query for web search."
        )

    class Feedback(BaseModel):
        """Model for providing evaluation feedback on research quality."""
        grade: Literal["pass", "fail"] = Field(
            description="Evaluation result. 'pass' if the research is sufficient, 'fail' if it needs revision."
        )
        comment: str = Field(
            description="Detailed explanation of the evaluation, highlighting strengths and/or weaknesses of the research."
        )
        follow_up_queries: list[SearchQuery] | None = Field(
            default=None,
            description="A list of specific, targeted follow-up search queries needed to fix research gaps. This should be null or empty if the grade is 'pass'."
        )
    ```
    *   **`BaseModel` & `Field`**: Define data types, defaults, and crucial `description` fields. These descriptions are sent to the LLM to guide its output.
    *   **`Literal`**: Enforces strict enum-like values (`"pass"` or `"fail"`), preventing the LLM from hallucinating unexpected values.

2.  **Assign the Schema to an `LlmAgent`:**
    ```python
    research_evaluator = LlmAgent(
        name="research_evaluator",
        model="gemini-2.5-pro",
        instruction="""You are a meticulous quality assurance analyst. Evaluate the research findings in 'section_research_findings' and be very critical.
        If you find significant gaps, assign a grade of 'fail', write a detailed comment, and generate 5-7 specific follow-up queries.
        If the research is thorough, grade it 'pass'.
        Your response must be a single, raw JSON object validating against the 'Feedback' schema.
        """,
        output_schema=Feedback, # This forces the LLM to output JSON matching the Feedback model.
        output_key="research_evaluation", # The resulting JSON object will be saved to state.
        disallow_transfer_to_peers=True, # Prevents this agent from delegating. Its job is only to evaluate.
    )
    ```

*   **`include_contents`**: Controls whether the conversation history is sent to the LLM.
    *   `'default'` (default): Sends relevant history.
    *   `'none'`: Sends no history; agent operates purely on current turn's input and `instruction`. Useful for stateless API wrapper agents.
    ```python
    agent = Agent(..., include_contents='none')
    ```

*   **`planner`**: Assign a `BasePlanner` instance (e.g., `ReActPlanner`) to enable multi-step reasoning and planning. (Advanced, covered in Multi-Agents).

*   **`executor`**: Assign a `BaseCodeExecutor` (e.g., `BuiltInCodeExecutor`) to allow the agent to execute code blocks.
    ```python
    from google.adk.code_executors import BuiltInCodeExecutor
    agent = Agent(
        name="code_agent",
        model="gemini-2.5-flash",
        instruction="Write and execute Python code to solve math problems.",
        executor=[BuiltInCodeExecutor] # Allows agent to run Python code
    )
    ```

*   **Callbacks**: Hooks for observing and modifying agent behavior at key lifecycle points (`before_model_callback`, `after_tool_callback`, etc.). (Covered in Callbacks).

### 2.3 LLM Instruction Crafting (`instruction`)

The `instruction` is critical. It guides the LLM's behavior, persona, and tool usage. The following examples demonstrate powerful techniques for creating specialized, reliable agents.

**Best Practices & Examples:**

*   **Be Specific & Concise**: Avoid ambiguity.
*   **Define Persona & Role**: Give the LLM a clear role.
*   **Constrain Behavior & Tool Use**: Explicitly state what the LLM should *and should not* do.
*   **Define Output Format**: Tell the LLM *exactly* what its output should look like, especially when not using `output_schema`.
*   **Dynamic Injection**: Use `{state_key}` to inject runtime data from `session.state` into the prompt.
*   **Iteration**: Test, observe, and refine instructions.

**Example 1: Constraining Tool Use and Output Format**
```python
import datetime
from google.adk.tools import google_search   


plan_generator = LlmAgent(
    model="gemini-2.5-flash",
    name="plan_generator",
    description="Generates a 4-5 line action-oriented research plan.",
    instruction=f"""
    You are a research strategist. Your job is to create a high-level RESEARCH PLAN, not a summary.
    **RULE: Your output MUST be a bulleted list of 4-5 action-oriented research goals or key questions.**
    - A good goal starts with a verb like "Analyze," "Identify," "Investigate."
    - A bad output is a statement of fact like "The event was in April 2024."
    **TOOL USE IS STRICTLY LIMITED:**
    Your goal is to create a generic, high-quality plan *without searching*.
    Only use `google_search` if a topic is ambiguous and you absolutely cannot create a plan without it.
    You are explicitly forbidden from researching the *content* or *themes* of the topic.
    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    """,
    tools=[google_search],
)
```

**Example 2: Injecting Data from State and Specifying Custom Tags**
This agent's `instruction` relies on data placed in `session.state` by previous agents.
```python
report_composer = LlmAgent(
    model="gemini-2.5-pro",
    name="report_composer_with_citations",
    include_contents="none", # History not needed; all data is injected.
    description="Transforms research data and a markdown outline into a final, cited report.",
    instruction="""
    Transform the provided data into a polished, professional, and meticulously cited research report.

    ---
    ### INPUT DATA
    *   Research Plan: `{research_plan}`
    *   Research Findings: `{section_research_findings}`
    *   Citation Sources: `{sources}`
    *   Report Structure: `{report_sections}`

    ---
    ### CRITICAL: Citation System
    To cite a source, you MUST insert a special citation tag directly after the claim it supports.

    **The only correct format is:** `<cite source="src-ID_NUMBER" />`

    ---
    ### Final Instructions
    Generate a comprehensive report using ONLY the `<cite source="src-ID_NUMBER" />` tag system for all citations.
    The final report must strictly follow the structure provided in the **Report Structure** markdown outline.
    Do not include a "References" or "Sources" section; all citations must be in-line.
    """,
    output_key="final_cited_report",
)
```

---

## 3. Orchestration with Workflow Agents

Workflow agents (`SequentialAgent`, `ParallelAgent`, `LoopAgent`) provide deterministic control flow, combining LLM capabilities with structured execution. They do **not** use an LLM for their own orchestration logic.

### 3.1 `SequentialAgent`: Linear Execution

Executes `sub_agents` one after another in the order defined. The `InvocationContext` is passed along, allowing state changes to be visible to subsequent agents.

```python
from google.adk.agents import SequentialAgent, Agent

# Agent 1: Summarizes a document and saves to state
summarizer = Agent(
    name="DocumentSummarizer",
    model="gemini-2.5-flash",
    instruction="Summarize the provided document in 3 sentences.",
    output_key="document_summary" # Output saved to session.state['document_summary']
)

# Agent 2: Generates questions based on the summary from state
question_generator = Agent(
    name="QuestionGenerator",
    model="gemini-2.5-flash",
    instruction="Generate 3 comprehension questions based on this summary: {document_summary}",
    # 'document_summary' is dynamically injected from session.state
)

document_pipeline = SequentialAgent(
    name="SummaryQuestionPipeline",
    sub_agents=[summarizer, question_generator], # Order matters!
    description="Summarizes a document then generates questions."
)
```

### 3.2 `ParallelAgent`: Concurrent Execution

Executes `sub_agents` simultaneously. Useful for independent tasks to reduce overall latency. All sub-agents share the same `session.state`.

```python
from google.adk.agents import ParallelAgent, Agent

# Agents to fetch data concurrently
fetch_stock_price = Agent(name="StockPriceFetcher", ..., output_key="stock_data")
fetch_news_headlines = Agent(name="NewsFetcher", ..., output_key="news_data")
fetch_social_sentiment = Agent(name="SentimentAnalyzer", ..., output_key="sentiment_data")

# Agent to merge results (runs after ParallelAgent, usually in a SequentialAgent)
merger_agent = Agent(
    name="ReportGenerator",
    model="gemini-2.5-flash",
    instruction="Combine stock data: {stock_data}, news: {news_data}, and sentiment: {sentiment_data} into a market report."
)

# Pipeline to run parallel fetching then sequential merging
market_analysis_pipeline = SequentialAgent(
    name="MarketAnalyzer",
    sub_agents=[
        ParallelAgent(
            name="ConcurrentFetch",
            sub_agents=[fetch_stock_price, fetch_news_headlines, fetch_social_sentiment]
        ),
        merger_agent # Runs after all parallel agents complete
    ]
)
```
*   **Concurrency Caution**: When parallel agents write to the same `state` key, race conditions can occur. Always use distinct `output_key`s or manage concurrent writes explicitly.

### 3.3 `LoopAgent`: Iterative Processes

Repeatedly executes its `sub_agents` (sequentially within each loop iteration) until a condition is met or `max_iterations` is reached.

#### **Termination of `LoopAgent`**
A `LoopAgent` terminates when:
1.  `max_iterations` is reached.
2.  Any `Event` yielded by a sub-agent (or a tool within it) sets `actions.escalate = True`. This provides dynamic, content-driven loop termination.

#### **Example: Iterative Refinement Loop with a Custom `BaseAgent` for Control**
This example shows a loop that continues until a condition, determined by an evaluation agent, is met.

```python
from google.adk.agents import LoopAgent, Agent, BaseAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from typing import AsyncGenerator

# An LLM Agent that evaluates research and produces structured JSON output
research_evaluator = Agent(
    name="research_evaluator",
    # ... configuration from Section 2.2 ...
    output_schema=Feedback,
    output_key="research_evaluation",
)

# An LLM Agent that performs additional searches based on feedback
enhanced_search_executor = Agent(
    name="enhanced_search_executor",
    instruction="Execute the follow-up queries from 'research_evaluation' and combine with existing findings.",
    # ... other configurations ...
)

# A custom BaseAgent to check the evaluation and stop the loop
class EscalationChecker(BaseAgent):
    """Checks research evaluation and escalates to stop the loop if grade is 'pass'."""
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        evaluation = ctx.session.state.get("research_evaluation")
        if evaluation and evaluation.get("grade") == "pass":
            # The key to stopping the loop: yield an Event with escalate=True
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            # Let the loop continue
            yield Event(author=self.name)

# Define the loop
iterative_refinement_loop = LoopAgent(
    name="IterativeRefinementLoop",
    sub_agents=[
        research_evaluator, # Step 1: Evaluate
        EscalationChecker(name="EscalationChecker"), # Step 2: Check and maybe stop
        enhanced_search_executor, # Step 3: Refine (only runs if loop didn't stop)
    ],
    max_iterations=5, # Fallback to prevent infinite loops
    description="Iteratively evaluates and refines research until it passes quality checks."
)
```

---

## 4. Multi-Agent Systems & Communication

Building complex applications by composing multiple, specialized agents.

### 4.1 Agent Hierarchy

A hierarchical (tree-like) structure of parent-child relationships defined by the `sub_agents` parameter during `BaseAgent` initialization. An agent can only have one parent.

```python
# Conceptual Hierarchy
# Root
# └── Coordinator (LlmAgent)
#     ├── SalesAgent (LlmAgent)
#     └── SupportAgent (LlmAgent)
#     └── DataPipeline (SequentialAgent)
#         ├── DataFetcher (LlmAgent)
#         └── DataProcessor (LlmAgent)
```

### 4.2 Inter-Agent Communication Mechanisms

1.  **Shared Session State (`session.state`)**: The most common and robust method. Agents read from and write to the same mutable dictionary.
    *   **Mechanism**: Agent A sets `ctx.session.state['key'] = value`. Agent B later reads `ctx.session.state.get('key')`. `output_key` on `LlmAgent` is a convenient auto-setter.
    *   **Best for**: Passing intermediate results, shared configurations, and flags in pipelines (Sequential, Loop agents).

2.  **LLM-Driven Delegation (`transfer_to_agent`)**: A `LlmAgent` can dynamically hand over control to another agent based on its reasoning.
    *   **Mechanism**: The LLM generates a special `transfer_to_agent` function call. The ADK framework intercepts this, routes the next turn to the target agent.
    *   **Prerequisites**:
        *   The initiating `LlmAgent` needs `instruction` to guide delegation and `description` of the target agent(s).
        *   Target agents need clear `description`s to help the LLM decide.
        *   Target agent must be discoverable within the current agent's hierarchy (direct `sub_agent` or a descendant).
    *   **Configuration**: Can be enabled/disabled via `disallow_transfer_to_parent` and `disallow_transfer_to_peers` on `LlmAgent`.

3.  **Explicit Invocation (`AgentTool`)**: An `LlmAgent` can treat another `BaseAgent` instance as a callable tool.
    *   **Mechanism**: Wrap the target agent (`target_agent`) in `AgentTool(agent=target_agent)` and add it to the calling `LlmAgent`'s `tools` list. The `AgentTool` generates a `FunctionDeclaration` for the LLM. When called, `AgentTool` runs the target agent and returns its final response as the tool result.
    *   **Best for**: Hierarchical task decomposition, where a higher-level agent needs a specific output from a lower-level agent.

### 4.3 Common Multi-Agent Patterns

*   **Coordinator/Dispatcher**: A central agent routes requests to specialized sub-agents (often via LLM-driven delegation).
*   **Sequential Pipeline**: `SequentialAgent` orchestrates a fixed sequence of tasks, passing data via shared state.
*   **Parallel Fan-Out/Gather**: `ParallelAgent` runs concurrent tasks, followed by a final agent that synthesizes results from state.
*   **Review/Critique (Generator-Critic)**: `SequentialAgent` with a generator followed by a critic, often in a `LoopAgent` for iterative refinement.
*   **Hierarchical Task Decomposition (Planner/Executor)**: High-level agents break down complex problems, delegating sub-tasks to lower-level agents (often via `AgentTool` and delegation).

#### **Example: Hierarchical Planner/Executor Pattern**
This pattern combines several mechanisms. A top-level `interactive_planner_agent` uses another agent (`plan_generator`) as a tool to create a plan, then delegates the execution of that plan to a complex `SequentialAgent` (`research_pipeline`).

```python
from google.adk.agents import LlmAgent, SequentialAgent, LoopAgent
from google.adk.tools.agent_tool import AgentTool

# Assume plan_generator, section_planner, research_evaluator, etc. are defined.

# The execution pipeline itself is a complex agent.
research_pipeline = SequentialAgent(
    name="research_pipeline",
    description="Executes a pre-approved research plan. It performs iterative research, evaluation, and composes a final, cited report.",
    sub_agents=[
        section_planner,
        section_researcher,
        LoopAgent(
            name="iterative_refinement_loop",
            max_iterations=3,
            sub_agents=[
                research_evaluator,
                EscalationChecker(name="escalation_checker"),
                enhanced_search_executor,
            ],
        ),
        report_composer,
    ],
)

# The top-level agent that interacts with the user.
interactive_planner_agent = LlmAgent(
    name="interactive_planner_agent",
    model="gemini-2.5-flash",
    description="The primary research assistant. It collaborates with the user to create a research plan, and then executes it upon approval.",
    instruction="""
    You are a research planning assistant. Your workflow is:
    1.  **Plan:** Use the `plan_generator` tool to create a draft research plan.
    2.  **Refine:** Incorporate user feedback until the plan is approved.
    3.  **Execute:** Once the user gives EXPLICIT approval (e.g., "looks good, run it"), you MUST delegate the task to the `research_pipeline` agent.
    Your job is to Plan, Refine, and Delegate. Do not do the research yourself.
    """,
    # The planner delegates to the pipeline.
    sub_agents=[research_pipeline],
    # The planner uses another agent as a tool.
    tools=[AgentTool(plan_generator)],
    output_key="research_plan",
)

# The root agent of the application is the top-level planner.
root_agent = interactive_planner_agent
```

---

## 5. Building Custom Agents (`BaseAgent`)

For unique orchestration logic that doesn't fit standard workflow agents, inherit directly from `BaseAgent`.

### 5.1 When to Use Custom Agents

*   **Complex Conditional Logic**: `if/else` branching based on multiple state variables.
*   **Dynamic Agent Selection**: Choosing which sub-agent to run based on runtime evaluation.
*   **Direct External Integrations**: Calling external APIs or libraries directly within the orchestration flow.
*   **Custom Loop/Retry Logic**: More sophisticated iteration patterns than `LoopAgent`, such as the `EscalationChecker` example.

### 5.2 Implementing `_run_async_impl`

This is the core asynchronous method you must override.

#### **Example: A Custom Agent for Loop Control**
This agent reads state, applies simple Python logic, and yields an `Event` with an `escalate` action to control a `LoopAgent`.

```python
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from typing import AsyncGenerator
import logging

class EscalationChecker(BaseAgent):
    """Checks research evaluation and escalates to stop the loop if grade is 'pass'."""

    def __init__(self, name: str):
        super().__init__(name=name)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # 1. Read from session state.
        evaluation_result = ctx.session.state.get("research_evaluation")

        # 2. Apply custom Python logic.
        if evaluation_result and evaluation_result.get("grade") == "pass":
            logging.info(
                f"[{self.name}] Research passed. Escalating to stop loop."
            )
            # 3. Yield an Event with a control Action.
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            logging.info(
                f"[{self.name}] Research failed or not found. Loop continues."
            )
            # Yielding an event without actions lets the flow continue.
            yield Event(author=self.name)
```
*   **Asynchronous Generator**: `async def ... yield Event`. This allows pausing and resuming execution.
*   **`ctx: InvocationContext`**: Provides access to all session state (`ctx.session.state`).
*   **Calling Sub-Agents**: Use `async for event in self.sub_agent_instance.run_async(ctx): yield event`.
*   **Control Flow**: Use standard Python `if/else`, `for/while` loops for complex logic.

---

## 6. Models: Gemini, LiteLLM, and Vertex AI

ADK's model flexibility allows integrating various LLMs for different needs.

### 6.1 Google Gemini Models (AI Studio & Vertex AI)

*   **Default Integration**: Native support via `google-genai` library.
*   **AI Studio (Easy Start)**:
    *   Set `GOOGLE_API_KEY="YOUR_API_KEY"` (environment variable).
    *   Set `GOOGLE_GENAI_USE_VERTEXAI="False"`.
    *   Model strings: `"gemini-2.5-flash"`, `"gemini-2.5-pro"`, etc.
*   **Vertex AI (Production)**:
    *   Authenticate via `gcloud auth application-default login` (recommended).
    *   Set `GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID"`, `GOOGLE_CLOUD_LOCATION="your-region"` (environment variables).
    *   Set `GOOGLE_GENAI_USE_VERTEXAI="True"`.
    *   Model strings: `"gemini-2.5-flash"`, `"gemini-2.5-pro"`, or full Vertex AI endpoint resource names for specific deployments.

### 6.2 Other Cloud & Proprietary Models via LiteLLM

`LiteLlm` provides a unified interface to 100+ LLMs (OpenAI, Anthropic, Cohere, etc.).

*   **Installation**: `pip install litellm`
*   **API Keys**: Set environment variables as required by LiteLLM (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
*   **Usage**:
    ```python
    from google.adk.models.lite_llm import LiteLlm
    agent_openai = Agent(model=LiteLlm(model="openai/gpt-4o"), ...)
    agent_claude = Agent(model=LiteLlm(model="anthropic/claude-3-haiku-20240307"), ...)
    ```

### 6.3 Open & Local Models via LiteLLM (Ollama, vLLM)

For self-hosting, cost savings, privacy, or offline use.

*   **Ollama Integration**: Run Ollama locally (`ollama run <model>`).
    ```bash
    export OLLAMA_API_BASE="http://localhost:11434" # Ensure Ollama server is running
    ```
    ```python
    from google.adk.models.lite_llm import LiteLlm
    # Use 'ollama_chat' provider for tool-calling capabilities with Ollama models
    agent_ollama = Agent(model=LiteLlm(model="ollama_chat/llama3:instruct"), ...)
    ```

*   **Self-Hosted Endpoint (e.g., vLLM)**:
    ```python
    from google.adk.models.lite_llm import LiteLlm
    api_base_url = "https://your-vllm-endpoint.example.com/v1"
    agent_vllm = Agent(
        model=LiteLlm(
            model="your-model-name-on-vllm",
            api_base=api_base_url,
            extra_headers={"Authorization": "Bearer YOUR_TOKEN"},
        ),
        ...
    )
    ```

### 6.4 Customizing LLM API Clients

For `google-genai` (used by Gemini models), you can configure the underlying client.

```python
import os
from google.genai import configure as genai_configure

genai_configure.use_defaults(
    timeout=60, # seconds
    client_options={"api_key": os.getenv("GOOGLE_API_KEY")},
)
```

---

## 7. Tools: The Agent's Capabilities

Tools extend an agent's abilities beyond text generation.

### 7.1 Defining Function Tools: Principles & Best Practices

*   **Signature**: `def my_tool(param1: Type, param2: Type, tool_context: ToolContext) -> dict:`
*   **Function Name**: Descriptive verb-noun (e.g., `schedule_meeting`).
*   **Parameters**: Clear names, required type hints, **NO DEFAULT VALUES**.
*   **Return Type**: **Must** be a `dict` (JSON-serializable), preferably with a `'status'` key.
*   **Docstring**: **CRITICAL**. Explain purpose, when to use, arguments, and return value structure. **AVOID** mentioning `tool_context`.

    ```python
    def calculate_compound_interest(
        principal: float,
        rate: float,
        years: int,
        compounding_frequency: int,
        tool_context: ToolContext
    ) -> dict:
        """Calculates the future value of an investment with compound interest.

        Use this tool to calculate the future value of an investment given a
        principal amount, interest rate, number of years, and how often the
        interest is compounded per year.

        Args:
            principal (float): The initial amount of money invested.
            rate (float): The annual interest rate (e.g., 0.05 for 5%).
            years (int): The number of years the money is invested.
            compounding_frequency (int): The number of times interest is compounded
                                         per year (e.g., 1 for annually, 12 for monthly).
            
        Returns:
            dict: Contains the calculation result.
                  - 'status' (str): "success" or "error".
                  - 'future_value' (float, optional): The calculated future value.
                  - 'error_message' (str, optional): Description of error, if any.
        """
        # ... implementation ...
    ```

### 7.2 The `ToolContext` Object: Accessing Runtime Information

`ToolContext` is the gateway for tools to interact with the ADK runtime.

*   `tool_context.state`: Read and write to the current `Session`'s `state` dictionary.
*   `tool_context.actions`: Modify the `EventActions` object (e.g., `tool_context.actions.escalate = True`).
*   `tool_context.load_artifact(filename)` / `tool_context.save_artifact(filename, part)`: Manage binary data.
*   `tool_context.search_memory(query)`: Query the long-term `MemoryService`.

### 7.3 All Tool Types & Their Usage

ADK supports a diverse ecosystem of tools.

1.  **`FunctionTool`**: Wraps any Python callable. The most common tool type.
2.  **`LongRunningFunctionTool`**: For `async` functions that `yield` intermediate results.
3.  **`AgentTool`**: Wraps another `BaseAgent` instance, allowing it to be called as a tool.
4.  **`OpenAPIToolset`**: Automatically generates tools from an OpenAPI (Swagger) specification.
5.  **`MCPToolset`**: Connects to an external Model Context Protocol (MCP) server.
6.  **Built-in Tools**: `google_search`, `BuiltInCodeExecutor`, `VertexAiSearchTool`. e.g `from google.adk.tools import google_search` 
Note: google_search is a special tool automatically invoked by the model. It can be passed directly to the agent without wrapping in a custom function.
7.  **Third-Party Tool Wrappers**: `LangchainTool`, `CrewaiTool`.
8.  **Google Cloud Tools**: `ApiHubToolset`, `ApplicationIntegrationToolset`.

---

## 8. Context, State, and Memory Management

Effective context management is crucial for coherent, multi-turn conversations.

### 8.1 The `Session` Object & `SessionService`

*   **`Session`**: The container for a single, ongoing conversation (`id`, `state`, `events`).
*   **`SessionService`**: Manages the lifecycle of `Session` objects (`create_session`, `get_session`, `append_event`).
*   **Implementations**: `InMemorySessionService` (dev), `VertexAiSessionService` (prod), `DatabaseSessionService` (self-managed).

### 8.2 `State`: The Conversational Scratchpad

A mutable dictionary within `session.state` for short-term, dynamic data.

*   **Update Mechanism**: Always update via `context.state` (in callbacks/tools) or `LlmAgent.output_key`.
*   **Prefixes for Scope**:
    *   **(No prefix)**: Session-specific (e.g., `session.state['booking_step']`).
    *   `user:`: Persistent for a `user_id` across all their sessions (e.g., `session.state['user:preferred_currency']`).
    *   `app:`: Persistent for `app_name` across all users and sessions.
    *   `temp:`: Volatile, for the current `Invocation` turn only.

### 8.3 `Memory`: Long-Term Knowledge & Retrieval

For knowledge beyond a single conversation.

*   **`BaseMemoryService`**: Defines the interface (`add_session_to_memory`, `search_memory`).
*   **Implementations**: `InMemoryMemoryService`, `VertexAiRagMemoryService`.
*   **Usage**: Agents interact via tools (e.g., the built-in `load_memory` tool).

### 8.4 `Artifacts`: Binary Data Management

For named, versioned binary data (files, images).

*   **Representation**: `google.genai.types.Part` (containing a `Blob` with `data: bytes` and `mime_type: str`).
*   **`BaseArtifactService`**: Manages storage (`save_artifact`, `load_artifact`).
*   **Implementations**: `InMemoryArtifactService`, `GcsArtifactService`.

---

## 9. Runtime, Events, and Execution Flow

The `Runner` is the central orchestrator of an ADK application.

### 9.1 The `Runner`: The Orchestrator

*   **Role**: Manages the agent's lifecycle, the event loop, and coordinates with services.
*   **Entry Point**: `runner.run_async(user_id, session_id, new_message)`.

### 9.2 The Event Loop: Core Execution Flow

1.  User input becomes a `user` `Event`.
2.  `Runner` calls `agent.run_async(invocation_context)`.
3.  Agent `yield`s an `Event` (e.g., tool call, text response). Execution pauses.
4.  `Runner` processes the `Event` (applies state changes, etc.) and yields it to the client.
5.  Execution resumes. This cycle repeats until the agent is done.

### 9.3 `Event` Object: The Communication Backbone

`Event` objects carry all information and signals.

*   `Event.author`: Source of the event (`'user'`, agent name, `'system'`).
*   `Event.content`: The primary payload (text, function calls, function responses).
*   `Event.actions`: Signals side effects (`state_delta`, `transfer_to_agent`, `escalate`).
*   `Event.is_final_response()`: Helper to identify the complete, displayable message.

### 9.4 Asynchronous Programming (Python Specific)

ADK is built on `asyncio`. Use `async def`, `await`, and `async for` for all I/O-bound operations.

---

## 10. Control Flow with Callbacks

Callbacks are functions that intercept and control agent execution at specific points.

### 10.1 Callback Mechanism: Interception & Control

*   **Definition**: A Python function assigned to an agent's `callback` parameter (e.g., `after_agent_callback=my_func`).
*   **Context**: Receives a `CallbackContext` (or `ToolContext`) with runtime info.
*   **Return Value**: **Crucially determines flow.**
    *   `return None`: Allow the default action to proceed.
    *   `return <Specific Object>`: **Override** the default action/result.

### 10.2 Types of Callbacks

1.  **Agent Lifecycle**: `before_agent_callback`, `after_agent_callback`.
2.  **LLM Interaction**: `before_model_callback`, `after_model_callback`.
3.  **Tool Execution**: `before_tool_callback`, `after_tool_callback`.

### 10.3 Callback Best Practices

*   **Keep Focused**: Each callback for a single purpose.
*   **Performance**: Avoid blocking I/O or heavy computation.
*   **Error Handling**: Use `try...except` to prevent crashes.

#### **Example 1: Data Aggregation with `after_agent_callback`**
This callback runs after an agent, inspects the `session.events` to find structured data from tool calls (like `google_search` results), and saves it to state for later use.

```python
from google.adk.agents.callback_context import CallbackContext

def collect_research_sources_callback(callback_context: CallbackContext) -> None:
    """Collects and organizes web research sources from agent events."""
    session = callback_context._invocation_context.session
    # Get existing sources from state to append to them.
    url_to_short_id = callback_context.state.get("url_to_short_id", {})
    sources = callback_context.state.get("sources", {})
    id_counter = len(url_to_short_id) + 1

    # Iterate through all events in the session to find grounding metadata.
    for event in session.events:
        if not (event.grounding_metadata and event.grounding_metadata.grounding_chunks):
            continue
        # ... logic to parse grounding_chunks and grounding_supports ...
        # (See full implementation in the original code snippet)

    # Save the updated source map back to state.
    callback_context.state["url_to_short_id"] = url_to_short_id
    callback_context.state["sources"] = sources

# Used in an agent like this:
# section_researcher = LlmAgent(..., after_agent_callback=collect_research_sources_callback)
```

#### **Example 2: Output Transformation with `after_agent_callback`**
This callback takes an LLM's raw output (containing custom tags), uses Python to format it into markdown, and returns the modified content, overriding the original.

```python
import re
from google.adk.agents.callback_context import CallbackContext
from google.genai import types as genai_types

def citation_replacement_callback(callback_context: CallbackContext) -> genai_types.Content:
    """Replaces <cite> tags in a report with Markdown-formatted links."""
    # 1. Get raw report and sources from state.
    final_report = callback_context.state.get("final_cited_report", "")
    sources = callback_context.state.get("sources", {})

    # 2. Define a replacer function for regex substitution.
    def tag_replacer(match: re.Match) -> str:
        short_id = match.group(1)
        if not (source_info := sources.get(short_id)):
            return "" # Remove invalid tags
        title = source_info.get("title", short_id)
        return f" [{title}]({source_info['url']})"

    # 3. Use regex to find all <cite> tags and replace them.
    processed_report = re.sub(
        r'<cite\s+source\s*=\s*["\']?(src-\d+)["\']?\s*/>',
        tag_replacer,
        final_report,
    )
    processed_report = re.sub(r"\s+([.,;:])", r"\1", processed_report) # Fix spacing

    # 4. Save the new version to state and return it to override the original agent output.
    callback_context.state["final_report_with_citations"] = processed_report
    return genai_types.Content(parts=[genai_types.Part(text=processed_report)])

# Used in an agent like this:
# report_composer = LlmAgent(..., after_agent_callback=citation_replacement_callback)
```
---

## 11. Authentication for Tools

Enabling agents to securely access protected external resources.

### 11.1 Core Concepts: `AuthScheme` & `AuthCredential`

*   **`AuthScheme`**: Defines *how* an API expects authentication (e.g., `APIKey`, `HTTPBearer`, `OAuth2`, `OpenIdConnectWithConfig`).
*   **`AuthCredential`**: Holds *initial* information to *start* the auth process (e.g., API key value, OAuth client ID/secret).

### 11.2 Interactive OAuth/OIDC Flows

When a tool requires user interaction (OAuth consent), ADK pauses and signals your `Agent Client` application.

1.  **Detect Auth Request**: `runner.run_async()` yields an event with a special `adk_request_credential` function call.
2.  **Redirect User**: Extract `auth_uri` from `auth_config` in the event. Your client app redirects the user's browser to this `auth_uri` (appending `redirect_uri`).
3.  **Handle Callback**: Your client app has a pre-registered `redirect_uri` to receive the user after authorization. It captures the full callback URL (containing `authorization_code`).
4.  **Send Auth Result to ADK**: Your client prepares a `FunctionResponse` for `adk_request_credential`, setting `auth_config.exchanged_auth_credential.oauth2.auth_response_uri` to the captured callback URL.
5.  **Resume Execution**: `runner.run_async()` is called again with this `FunctionResponse`. ADK performs the token exchange, stores the access token, and retries the original tool call.

### 11.3 Custom Tool Authentication

If building a `FunctionTool` that needs authentication:

1.  **Check for Cached Creds**: `tool_context.state.get("my_token_cache_key")`.
2.  **Check for Auth Response**: `tool_context.get_auth_response(my_auth_config)`.
3.  **Initiate Auth**: If no creds, call `tool_context.request_credential(my_auth_config)` and return a pending status. This triggers the external flow.
4.  **Cache Credentials**: After obtaining, store in `tool_context.state`.
5.  **Make API Call**: Use the valid credentials (e.g., `google.oauth2.credentials.Credentials`).

---

## 12. Deployment Strategies

From local dev to production.

### 12.1 Local Development & Testing (`adk web`, `adk run`, `adk api_server`)

*   **`adk web`**: Launches a local web UI for interactive chat, session inspection, and visual tracing.
    ```bash
    adk web /path/to/your/project_root
    ```
*   **`adk run`**: Command-line interactive chat.
    ```bash
    adk run /path/to/your/agent_folder
    ```
*   **`adk api_server`**: Launches a local FastAPI server exposing `/run`, `/run_sse`, `/list-apps`, etc., for API testing with `curl` or client libraries.
    ```bash
    adk api_server /path/to/your/project_root
    ```

### 12.2 Vertex AI Agent Engine

Fully managed, scalable service for ADK agents on Google Cloud.

*   **Features**: Auto-scaling, session management, observability integration.
*   **Deployment**: Use `vertexai.agent_engines.create()`.
    ```python
    from vertexai.preview import reasoning_engines # or agent_engines directly in later versions
    
    # Wrap your root_agent for deployment
    app_for_engine = reasoning_engines.AdkApp(agent=root_agent, enable_tracing=True)
    
    # Deploy
    remote_app = agent_engines.create(
        agent_engine=app_for_engine,
        requirements=["google-cloud-aiplatform[adk,agent_engines]"],
        display_name="My Production Agent"
    )
    print(remote_app.resource_name) # projects/PROJECT_NUM/locations/REGION/reasoningEngines/ID
    ```
*   **Interaction**: Use `remote_app.stream_query()`, `create_session()`, etc.

### 12.3 Cloud Run

Serverless container platform for custom web applications.

*   **Deployment**:
    1.  Create a `Dockerfile` for your FastAPI app (using `google.adk.cli.fast_api.get_fast_api_app`).
    2.  Use `gcloud run deploy --source .`.
    3.  Alternatively, `adk deploy cloud_run` (simpler, opinionated).
*   **Example `main.py`**:
    ```python
    import os
    from fastapi import FastAPI
    from google.adk.cli.fast_api import get_fast_api_app

    # Ensure your agent_folder (e.g., 'my_first_agent') is in the same directory as main.py
    app: FastAPI = get_fast_api_app(
        agents_dir=os.path.dirname(os.path.abspath(__file__)),
        session_service_uri="sqlite:///./sessions.db", # In-container SQLite, for simple cases
        # For production: use a persistent DB (Cloud SQL) or VertexAiSessionService
        allow_origins=["*"],
        web=True # Serve ADK UI
    )
    # uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080))) # If running directly
    ```

### 12.4 Google Kubernetes Engine (GKE)

For maximum control, run your containerized agent in a Kubernetes cluster.

*   **Deployment**:
    1.  Build Docker image (`gcloud builds submit`).
    2.  Create Kubernetes Deployment and Service YAMLs.
    3.  Apply with `kubectl apply -f deployment.yaml`.
    4.  Configure Workload Identity for GCP permissions.

### 12.5 CI/CD Integration

*   Automate testing (`pytest`, `adk eval`) in CI.
*   Automate container builds and deployments (e.g., Cloud Build, GitHub Actions).
*   Use environment variables for secrets.

---

## 13. Evaluation and Safety

Critical for robust, production-ready agents.

### 13.1 Agent Evaluation (`adk eval`)

Systematically assess agent performance using predefined test cases.

*   **Evalset File (`.evalset.json`)**: Contains `eval_cases`, each with a `conversation` (user queries, expected tool calls, expected intermediate/final responses) and `session_input` (initial state).
    ```json
    {
      "eval_set_id": "weather_bot_eval",
      "eval_cases": [
        {
          "eval_id": "london_weather_query",
          "conversation": [
            {
              "user_content": {"parts": [{"text": "What's the weather in London?"}]},
              "final_response": {"parts": [{"text": "The weather in London is cloudy..."}]},
              "intermediate_data": {
                "tool_uses": [{"name": "get_weather", "args": {"city": "London"}}]
              }
            }
          ],
          "session_input": {"app_name": "weather_app", "user_id": "test_user", "state": {}}
        }
      ]
    }
    ```
*   **Running Evaluation**:
    *   `adk web`: Interactive UI for creating/running eval cases.
    *   `adk eval /path/to/agent_folder /path/to/evalset.json`: CLI execution.
    *   `pytest`: Integrate `AgentEvaluator.evaluate()` into unit/integration tests.
*   **Metrics**: `tool_trajectory_avg_score` (tool calls match expected), `response_match_score` (final response similarity using ROUGE). Configurable via `test_config.json`.

### 13.2 Safety & Guardrails

Multi-layered defense against harmful content, misalignment, and unsafe actions.

1.  **Identity and Authorization**:
    *   **Agent-Auth**: Tool acts with the agent's service account (e.g., `Vertex AI User` role). Simple, but all users share access level. Logs needed for attribution.
    *   **User-Auth**: Tool acts with the end-user's identity (via OAuth tokens). Reduces risk of abuse.
2.  **In-Tool Guardrails**: Design tools defensively. Tools can read policies from `tool_context.state` (set deterministically by developer) and validate model-provided arguments before execution.
    ```python
    def execute_sql(query: str, tool_context: ToolContext) -> dict:
        policy = tool_context.state.get("user:sql_policy", {})
        if not policy.get("allow_writes", False) and ("INSERT" in query.upper() or "DELETE" in query.upper()):
            return {"status": "error", "message": "Policy: Write operations are not allowed."}
        # ... execute query ...
    ```
3.  **Built-in Gemini Safety Features**:
    *   **Content Safety Filters**: Automatically block harmful content (CSAM, PII, hate speech, etc.). Configurable thresholds.
    *   **System Instructions**: Guide model behavior, define prohibited topics, brand tone, disclaimers.
4.  **Model and Tool Callbacks (LLM as a Guardrail)**: Use callbacks to inspect inputs/outputs.
    *   `before_model_callback`: Intercept `LlmRequest` before it hits the LLM. Block (return `LlmResponse`) or modify.
    *   `before_tool_callback`: Intercept tool calls (name, args) before execution. Block (return `dict`) or modify.
    *   **LLM-based Safety**: Use a cheap/fast LLM (e.g., Gemini Flash) in a callback to classify input/output safety.
        ```python
        def safety_checker_callback(context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
            # Use a separate, small LLM to classify safety
            safety_llm_agent = Agent(name="SafetyChecker", model="gemini-2.5-flash-001", instruction="Classify input as 'safe' or 'unsafe'. Output ONLY the word.")
            # Run the safety agent (might need a new runner instance or direct model call)
            # For simplicity, a mock:
            user_input = llm_request.contents[-1].parts[0].text
            if "dangerous_phrase" in user_input.lower():
                context.state["safety_violation"] = True
                return LlmResponse(content=genai_types.Content(parts=[genai_types.Part(text="I cannot process this request due to safety concerns.")]))
            return None
        ```
5.  **Sandboxed Code Execution**:
    *   `BuiltInCodeExecutor`: Uses secure, sandboxed execution environments.
    *   Vertex AI Code Interpreter Extension.
    *   If custom, ensure hermetic environments (no network, isolated).
6.  **Network Controls & VPC-SC**: Confine agent activity within secure perimeters (VPC Service Controls) to prevent data exfiltration.
7.  **Output Escaping in UIs**: Always properly escape LLM-generated content in web UIs to prevent XSS attacks and indirect prompt injections.

---

## 14. Debugging, Logging & Observability

*   **`adk web` UI**: Best first step. Provides visual trace, session history, and state inspection.
*   **Event Stream Logging**: Iterate `runner.run_async()` events and print relevant fields.
    ```python
    async for event in runner.run_async(...):
        print(f"[{event.author}] Event ID: {event.id}, Invocation: {event.invocation_id}")
        if event.content and event.content.parts:
            if event.content.parts[0].text:
                print(f"  Text: {event.content.parts[0].text[:100]}...")
            if event.get_function_calls():
                print(f"  Tool Call: {event.get_function_calls()[0].name} with {event.get_function_calls()[0].args}")
            if event.get_function_responses():
                print(f"  Tool Response: {event.get_function_responses()[0].response}")
        if event.actions:
            if event.actions.state_delta:
                print(f"  State Delta: {event.actions.state_delta}")
            if event.actions.transfer_to_agent:
                print(f"  TRANSFER TO: {event.actions.transfer_to_agent}")
        if event.error_message:
            print(f"  ERROR: {event.error_message}")
    ```
*   **Tool/Callback `print` statements**: Simple logging directly within your functions.
*   **Python `logging` module**: Integrate with standard logging frameworks.
*   **Tracing Integrations**: ADK supports OpenTelemetry (e.g., via Comet Opik) for distributed tracing.
    ```python
    # Example using Comet Opik integration (conceptual)
    # pip install comet_opik_adk
    # from comet_opik_adk import enable_opik_tracing
    # enable_opik_tracing() # Call at app startup
    # Then run your ADK app, traces appear in Comet workspace.
    ```
*   **Session History (`session.events`)**: Persisted for detailed post-mortem analysis.

---

## 15. Advanced I/O Modalities

ADK (especially with Gemini Live API models) supports richer interactions.

*   **Audio**: Input via `Blob(mime_type="audio/pcm", data=bytes)`, Output via `genai_types.SpeechConfig` in `RunConfig`.
*   **Vision (Images/Video)**: Input via `Blob(mime_type="image/jpeg", data=bytes)` or `Blob(mime_type="video/mp4", data=bytes)`. Models like `gemini-2.5-flash-exp` can process these.
*   **Multimodal Input in `Content`**:
    ```python
    multimodal_content = genai_types.Content(
        parts=[
            genai_types.Part(text="Describe this image:"),
            genai_types.Part(inline_data=genai_types.Blob(mime_type="image/jpeg", data=image_bytes))
        ]
    )
    ```
*   **Streaming Modalities**: `RunConfig.response_modalities=['TEXT', 'AUDIO']`.

---

## 16. Performance Optimization

*   **Model Selection**: Choose the smallest model that meets requirements (e.g., `gemini-2.5-flash` for simple tasks).
*   **Instruction Prompt Engineering**: Concise, clear instructions reduce tokens and improve accuracy.
*   **Tool Use Optimization**:
    *   Design efficient tools (fast API calls, optimize database queries).
    *   Cache tool results (e.g., using `before_tool_callback` or `tool_context.state`).
*   **State Management**: Store only necessary data in state to avoid large context windows.
*   **`include_contents='none'`**: For stateless utility agents, saves LLM context window.
*   **Parallelization**: Use `ParallelAgent` for independent tasks.
*   **Streaming**: Use `StreamingMode.SSE` or `BIDI` for perceived latency reduction.
*   **`max_llm_calls`**: Limit LLM calls to prevent runaway agents and control costs.

---

## 17. General Best Practices & Common Pitfalls

*   **Start Simple**: Begin with `LlmAgent`, mock tools, and `InMemorySessionService`. Gradually add complexity.
*   **Iterative Development**: Build small features, test, debug, refine.
*   **Modular Design**: Use agents and tools to encapsulate logic.
*   **Clear Naming**: Descriptive names for agents, tools, state keys.
*   **Error Handling**: Implement robust `try...except` blocks in tools and callbacks. Guide LLMs on how to handle tool errors.
*   **Testing**: Write unit tests for tools/callbacks, integration tests for agent flows (`pytest`, `adk eval`).
*   **Dependency Management**: Use virtual environments (`venv`) and `requirements.txt`.
*   **Secrets Management**: Never hardcode API keys. Use `.env` for local dev, environment variables or secret managers (Google Cloud Secret Manager) for production.
*   **Avoid Infinite Loops**: Especially with `LoopAgent` or complex LLM tool-calling chains. Use `max_iterations`, `max_llm_calls`, and strong instructions.
*   **Handle `None` & `Optional`**: Always check for `None` or `Optional` values when accessing nested properties (e.g., `event.content and event.content.parts and event.content.parts[0].text`).
*   **Immutability of Events**: Events are immutable records. If you need to change something *before* it's processed, do so in a `before_*` callback and return a *new* modified object.
*   **Understand `output_key` vs. direct `state` writes**: `output_key` is for the agent's *final conversational* output. Direct `tool_context.state['key'] = value` is for *any other* data you want to save.
*   **Example Agents**: Find practical examples and reference implementations in the [ADK Samples repository](https://github.com/google/adk-samples).


### Testing the output of an agent

The following script demonstrates how to programmatically test an agent's output. This approach is extremely useful when an LLM or coding agent needs to interact with a work-in-progress agent, as well as for automated testing, debugging, or when you need to integrate agent execution into other workflows:
```
import asyncio

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from app.agent import root_agent
from google.genai import types as genai_types


async def main():
    """Runs the agent with a sample query."""
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="app", user_id="test_user", session_id="test_session"
    )
    runner = Runner(
        agent=root_agent, app_name="app", session_service=session_service
    )
    query = "I want a recipe for pancakes"
    async for event in runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=genai_types.Content(
            role="user", 
            parts=[genai_types.Part.from_text(text=query)]
        ),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)


if __name__ == "__main__":
    asyncio.run(main())
```





Internal FAQ for Google Agentspace 
go/agentspace-faq NotebookLM to chat with all the AS docs

SECTIONS HIGHLIGHTED IN GREEN ARE NEW ADDITIONS TO THE FAQ.
For new feature launches, please refer to go/agentspace-featurelaunch

Product Status: Allowlist GA
Questions? WE ARE MOVING TO A NEW PROCESS
Option 1: Please use YAQS > eng and use the Agentspace tag. 
Option 2: Join Agentspace Office hours on
Fridays at 8AM PST. Please use this LINK.
Wednesday at 10:00 PM PST. Please use this LINK 


Last Updated: Jul 7, 2025 
How to use this document 
This document is intended for customer facing teams to learn about Agentspace. Except for any content expressly labeled as externally-friendly, this document is for internal use only. This  document cannot be shared directly with customers or partners. 

For additional questions, please refer to the comment at the end of this document and we will add it to the document with a response. 

Reference Docs

For all assets, please visit go/agentspace-gtm
Connector status: go/es-connectors 
Agentspace Roadmap [June Update] go/agentspace-roadmapdeck, go/seller-agentspace-roadmap 
go/agentspace-value-pitch 

* ‘Please check with PM/ Tech onboarding lead before leaving behind the deck with customer

Cloud NEXT ‘25 Updates: 

IMPORTANT: Agentspace now included in the Google Cloud SST/ToS 


Agentspace: The entry point for employees to use agents and organizations to govern employee agents
Agentspace is the fastest growing Enterprise product in Google Cloud 
Agentspace is a single integrated platform that lets you easily apply search, AI and agents across your enterprise to fundamentally change how work gets done.
Not every agent needs to be built on Google or by Google: Choose from Google provided, customer built, and partner built agents
Partners like Revionics, Accenture, Palo Alto Networks, Deloitte, Incorta, and Fullstory are building agents for Agentspace
Agentspace offers a single platform for all your employees to access Google’s latest AI innovations such as Gemini, Imagen, Veo (will be available in Q2), NotebookLM, and expert agents in the simplicity of a SaaS solution that still maintains the enterprise readiness of our Cloud
Feature Launches for NEXT ‘25: 

The following features were announced at Next ‘25: 

Agent Designer (basic no code) [Public Preview]
Enabled for all customers, does not require an allowlist 
Agent Gallery (GA Allowlist) 
Enabled for all customers
Support for ADK Agents to be registered in Agentspace 
Please raise bug and assign it to Krzysztof Wiśniewski 
Support for partner agents from Marketplace [Private Preview]
Support for Agent Developer Kit (ADK) Agents to be registered in Agentspace  [Private Preview]
How to publish agents to Agentspace [ Note: This is a temp process. Formal process to be launched by Eng later] 
Please raise bug and assign it to Krzysztof Wiśniewski 
Idea Generation Agent [ Private Preview]
Additional allow listing is needed, http://goto.google.com/if-allowlist
Agentspace and Chrome Enterprise Integration  [ Private Preview]
Request Form 

With respect to building agents, at Google Next ‘25, we announce the capability to help business users and developers build agents with a suite of products. 

Business Users can create agents within Agentspace using a no code builder called Agent Designer  for specific tasks, improve productivity, and streamline workflows with a chat-based interface followed by a Low-Code Builder (Q2 '25) with a visual editor for enhanced customization.  
Developers can define sophisticated, multi-agent applications with Agent Development Kit (ADK) (Private Preview) and deploy them to Agent Engine (GA) to use in production with robust monitoring and observability. These agents can be integrated with Agentspace for all users to leverage.  

Agent Gallery

What is Agent Gallery? 
Agent Gallery is a product surface in Agentspace where employees can find the right agent for the job needed through discover from:
Google expert agents
Customer built agents (Full code or no code)
Partner agents from Marketplace 

How do customers get access to it? 
Agent Gallery is a feature that comes with Agentspace by default.  

Agent engine roadmap

No code agent builder/Agent Designer 

What is Agent Designer / No Code Agent Builder? 

Agent Designer is a no-code feature that allows end users to quickly define agents that will help them automate repeated tasks and workflows, with access to the built in actions that the Agentspace main assistant has (e.g. create calendar events, search Jira etc). 

Users can describe in natural language the problem they want to solve, or the workflow they want to automate, and Agentspace will assist them in building a personalized companion. These agents have access to all the data sources and actions that are available to their Agentspace instance. 

Example prompts that users can do: 
As a product manager, I want to build an agent that helps me write PRDs
As a program manager, I want to build an agent that automates the process of creating Jira tickets based on meeting notes.


How do customers get access to it?
Any customer who is part of the allowlisted GA for Agentspace will have the no-code builder present in their instance of Agentspace.

When will knowledge source selection be launched?
See roadmap for the latest.

What tools will be available for Agent Designer?
The no code agents can access all tools the main assistant has - in most cases, this means email creation, calendar event creation, searching Jira, Workday event creation, video and image generation. For details see go/agentspace-featurelaunch. 

Can admins control who can create agents?
This functionality is planned for early Q2.

How will access control for files be handled?
This will be in line with ACLs set up when data sources are connected to Agentspace - so access for no code agents will reflect what access the end user has.

Are all agents going to be chat based?
Initially, all agents are interactive, however we will launch background agents (i.e. those that run based on a time or event trigger) and notify the end user when they have a result or need more input.

Support for 3P agent builder platforms

How do customers get access to it? 
Customers can purchase agents from the Google Marketplace and integrate them with their Agentspace instance. 

Partners can list AI Agents on Marketplace as standalone listings (we have 60+ listed in "AI Agent" category) - these are agents that are not yet transactable i.e. these are marketing listings only (maybe in process of being built etc.). Partners can continue listing in this category.
Once a partner is ready to publish an agent into Agentspace, they will fill an Onboarding Validation form - marketplace will own it as part of product listing. This will capture details around use case, infrastructure, architecture, security, pricing, deployment etc about agents.



For the time being (maybe initial 50 agents), we (Marketplace + Agentspace ) will manually review all inputs and have demo/Q&A sessions with partners
Based on the form + session, we will jointly decide to approve the 3P Agent or NOT
If approved, Partners will publish Agent as SaaS or k8s product type on Marketplace.
Marketplace will do check for security , quality , availability and certify that the product is "compatible" with Agentspace
Marketplace will put "Agentspace" compatible label/type on it. This is the only approved list that Agentspace customers will be allowed to buy/register/use
For details on marketplace, please contact @Oliver Schulz from Marketplace BD for Marketplace Agent Onboarding or PM  Pritish Sinha




How are these priced ?
Customers will buy the agents via a private offer on the marketplace. The partner providing the agent has control over the pricing and the billing is handled by Google through the marketplace. 

1P Google agents: Idea Generation Agent, Deep research Agent

What is the Deep research agent? 
The Deep Research Agent conducts complex investigations and generates comprehensive reports from various sources.

How do customers get access to it? 
This is available to all customers who are in the early access program for Agentspace by default. 
What is the Idea Generation agent? 
It's an OOB AI Agent within Agentspace that helps  enterprise users generate novel ideas, solutions, concepts and rank them using a unique tournament-style framework based on user-defined goals and criteria.

How does it work? 
The Idea generation agent coordinates a system of autonomous agents working together for dynamic ideation. It learns from your enterprise data and iterates to generate truly applicable solutions 


How do customers get access to it? 
This is being launched in preview for customers that are using Agentspace. 

Customers need to get a special allowlist done for Idea Generation Agent. We will work with AISS leaders to select the initial list of companies to work with.  Note this is for Agentspace customers who have bought licenses or are in TTP. 

Agent Pricing

Will pre-built agents have additional pricing?

Pre-built Agent inclusion, availability and included quotas will vary by edition. Included quotas, if applicable, will be higher for higher edition tiers. 
Agents may be monetized separately depending on how resource intensive the agent is. For example: Idea generation. But details are not finalized yet. 


How will no code agents be priced? 
Below is the ethos based on which pricing for no-code agents will be finalized. As a key note, Monetization shared below is subject to change. Final Agent monetization is expected by I/O.  

No-code Agents
Agent creation 
Unlimited for Enterprise Plus
Limited for Enterprise  [Limitation details TBD] 

Agent Usage
Underlying component usage of no-code agents will count against rate limits and quotas 
Usage metered by call type: content gen, tool (e.g., search grounding), action (e.g., create ticket)

Will there be any specific quotas or rate limits in Agentspace? If so, how will they work?


Yes, there will be rate limits and quotas for specific features in Agentspace to limit abuse or disproportionate resource consumption. 


Rate limits and quotas are expected to be established based on estimated average complexity and usage recurrence per target persona of each edition. Below is the ethos based on which quotas and rate limits will be applied. All quotas and rate limits are subject to change and are not currently implemented.


Rate Limits
Rate limits for our highest edition, Enterprise Plus, will be designed to offer effectively unlimited usage while limiting abuse or disproportionate resource consumption. Rate limits for lower editions will be more constrictive. 
Agentspace reserves the right to reduce limits during peak usage times to ensure broader accessibility 


Quotas (hard cap)
Quotas will apply to high value, highly specialized features (e.g: video generation)
A hard cap quota already exists for Data Indexing (e.g., 50 GiB pupm, pooled)
Quotas will be higher for higher edition tiers
Once a quota is met, users may continue usage via PayG rates (with administrator controls) [ to be determined] 


NotebookLM Enterprise integrated into Agentspace

What is the NotebookLM Enterprise x Agentspace integration
Customers using Agentspace now also get access to NotebookLM Enterprise capabilities such as upload documents, chat with the content, create reports, generate podcast, mind maps, etc. With this integration, users can add search results in their existing / new notebooks, search through Notebooks and even get offered notebooks as autocomplete content suggestions. All this Matching data regionality as Agentspace with cloud first compliance and security including  SEC4, AXT, DRZ, CMEK, VPC-SC, HIPAA. 

How do customers get access to it? 
Customers in the early access program automatically get the NotebookLM Enterprise as a part of that. 

How is this priced? 
Included in the pupm license cost for Agentspace. 

Agentspace and Chrome Enterprise Integration

What is the Chrome x Agentspace integration
The new Chrome Enterprise integration with Google Agentspace brings AI enterprise search into the browser. This allows faster discoverability with instant search across internal data, third-party SaaS apps, and external sites directly from Chrome. Employees can also trigger AI-driven searches and get instant suggestions and summaries. This will launch in private preview at Cloud Next and requires an Agentspace license. Interested customers should work with their Google Cloud sales rep to learn more about Agentspace and their purchasing options.

When are we announcing the Chrome x Agentspace integration?
We are announced the availability of the Private Preview on April 9th, at Google Cloud Next ‘25. 

How can customers express interest in participating in the Private Preview?
At this time, we are only accepting submissions from existing Cloud and Chrome customers. The Cloud and / or Chrome GTM representatives for these customers can submit this application form on behalf of their interested customers: go/chrome-agentspace-form 

When do we expect this product to be Generally Available?
During the Private Preview, we will validate that the current solution works well across a broad range of enterprise environments. We intend to move to General Availability after the Private Preview. There is no set date for this at this time, but we believe this is likely to happen in the second half of 2025.

What are the product changes as a result of launch?
Summary:
Agentspace end users can get search suggestions directly from Chrome’s address bar
Agentspace end users can send commands to Agentspace directly from Chrome’s address bar

This results in a better search experience for Agentspace users, as they get more relevant search suggestions as they work in Chrome, without needing to modify their existing search habits.



Which Agentspace SKUs are supported?
All Agentspace SKU tiers are supported. 

Does my customer need active Agentspace licenses in order to use this?
Yes.

More details here:
go/chrome-agentspace-overview-gtm
go/chrome-agentspace-chrome-admin-setup
go/chrome-agentspace-form
go/chrome-agentspace-faq
go/chrome-agentspace-pitch-deck



Overall Messaging and Framing: 

Q: What is Google Agentspace? 
See Agentspace elevator pitch and differentiators



Activation FAQ: 

Q: What are the launch timelines for Google Agentspace? 

Google Agentspace has been publicly launched on December 13th, 2024.
It is launched as “GA Allowlist”. Public GA date is yet to be determined but likely H1 2025. 



Q: What does GA Allowlist mean and how do I get my customer on Google Agentspace? 

GA Allowlist means Google Agentspace is available with the guarantees that come with a GA product but use and onboarding is reserved for a select set of customers as of today. You need to nominate your account to be included in the Trusted Tester program.  Please note nomination DOES NOT yield to automatic addition to the Trusted Tester Program. 
Accounts are added to the TTP program based on customer use case, compete situation, connectors & features required by customer vs. what is available in the product, etc. Please file for nomination at go/Agentspace-ttp and check the status of your nominations at go/ttp-Agentspace-status
Prioritization of the accounts is managed by AISS teams. Hence please email regional AISS leaders to prioritize your accounts. Accounts not in the TTP will show as “Not started and Not assigned” 


Q: I have nominated my customer for Agentspace TTP but I have not received any acceptance communication. 

The display of a customer in Agentspace TTP dashboard does not impact customer access to the product. That is only done via the allowlist bug. Form is information for Product/ Eng to capture details on trial period, Onboarding team (GCC/ Partner) and connectors. 

Please follow below steps


Please raise allowlist bug at go/spark-allowlist-request to enable customers to project for Agentspace.  The trial period begins once the allowlist bug is complete. Details here. 

Onboarding Options:

Onboarding Lead Options: 
1) GCC Accelerator See Offer and contact region-wise GCC teams to leverage this option
 2)   Partner: Please select partner from this list of approved partners. For questions, contact Sergio Villani 




What will the Onboarding Lead Do:
a) Lead the Technical Kickoff with the customer
b) Define the PoC with the customer
c) Allowlist the customer's project for the TTP trial  


For status check go/ttp-agentspace-status


Q: How to tag Agentspace opportunity in Vector?



Q: What is the typical Onboarding timeline? 

Onboarding depends on the connectors chosen for the PoC. If we are integrating GWS, that can be done in a day. However please note that we DO NOT commit these timelines to the customer since implementations differ customer to customer. 
We usually ask for 6-8 weeks for a complete PoC which includes integration of 2-4 connectors + evaluation of search results based on a Golden QA set shared by the customer + evaluation of assistant features. 




Q: Why are we not allowing ALL customers to try Google Agentspace? 

We are allowing every customer to try Google Agentspace. With the product being in allowlist GA, we would like to offer a white glove service through our GCC or enabled set of partners for the best experience and so during this time, a partner/GCC is required for onboarding.
With the TTP program, we are also learning with every implementation. While we have launched a set of connectors, the implementations of certain connectors such as Sharepoint, SNOW are very diverse  across customers (Nested ACLs, etc)  The focussed approach of working with a set of enabled partners and our GCC helps us bring these diverse aspects back to the product. 

Q: Will we be offering any PSO offers for onboarding?
Yes. GCC accelerator opinion has been launched in which GCC teams will work with you for onboarding, paperwork, BIF approval etc. Details here 


Q: Can I actively share information about Google Agentspace with my customers? 

Yes you can share information but note that you cannot guarantee that they will get access during the allowlist period. We do encourage you to start the sales cycle by qualifying the customer and starting the conversation on the value proposition of the product and get them onboarded onto NotebookLM Enterprise which offers a faster and self-onboarding path. For detailed instructions refer to Google Agentspace Lead Nurturing: A Step-by-Step Guide


Q: How do I try Google Agentspace?

We’ve launched a version of Google Agentspace on a select set of internal Google data at go/spark-dogfood. You will see a fully hosted web application with personalized home page and can search across:
Google Cloud Documentation
Your personal Google Drive: This is any content that you have access to in the org’s workspace.
Buganizer data under component 395946. Data is indexed on a biweekly basis, so there may be a lag.
YAQS: All of Yaqs data. Data is indexed on a biweekly basis, so there may be a lag.
Google’s people directory
Note: 
We frequently update the internal dogfood based on Googler feedback so you may see ongoing improvements to the features, UX, and quality in the dogfood. 
Agentspace dogfood only has a small subset of data sources (outlined above) compared to MOMA, so search results and answers may not be the same for the same search across Agentspace and MOMA. 
As Uncle Ben said, “With great power comes great responsibility”. Help us improve Agentspace by filing bugs and feature requests at go/Agentspace-dogfood-newissue. 




Q: I want to create my own demo or internal Agentspace instance. How do I do that?

You can also get your own project allowlisted at - go/spark-allowlist-request. You will be able to integrate 1p connectors and 3p instances if you have personal access to any. 

 

Q: Are demos for Google Agentspace available on go/demos? 

Demo is  live at go/demos/demo/1145. 
Individual sellers can access the username/password from Valentine. 
Sellers will get access to a Demo Guide which takes them through Logins, Scenarios, Suggested prompts and known issues. 
Demo site is also being uploaded with demo videos on How to and a few scenarios.
Additional click-through and video demos for specific industries and use cases are also available in the Agentspace - GTM Playbook



Q: Do I need to request an additional quota for my customer?

By default all Agentspace customers get an ingestion quota of 1M docs. If your customer is using connectors such as Slack, Jira or Confluence, they will quickly run out of quota since we consider each message/comment in these systems as a document.

Please make sure you get the total count of all documents across all connectors from your customers. If the count exceeds 1M then please follow the process outlined here to request a quota of up to 100M docs 



Q:  What costs are expected in the Pilot implementation of Agentspace?

Currently Google is offering 0/30/60/90 days of free  trial at seller and their manager’s discretion. GCC/Partners can help with the initial set-up and BIF/DAF can be used to offset the cost. Some large organizations may have multiple free trials with different teams.

Q: Can customers get our different support tiers for this? And if so, how is support priced for this? Is it the regular x% based on the tier?

Customers can take advantage of their GCP Support tiers for Agentspace. Agentspace pupm license cost doesn’t include GCP support cost.

Q: What if I have a smaller customer like an SMB/startup that is not likely to get prioritized for allowlist?
For smaller customers, a regional or global SI is an option to help with onboarding.  We have onboarding 19 priority GSI/SIs that can help with onboarding some of the smaller accounts that do not get prioritized for the initial cohort of onboardings driven by internal teams. Additionally, if there is an existing G/RSI that you or the customer are already engaging with, we can give them access to the onboarding methodology so they can perform self-onboarding, familiarize themselves with the process, and onboard the end customer. See partner FAQ below for more details. For SMB, please reach out to Mack Bari who can share more guidance.

In parallel, the product is evolving to support a more self-service onboarding and as we head in that direction and lift allowlist (targeted for Next 2025) smaller customers will be able to self-onboard.

Lastly, NotebookLM is also an option to allow customers to get onto the platform while they wait for Agentspace onboarding.

Q: How do I generate a quote, get approvals, generate a contract, and get signatures, for my Customer?:  

Follow the steps outlined in Agentspace Quoting in CPQ.   

Product FAQ: 

Q: Who is the target customer and buyer and user personas? 

Target customers: 
Customer prioritizes employee experience and the importance of improving employee journeys to drive efficiency and productivity as well as adoption of AI and advanced technologies like assistants and agents.
Industry: Primarily service-oriented industries where employee interactions are critical to success (e.g., retail, hospitality, financial services (retail banking), healthcare, telecommunications, insurance, automotive, software & internet).
Size: Mid-sized to large enterprises that have over 1K employees and ideally over 10K
Segments: Select, Ent, Startup (Later stage)
NORTHAM: Subregions: All Geo Subregions and all strategic industries
EMEA: Geo-clusters: UK/IE, North, South
LATAM: Enterprise and Corporate accounts; retail industry; all sub-regions
Technology Adoption: Companies that are already investing in digital transformation initiatives and have some familiarity with AI or machine learning concepts, contact center solutions, eCommerce investments, mobile application development (omni channel), mature marketing stack.


User Personas :
Knowledge Workers:
Description: Employees who rely heavily on accessing, creating, and sharing information as a core part of their job. This includes professionals such as analysts, consultants, engineers, and content creators.
Needs: Quick and easy access to relevant information, seamless collaboration tools, and centralized knowledge repositories.
Use Cases: Searching for documents, policies, or past project details; collaborating on content creation; sharing best practices across teams.
Examples: 
Query
Possible Datasources
Summarize the top customer issues for XX product from the last 3 months” 
Jira, Salesforce, Slack
Find bugs in my code
Gitlab, Github
My customer is in the retail industry, help me understand what are the recent trends that could be impacting my customer” (search information across salesforce and grounded by ex
Salesforce, Web grounding
List P0 jira issues assigned to Alex
Jira, Slack
Generate content for an Instagram post for our new product targeted at Millennial dog owners in the US
Content grounded on GDrive documents


Target buyer/s: 
Chief Information Officer (CIO) / Chief Technology Officer (CTO):
Needs: Advanced AI capabilities, integration with existing IT infrastructure, enhanced productivity tools, and secure, scalable solutions.
Motivation: Streamlining IT operations, reducing costs, increasing technological capabilities, and driving digital transformation.
Chief Operating Officer (COO) and Chief Executive Officer (CEO) or Head of LoB / Departments such as HR, Customer Service, Sales, IT, etc
Needs: AI-powered tools to improve employee engagement, customer support, reduce response times, and provide personalized experiences.  


Q: What are the Google Agentspace SKUs? 

Agentspace is offered as a fully-managed, out of the box SaaS application in four SKUs: NotebookLM Enterprise, Google Agentspace Enterprise and Google Agentspace Enterprise Plus and Frontline Worker Sku. These are sold on a per user per month license model. Details on pricing are covered later. 


NotebookLM Enterprise tier includes:
Enterprise version of NotebookLM Plus 
Same UI as Consumer NotebookLM
Easy setup (no connectors)
Support for Google and non-Google identity
Sec4 compliance (coming Q1 FY25)
Cloud ToS




Google Agentspace Enterprise tier includes:
Blended search across all enterprise applications (1p/3p)
Summarization
Citations
People and multimodal search
NotebookLM Enterprise
With data from connectors


Google Agentspace Enterprise Plus tier include all Agentspace Enterprise features plus: 
Ability to ask follow up questions
Actions in 1p/3p applications
Upload+QA with content
Agent creation
Deep Research Agent


Google Frontline Worker SKU tier which is targeted towards shopfloor, retail frontline employees, etc. SKU details are here: [Seller-facing] Frontline Worker Edition




Q: What can a customer experience when they start using Google Agentspace? 
See Day 1 value FAQ

Q: What connectors are available and what's the roadmap? 

Here are the connectors available within Google Agentspace with the roadmap details. A few Happy connectors implemented in customers' environments are marked in green. 

Connector Ranked Order
Stage
GCS
Public GA
BQ
Public GA
Cloud SQL
Public Preview
Spanner
Public Preview
BigTable
Public Preview
Firestore
Public Preview
AlloyDB
Public Preview
Google Drive
Private GA
Sharepoint Online
Private GA
ServiceNow
Private GA
Salesforce
Private GA
Confluence on Cloud
Private GA
Jira on Cloud
Private GA
Slack
Private GA
Microsoft OneDrive
Private GA
Box
Private GA
Dropbox
Private GA


Vertex AI Search Connectors Roadmap (go/es-connectors)
Q: We have On Prem connectors. Will those work? 
We have Jira and Confluence Onprem connectors available. To learn more about the connector available and status of launch and roadmap, please check here. 

Q: Where can I find technical documentation on the connectors? 

The go/es-connectors has Internal FAQ and external documentation links per connector. Please refer to those until we launch the formal documentation on the site. 


Q: What if my customer's connector is not on the roadmap?

Customers can upload data to GCS based on VAIS schema and also ingest ACL. Details below: 
https://cloud.google.com/generative-ai-app-builder/docs/data-source-access-control

Q: Can I use Agentspace ACL connectors in other products?
See details here


Q: What languages are supported by Google Agentspace? 

Please refer to the roadmap for languages supported.

Agentspace language availability depends on language support in Vertex Search and Gemini. Hence please do not assume that Vertex Search languages will be automatically supported in Agentspace. 

Note: 
Languages supported in Search Tier will also be supported in Assistant
We do not support multi-lingual data sources 

If you have a request for a new language, please file a language FR here. Please note this does not guarantee addition to the roadmap.   


Q: What ‘multimodal' capabilities are supported in Agentspace? Does it include images and video? 

Agentspace includes the capability to upload, search and ask questions of images and videos. For example; You could upload a  company logo and ask a query regarding the color of the logo. The current file size limit for images is 30MB 

Q: Can I ask the assistant to format the answer in the form of tables and/or charts? 
Yes you can ask the assistant to create the answer in a table. Check this out and try it out in Dogfood. Creation of a chart / graph is a capability being worked on. 




Q: Can I sort the documents based on the latest (recency)?
Yes, Agentspace offers sorting options similar to Google search. Below is an example from dogfood: 




Q: Does Agentspace lean on Knowledge Graphs (KG) for search? 

Yes. Agentspace creates a Knowledge Graph from the connector data ingested and indexed by us. The integration of this KG into Agentspace dramatically enhances the output  in several key ways:

Improve search relevance
Entity linking: resolving people/org identities in queries + relationships (e.g. “my manager”)
Personalization: E.g. role-based personalization for cold start (e.g boost the document belonging to people in the same org)
Content recommendations
Recommend content that people around you in the KG are looking
Autocomplete
Include top queries from people around you in the KG
UI Features
People search cards
Knowledge cards from public KG (with permission from Search)

Data sources:
Launched: 
3P People data connectors (Person entity): e.g. People via Custom Connector, Microsoft Entra ID.
Sharepoint (Articles) connected to people


How does it work?
Entity linking detects the entity mentions in search query and matches with KG entities. (e.g Resolving “X” to my connections in KG)
KG annotations provides confidence score and user relevance score based on org chart, affinity, etc. (for personalization)

What kinds of queries get the benefit?
Entity-Centric Queries (e.g people queries)
Relationship-Based Queries (e.g “bugs by my manager”) 
Multi-Hop or Complex Queries (to be prioritized based on customer demand)


Other Benefits: 

Enhanced search accuracy and quality: KGs enable more precise entity matching, and boost more authoritative results based on their connections in the KG leading to better search results. 

Unlocking relational queries: Ask questions that connect multiple entities at once and expand how much you can achieve with Search. Example queries you will be able to answer: "Who is the manager for Project X?" or "Show me bugs assigned to me?”

Personalized Search Experiences: By leveraging the relationships within your data, we can personalize search results. You'll see documents, projects, jira issues or people more relevant to your role and interactions. Example: documents clicked by your team members will be higher in ranking (Dec ‘24-Q1 ‘25) 

Public information: Access public KG information about organizations and concepts helping supplement your search with useful information from Google’s KG. 

Knowledge panels for deeper insights: Get rich context and insights directly in your search results. Knowledge panels will surface key information about entities like organizations, and concepts, giving you a comprehensive view in one place.


To enable the Knowledge Graph un customers account, please follow the below steps: 

What to do: Here are the steps that need to be followed:
Find your customers: Please locate your ldap on this spreadsheet. go/agentspace-data-quality-onboarding. (OnboardingOwners tab, column C). If you do not see an account that you are involved with, please add a row here.
Confirm your role: Please make sure you own the account’s onboarding. If not, please add the right owner on Column C, and forward this email to them, cc-ing vais-spark-triage@google.com.
Engage your customers: For your account, please reach out to the customer, and ask them to enable private KG feature if not enabled and ingest the people data. The steps are outlined in this document: go/agentspace-kg-get-people-data.
Notify upon completion: Once this is done, please reply to vais-spark-triage@google.com confirming the change.

For any of the above steps, please do reach out to vais-spark-triage@google.com and we can help.


Q:  Is the knowledge graph at app engine level? or at datastore level such as GraphRAG datastores?
it is at app level

Q: Will Agentspace come in a mobile app?
Agentspace UI is mobile friendly. Mobile apps will be launched at the end of Q3.

Q: How does Agentspace ensure access control is being honored? 

Connectors read data & ACLs from a 3P data source (per entity/table) and convert them into documents and ingest & index them via Document Service
During search, ACLs (user’s email and Idp groups) are read from their authentication token, along with the external groups, in the search service
ACLs are used to filter out documents that authenticated users do not have access
Search Service returned back to the end user in the search response with only the documents that user can access

This deck has details on access controls

Q: If customer’s enterprise data is messy, how does Agentspace help?
There are a few Agentspace capabilities that can help with this:
Boost and bury features for specific data sources
As a user you can filter sources or by date
Search results are also organized based on user activity, popularity signals etc thus bumping up the most used documents. 

Q: Which models are used in Agentspace? When will we add Gemini 2.0 in Agentspace? 
Agentspace uses a plethora of models including hi-fidelity flash, fine tuned models of Flash. Gemini 2.0 has been integrated. Agentspace Chat Assistant is NOW supported by Gemini 2.5 Flash. Details in go/agentspace-featurelaunch


Q: What is Workforce Identity Federation? 

The Onboarding team will take the customer through these details.

Workforce Identity Federation allows customers of GCP to seamlessly use their existing IdP to securely access GCP services without the need of identity provisioning.

Enables customers to access GCP services and APIs via Syncless Federation

Enables customers to leverage their existing identity investments using standard identity protocols (OIDC, SAML) 

Enables customers to meet sovereignty requirements for identity / personal data. 


WIF is based on exchanging external IdP credentials (e.g., SAML assertion) for GCP credentials (e.g., 
).
The GCP STS Token carries information from the original token in the form of attributes that IAM can use for authorization decisions.



Enables customers to use external IdP to authenticate and authorize their users to access Google Cloud services.
Customers don’t need to sync external Idp into Google identity.
After user authentication, information received from the IdP is used to determine the scope of access to the Google Cloud resources
The information retrieved from IdP is cached until the STS token expires
Requires Workforce Pool setup from customers within their GCP org
Workforce pool is a set of users like employees or partners having similar access requirements.
Within workforce pool, customers add workforce pool providers
Used to define a relation between a customer's GCP org and their IdP.
Requires customers to create an integration app within the Idp
Setup attribute mapping for subject and groups
Assign Discovery Engine viewer role to the workforce pool within IAM.
Preview search supported in Google Cloud Federated console.



Q. What are the Requirements and IAM permissions for Agentspace? 

For Datastores and App set up:

1. Enable Vertex AI API
2. Grant ‘Discovery Engine Editor’ to the tech lead setting up datastores and app engines

For UI/WebApp End User:

Google Identity:
1. Enable Vertex AI API
2. Grant 'Vertex AI User' + 'Discovery Engine User' to the end_user group (this group should include the admin setting up the search app)

Third Party Identity:
In addition to the above,  attach 'Vertex AI User' + 'Discovery Engine User' permissions to WIF pool. To set this up, in the IAM UI, one can type ‘principalSet://iam....’ to add roles to a workforce pool, see this example: 



Specifically "aiplatform.endpoints.predict" and "discoveryengine.assistants.assist" are needed
Note: "discoveryengine.assistants.assist" is not covered by 'Discovery Engine Editor’


Q: How many instances of Agentspace should I create internally? If I have a set of users that use Sharepoint and OneDrive and a set of users that use SFDC and box should I create 2 Agentspace Instances? 

Ideally you should only create one Agentspace instance for your entire organization. Agentspace respects ACLs on the source systems so the 2 sets of personas will only have access to data from the 3P systems that they have access to. If a user has access to all the systems then they don’t need to jump between different Agentspace instances. Having one instance also reduces the maintenance overhead on the customer’s end.  


Q: Why does Google need to ingest all my 3p data for Agentspace? 

Search Quality is an important foundation for the productivity tasks facilitated by Agentspace. To ensure Google-grade search quality, it is essential we ingest and index all the data for KG creation and efficient search. Access to the data also helps in adding enrichments and provides the opportunity to utilize our deep parsers, chunking mechanisms etc when required to improve the search signals. 


Q: Why does Google need ‘Global Admin’ permissions to my data sources? 

All of the Agentspace connectors are ingesting document content and access control lists (ACLs). ACL information is considered highly sensitive, so ingesting it from a 3rd party application requires a highly privileged role. Normally SaaS applications will reserve ACL ingestion to ‘Global Admins’ or other similar roles. Keep in mind that these roles will have both read and write permissions. Although data ingestion does not require write permissions, write permissions make a role highly privileged. 

If your customer expresses concern about sharing admin permissions for their applications, please bring us in the conversation. Please email here for help from PM & Eng teams. 


Q: How often is data (content and ACLs) refreshed as changes are made in my SaaS Applications?

Customers can select the frequency at which data is ingested from 1p and 3p sources into Agentspace. Currently we offer the following frequency intervals: 

Each connector allows for a synchronization frequency. Currently, the shortest amount of time supported is every 24hrs. During these incremental syncs, all new content, modified content, and deleted content is synced over. This includes access control changes made in the SaaS application. We are working on reducing entity syncs to 3 hours and ACL sync to 30 min in early Q1
Access changes made in the identity provider (ex: new users joining, users leaving, group membership changing) sync much faster and are dependent on how often authentication tokens are refreshed. This is usually configurable within the customer’s Identity Provider. 

Q: Can I filter what data can be accessed / ingested by Agentspace? 
At the time of ingestion, connectors can be set up to ingest specific data into Agentspace’s data stores. Eg:- specific Sharepoint sites like sales & marketing sites but not legal. This limits the data Agentspace has access to and indirectly what the end users can search/interact with.
At the time of query, users can select to exclude data stores from their search/interact process.  Eg: While searching, a user can toggle off Jira. 

Q: Will the customer be charged ingress/egress?

We don't charge for ingress/egress. We only charge the data stored in our system ($5 per GB per month) and for Agentspace, customers will receive a 50GiB per user monthly data indexing pool, shared across all active licenses, providing a strong foundation for most use cases. 


Q: Is there an estimated time for data ingestion for different connectors? 
	
Data ingestion is affected by a couple of factors, e.g. number of documents, type of data, advanced features enabled, etc. The following number is as of Sept 2024 and is subject to change.
Structured: 200 documents / second
Unstructured (digital parser): 200 documents / second
Unstructured (OCR): ~ 37 pages/second 


Q: How can we ensure that the data ingested by Agentspace doesn't leak? 


AgentSpace employs a hybrid tenancy architecture to address stringent security, isolation, and compliance requirements. The serving architecture and indexing storage operate in a multiplexed way with strong isolation guarantee, leveraging Google's Borg infrastructure, while other storage and processing components are built on a multi-single-tenant model, backed by tenant projects within customer-specific GCP environments. This ensures that critical data and processes remain isolated and under the customer’s control.
We implement industry-leading security controls such as VPC SC, CMEK, AXT, robust tenant isolation, RBAC, and ABAC to enforce strict data separation. Authentication and IAM policies adhere to least-privilege principles, ensuring no customer’s data or users can interact with another tenant’s environment. Our infrastructure aligns with stringent compliance frameworks like ISO 27001, SOC 1/2/3, and DRZ, regularly audited by independent third parties to maintain best-in-class security and privacy standards..


Agentspace is AxT compliant - what that means is  we provide detailed AXT logs that record Google administrator activities in real-time, offering full visibility into any access to customer 
data. Integrates with monitoring tools like Cloud Logging and SIEM solutions, allowing customers to monitor, audit, and receive real-time alerts on access events. Offers AXA allowing customers to explicitly approve or deny Google support engineer access to their data before it happens.


Q. Where is customer data stored? 
The data is stored in a Google managed tenant project and also backed by Google’s underlying proprietary storage system. This tenant project is directly linked to a customer GCP project that manages the configuration of the tenant project, and includes support for VPC-SC, CMEK encryption, IAM policies and other security controls. The data is owned and controlled by WF and all data stored adheres to the security controls defined by the customer.  


Q: Can I index my external website as one of the data sources within Agentspace?
No, Currently we do not support indexing external websites as a part of Agentspace. However we will soon support internal crawlers as a part of Agentspace. For timeline please monitor go/es-connectors


Q: Is there a limit in the number of data sources, data or file sizes that I need to be cognizant about? 

Currently  a maximum of 50 datastores can be connected to an engine at a given time. Keep in mind that a single 3P connector can produce several datastores (1 datastore per entity). 

https://cloud.google.com/generative-ai-app-builder/quotas
Notably, for each project, the default limit is as follows and can be requested by customers to increase on Cloud Console following the guide:
Up to 100 engines (Upper hard limit 500)
Up to 100 data stores (Upper hard limit 500)
Up to 1M documents

Other notable constraints (normally not adjustable):
Structured data: up to 500KB per doc
Unstructured data (HTML/TXT): up to 2.5MB per doc
Unstructured data (PDF/DOCX/PPTX): up to 200MB per doc (adjustable per data store with exceptions)


Q: Do I need to use the Agentspace UI? Can I integrate Google APIs in my intranet UI? 

Yes, We suggest you use the Agentspace UI for accessing the added functionality we are planning for aka announcements, recent documents, recommendations etc. 

The UI / Knowledge hub can also be customized to your branding and organization applications look and feel. Having said that, Agentspace functionality is available via OOTB and API. You can use the APIs and integrate with your organization intranet portal as well, if any. 


Q:  Are connectors available for Vertex AI Search use cases? 

Going forward “Vertex AI Search datastore” connectors will be limited to Agentspace customers. Vertex AI Search (VAIS) applications will have access only to the following connectors: website content, BigQuery, Cloud storage, Cloud SQL, Spanner, BigTable, Firestore, AlloyDB, and APIs.

Both Workspace connectors (Gmail, Gdrive etc) and 3P connectors (Jira, Sharepoint etc) for Vertex AI Search will be available via Agentspace license, unless an exception is made. To make an exception please file a bug with the details below. CES customers will get an exception as long as it is an external use case. The OPM and PM team will review bugs filed on a weekly basis to approve or reject the exception.

Use case of the customer for the connector with details of internal employee vs external chatbot use cases.
Expected number of API calls per month if the customer needs a consumption pricing model 
Any executive sponsorship information for the customer making them a strategic customer for Google

Customers that already have access to connectors outside of Agentspace will continue to have access. However, a separate MSA will be sent out to specific customers in advance with ample time, in case their connector access would be removed sometime in the future. 


You can reference the details here.




Q. Can my customer/ partner build a custom connector?
Documentation is being defined right now but if you have an urgent ask for it, please use this. 

Q: Are any impacts expected on the technologies that will be consumed? (SharePoint and ServiceNow)
The connectors are configured to ingest data/ACLs at a specific frequency (daily, every 3 or 5 days). These data reads are non-invasive with minimal stress to the system. They can be set up so they can run during off-peak hours.
Q: What are User events? 
User events capture end user activity and feed into the Analytics dashboards in Agentspace and capture metrics 
User events capture what your users are searching and interacting with; this data is used to improve future search and recommendation quality over time 
User Event Object
Export User Events to BQ or GCS: go/vais-export-user-events-guide   

Q: Do we have auditable logs? 
We have built a private feature to allow customers to specify a BigQuery table to store their detailed audit logs (guide). File a bug to Andy Zhang Lei Chen to allowlist if customers need detailed logs beyond the regular Cloud Audit Logging.

Q: Which analytics will be made available to the admin on search, agents created by users and their usage.


Usage metrics: We will have search/answer/summary DAU/MAU, user retention (7D, 28D), page visits (e.g. homepage, agent gallery, prompts, etc). Available by GA
Value metrics: Value generated by agentspace (search value metrics  will be in GA, summary/answers will be in private preview)
Metrics Export API: An API will be provided to allow customers to export analytics metrics from bigquery view to their own storage solutions. Available by GA

Q: When will Gemini 2.5 be incorporated into Agentspace?
Right now, Gemini 2.5 flash is in Preview for Vertex LLM. We will introduce it once
1. Gemini 2.5 is GAed on Vertex LLM.
2. Finish Agentspace evaluation / system testing
And we are targeting the end of June. 
Q: Does Agentspace have versioning? Additionally, how are we planning to communicate feature releases?
Agentspace does not currently support versioning and is not on the current roadmap. New feature releases will be announced via Agentspace Release Notes
Q: What is the roadmap for Integration with Slack and / or Google chat to invoke Agentspace. 
Slack integration is planned for end of Q3. Google chat is not yet planned. 
Q: Where can I find more information about Agentspace security?
Check this FAQ - go/agentspace-security-faq

Day 1 Value FAQ
Q: How does my customer get Day1 Value from Agentspace? 

The Day 1 value initiative is a focus on providing a progressive journey for a customer where we can provide a meaningful experience, without connectors, as a first step in Agentspace in parallel to connector setup.
Customers can benefit from Agentspace the moment they are allowlisted and get access. We can showcase the platform's immediate, out-of-the-box (OOTB) capabilities that provide value without requiring deep integration into existing enterprise systems or connecting proprietary data on day one. This initial experience emphasizes ease of user ramp-up, minimal training, expedited change management, and a straightforward technical setup. It also inherently includes Google's robust security, privacy, compliance, data residency, and responsible AI practices.
Please use this deck for Day 1 Value overview with suggested prompts and demos. Check the Appendix section for additional assets including the internal Day 1 value guide.
Go/agentspace-gtm has a section dedicated to day 1 value as well. You can find the links to the PoC guide and Playbook in the deck and on the site. 
Please  note: Day 2 value experience is NOT a separate SKU for Agentspace. Rather it is the initial experience included in Enterprise Plus edition.  


Agents FAQ

Q: How do I register ADK agents on Agentspace? 

Feature: Register your ADK agents to Agentspace [Private Preview] 
Agentspace customers can use the ADK to build these agents and register them on Agentspace. With this, customers can now access both no-code and high-code agents in the same UI. Please review this attached document here for the registration process. 
How to register and use ADK Agents with Agentspace.pdf
This same documentation will soon be published on the Agentspace GCP documentation website as well. This is currently launched in Private Preveiw and subject to change. Note: the PDF can be shared with customers.  
To make it easy, we also created this python tool that will help you manage agent registration into the agent gallery using CLI.  Here is a guide that explains how to deploy an agent to Agent Engine and register with Agentspace. Publishing Agents to Agentspace
FAQ: 
For the agents that are already registered with Agentspace through the past workaround, do they need to be re-registered?
Yes You need to re- register ADK agents. There has been some code refactoring which necessitates the re-registration which impacts only early internal demo users. You can keep using these previously registered agents until the engineering team will deprecate and sunset them.
Is the same interface used to update even no-code agents?
This is a common interface with no-code agents. However no-code agents are not yet allowlisted for update.
Is there a way to search an agent by agent display name?
                          No, there is no search at the moment, you have to do a list + do the matching yourself.
Do customers need to pay extra for the ADK agent? 
ADK is an open source framework however customers will need to pay for the run-time. Also to register them in Agentspace, customers need to have an Agentspace license, and there may be a limit on the number of agents allowed for registration depending on the subscription tier in the future. 
Can I use Agentspace datastores to create the ADK agent? 
Yes you can, PTAL at the example colab. 
How can I use the Oauth tokens passed in by Agentspace in my ADK agent?
Here is an example colab.
Would the agents running in Agentspace automatically have access to the search capabilities that cuts across all the datasources along with the Knowledge Graph that built into the Agentspace application?
See above colab on how a VAIS backend could be accessed from within an ADK agent.
I have trouble with Agent triggering, what can I do?
This will soon be fixed. When you access an agent on the left navigation panel, all the queries should go directly to your agent, without the Assistant planner interfering.
Agents added using this method are shown in the “Your Agents” section in the Agent Gallery instead of in the “From your Organization” section
                          This is a known issue. It is being fixed.
How do we register agents built using Langgraph, Langchain and other open source frameworks? Is the process the same?

This is not supported yet, but stay tuned, A2A support coming soon.


Q:Is there a way to embed custom UI in Agentspace or is there a plan ? Something like how creating a calendar invite shows the Calendar UI for editing or Email UI is displayed before sending. If we have a Custom Agent that needs inputs or we want to generate a multi Modal output, how to accomplish it ?
We are gathering requirements for UI for other agents in Agentspace. We are looking at using iFrames as a stopgap solution but are looking at integrating Appsheet apps as a F/E builder and UI for agents, and looking for ways we can extend the A2A framework to also communicate UI requirements to Agentspace so we can render it within the Agentspace UI. 

Q: Will our Looker product integrate into Agentspace? 
We are working with the Looker team to integrate their conversational Analytics agents into AS. ETA for private preview is EoSept ‘25. 

Q: Please elaborate on SNOW permissions to  create a custom action / agent.
Create a dedicated role and assign ACLs
When you use a non-admin user role, you might encounter issues when you perform a ServiceNow action even though your user role contains the required permissions. To avoid this issue, create a dedicated role and assign the required permissions.
Create a new role
Navigate to User Administration > Roles.
Click New.
Specify a name for the new role, such as database_admin_restricted.
Optionally, enter a description.
Click Submit.
Assign the new role to the authenticating user
Navigate to User Administration > Users.
Select the user requiring the restricted access.
In the user record, navigate to the Roles related list.
Click Edit....
In the Collection column, locate and select the newly created role.
Click Add to move the role to the Roles List.
Click Save.
Implement row-level permissions
Navigate to System Security > Access Controls (ACL).
Click New.
Configure the following fields:
Type: Select record.
Operation: Select read.
Name:
In the first drop-down, select the sys_db_object table.
In the second drop-down, select None
Requires role: In the Insert a new row... field, search and select the newly created role.
Click Submit.
Repeat the steps to implement row-level permissions for the sys_glide_object and sys_dictionary tables.
Implement field-level permissions
Navigate to System Security > Access Controls (ACL).
Click New.
Configure the following fields:
Type: Select record.
Operation: Select read.
Name:
In the first drop-down, select the target table, such as sys_db_object.
In the second drop-down, apply the permission on all fields in the table or select a specific field name, such as name.
Requires role: In the Insert a new row... field, search and select the newly created role.
Click Submit.
Repeat the steps to implement for each specific field within the sys_db_object, sys_glide_object, and sys_dictionary tables. You can also apply the permission on all fields in the tables if required.
The above should be availale in public documentation by July 10th







	Security & Compliance FAQ: 

Q: What compliance standards does Agentspace follow?
		
VPC-SC*: Offers robust integration with VPC SC, enabling customers to create a secure perimeter around their data and services. Provides comprehensive guidance throughout deployment, including best practices for network segmentation, IAM roles, and firewall configurations.

DrZ: Ensures data residency compliance by allowing customers to store and process data within selected regions and zones. Provides encryption at rest and in transit using industry-standard AES 256-bit encryption. Offers DRZ guarantees for data at rest, in use, and during ML processing, ensuring that data remains within specified geographic boundaries throughout its lifecycle.

AxT*: Provides detailed AXT logs that record Google administrator activities in real-time, offering full visibility into any access to customer data. Integrates with monitoring tools like Cloud Logging and SIEM solutions, allowing customers to monitor, audit, and receive real-time alerts on access events. Offers AXA allowing customers to explicitly approve or deny Google support engineer access to their data before it happens.
ETA Dec ‘24

CMEK*: Supports CMEK through Cloud KMS, allowing customers full control over encryption keys. Offers key rotation policies, and supports integration with Hardware Security Modules (HSMs), and support for External Key Management (EKM) systems.

HIPAA: Complies with HIPAA requirements by offering robust security features, including encryption, access controls, and audit logging. Provides BAAs and tools for data de-identification and anonymization, allowing customers to handle PHI securely.

*VPC-SC compliant for all use cases except for actions (e.g. file a Jira ticket, send a slack message etc)

Q: What is the regional availability for Agentspace? 

For the latest on regionalization please go to go/genai-data-residency 

Q: Is Agentspace GDPR compliant? 

We achieve GDPR compliance by offering robust data localization and processing controls (aka DRZ at-rest and in-use), ensuring that customer data stays within specified geographic boundaries (such as the US or EU) as required by law.

https://blog.google/products/google-cloud/google-cloud-rolls-out-data-processing-terms-addressing-gdpr-changes/


Q: What are the Terms of Service for Agentspace?
GCP TOS
https://cloud.google.com/terms 
https://cloud.google.com/terms/service-terms



Q: Can I get help in customer meetings for Security and Compliance discussions?
Yes, please email file an er first at go/new-er then choose NorthAM Tech Solutions, Product Security. Please email  northam-tt-security-ce-aisec@google.com in addition.  This is the NATT team which also works on Agentspace and will help you with customer discussions. 

For EMEA accounts please email: gcct-ce-emea-ai-security@google.com


For APAC accounts please email:apac-secure-ai@google.com


Q: What data is being stored on Google Cloud when I use a 3rd party connector
Agentspace makes a full copy of the data the customer has provided from the 3rd party connector. This copy is stored in a Google Cloud Storage bucket in a Google Managed tenant project. This data is encrypted by default with AES 256 encryption. Data can be protected with CMEK.

Embeddings data is created from the full copy of the data in a separate Google Managed instance.

https://yaqs.corp.google.com/eng/q/8966441458210963456#a1 

If your customer has concerns about this please contact security teams listed above. If we are unable to address their concerns they can wait until the Federated Search option is available.

Federated search would not copy the data and instead make API calls to the source data. This would mean search speed and quality could be degraded compared to copying data. 


Q: How long is data retention on Agentspace?
There is no ability to set retention periods within Agentspace. 
Data remains within Google’s systems until:
Incremental synchronizations occur, reflecting changes or deletions made in your source application.  You could set those on the 3rd party application and with Incremental Syncs that data will be deleted from Agentspace. For example if you set data retention policies on a source system like Slack, data will be deleted from the conversations on Slack. With incremental syncs Agentspace will delete those same conversations stored on the Google data store. Incremental syncs as of March 26th 2025 are not real time.  https://slack.com/help/articles/203457187-Customize-data-retention-in-Slack
You initiate deletions of your data stores within the Cloud Console interface.
Data Deletion Policy: Once data is deleted from AgentSpace, it adheres to Google's standard data deletion policies. This ensures that data is securely and permanently removed from our systems.
https://cloud.google.com/docs/security/deletion 


Q:How do you secure your GenAI Models against adversarial attacks?
At Google, safeguarding our Gemini models against real-world threats is a top priority. We employ a robust, multi-layered security strategy. Here's a detailed look at the key components:
Extensive Testing and Benchmarking: We begin by rigorously evaluating our training data and models using a wide range of internal and public standard benchmarks. This allows us to proactively identify and address potential biases and stereotypes that could lead to unfair, discriminatory, or harmful outputs.
Robust Adversarial Testing and Resilience Building: Our dedicated internal teams actively engage in "red teaming." This involves simulating real-world attack scenarios and attempting to elicit undesirable behaviors or vulnerabilities from the Gemini models. To gain diverse perspectives and challenge our models from different angles, we collaborate with external security experts to conduct thorough red teaming exercises.
Layered Output Filtering Mechanisms: We implement sophisticated input and output filters on Gemini that are specifically configured to detect and flag potentially harmful or policy-violating content.
Clear and Explicit Content Policies: We have established clear and publicly available content policies that outline the prohibited uses of Gemini, including illegal activities, hate speech, harassment, and the generation of harmful misinformation.












Cloud AI Offerings FAQ: 


Q: How can I position Google Workspace with Google Agentspace? 

Agentspace should not be positioned as a competitor or alternative to Workspace
Workspace is a collaboration and productivity suite
Agentspace unlocks enterprise expertise for employees with agents that bring together Gemini’s advanced reasoning, Google-quality search, and enterprise data, regardless of where it’s hosted

Note: There is product product overlap that needs to be navigated

The second Agentspace product tier includes assistant like functionality that has overlapping capabilities with the Gemini assistant in Workspace

Greenfield Customers
Key qualifiers
Federated knowledge retrieval/search against multiple enterprise systems and applications: lead with Agentspace. 
Enable productivity and AI assistants within Workspace, lead with the Gemini assistant within Workspace. 
Broadly deploy agents at the enterprise to boost productivity: lead with Agentspace

Workspace & Gemini customers
Pitch only Agentspace Enterprise (search) tier since customer is already paying for and using Gemini assistant
Take advantage of approved 15% discount on Agentspace for Workspace customers


Customer need/scenario
Recommended
solution
Key
considerations
Enhance productivity
Gemini for Workspace


Agentspace
Best for teams that are heavy users of Workspace apps (Docs, Sheets, etc.) for content creation and collaboration.

Best for teams looking to leverage agents in the enterprise  to boost productivity
Centralize enterprise information and streamline workflows
Agentspace
Ideal for organizations seeking a unified platform to access, analyze, and act on enterprise data.

Leverage AI-powered assistance for various tasks
Both (with considerations)
Agentspace: Best for enterprise-wide search and AI-powered insights across 3P applications.
Gemini: Best for AI assistance within Workspace apps.


Q: How can Google Workspace customers take advantage of Agentspace? 

Google Workspace customers, if they wish to, can take advantage of Agentspace to search across 1p and 3p data sources and Agentspace allows you to do that. 

We offer a discount of 15% to Workspace customers who would like to purchase Agentspace licenses. Please refer to Pricing at the end of this document. 



Q: We have some agents already built using Vertex AI Agent Builder (generative playbooks) - Can those be integrated with Google Agentspace? 

Yes, Agentspace will be able to integrate with Agents. The idea is to initially offer integration with Vertex AI Agent Builder agents and secondarily add capabilities to build low code agents. Roadmap will be shared on Feb 1 2025 



Q: What are upsell opportunities with Google Agentspace?
Google Agentspace is a SaaS like offering that is priced on a Per User Per Month (PUPM) and includes two product editions.  Agentspace is also intended to be deployed broadly within an enterprise with the potential for 10s of thousands or even 100s of thousands of user licenses. Upsell opportunities include:
Selling additional licenses: It is unlikely for a large organization to deploy Agentspace broadly from day one therefore Agentspace will offer opportunities to land into a specific functional department or LoB and expand into other areas in the organization
Moving users up tiers: Upsell search licenses to search + assistant licenses
Migrating customers from Cloud Search or DIY Intranet search to Agentspace




Differentiation FAQ: 

Resources: Agentspace Compete One Pager go/agentspace-compete-1pager
Agentspace Compete go/agentspace_battlecards

Q: What really sets Agentspace apart from other solutions in the market? 
 	
Google Agentspace brings together Google-quality search, the latest foundation models, and innovation in agentic workflows into an intelligence knowledge hub that provides a cloud-compliant search, generative, and agentic experience, all grounded on enterprise data via connectors, and with access controls. 


Agentspace brings together Google’s biggest strengths in Search and Gemini’s advanced reasoning on enterprise data. 

Agentspace includes the privacy, security, data residency, and other compliance guarantees that enterprise Cloud customers expect from our products. 

Best in class Google-quality search blended across enterprise data with the opportunity to ground in web data, if required. The search functionality is based on Google KG developed over decades and investments in document understanding. 

Multimodal capabilities to search, chat with and generate images, all staying within the Agentspace HUB. 

Agentspace boasts of an Integrated experience offering enterprise search, advanced assistant and personalized landing page. This is made possible via the  vertically  integrated architecture supported by Google Infrastructure, Google search capabilities along with the assistant capabilities that work on 1p and 3p data. 



Q: How is Agentspace different from Cloud Search ? 

Cloud Search is Google’s older offering for Intranet search. While the offering has not been deprecated, we would highly encourage you to talk to your customers about Agentspace to bring them in the Gen AI powered Intranet search capability coupled with assistant and agent capabilities. Agentspace also offers the opportunity to access technologies from Google research that enhance the working for a knowledge worker; case in point- the ability to do multistep reasoning and deep research. 

If you wish to upsell Agentspace to your current Cloud search customer, please nominate them. 


Q: How is Agentspace different from Vertex AI search? When should I position which product?

Vertex AI search enables users to design AI-powered search and recommendation experiences for websites and mobile apps, leveraging large language models (LLMs). Customers can integrate both first-party and third-party data to bring this functionality to life, although the current capabilities are limited to search functions. 

Users today expect a broader conversational experience, allowing them to pose natural language queries, create content, conduct research, and interact dynamically with third-party applications for both read and write actions. For this complete experience, customers need to use Agentspace. 

Also note that connector access will be available on a case-to-case basis for vertex AI Search .


Features/ Functionality
Vertex AI Search 
Agentspace
Connectors
Yes. Fill this for access
Yes, needs allowlisting
Blended search across 1p & 3p 
Yes
Yes
Search using Knowledge Graph
No
Yes (Nov-People directory, Jira, Sharepoint) 
Knowledge Hub / UI 
No
Yes
Chat / Search Assistant 
Multiturn Search or integrate Conversational AI (Generative playbook) 
Yes, needs allowlisting
Actions in Assistant (Generate code, text, write back to 3p) 
No
Yes
People Search
No
Yes
Talk2Content
Limited (using Talk2Docs in a RAG) 
Yes
Document Processing (Digital Parsers, etc) 
Yes (in a RAG) 
Yes
Multimodal Search
Limited
Yes (image, video) (V-Dec launch) 
Agentic Framework
Can integrate with Agent (Needs custom integrations with APIs)
Yes, needs allowlisting 
Implementation effort for Agentspace features 
High
Low
Pricing
Pay as you go
SaaS


Here’s a decision tree that sellers can use:


Overall, it would depend on the number and type of connectors they have. This was the case with Meta and they ended up going with us cause they realized their cost of DIY was going to be higher. We have some slides on the value narrative here too:  Agentspace Value Narrative: Customer Q&A. Additionally, even without agents, AS gives you models such as Imagen and Veo, NotebookLM, and enables you to not just search but understand and generate. 

Q: How is Agentspace different from Agent Builder?

Please refer to above for the answer. Agent Builder fka Vertex AI Search. 


Pricing & Licensing FAQ: 

Q What is the Frontline Worker Edition?  (New!!)

The Frontline worker role to expand our Agentspace footprint and impact across the enterprise. The SKU will become available to transact beginning June 9th 2025 (manual process). Details in FAQ.  
List Price: $9 pupm on a monthly term
Seat Restrictions: Customer must have 150 active Enterprise Plus licences to purchase
Data Storage: 2 Gib per user (pooled across both FLW and Enterprise Plus editions)
Included Features: Enterprise search across 1p/3p data,Grounding with Google Search, tasks & actions and Agentic usage for agents created in No code, High Code and by partners. 
Limited feature capability for NotebookLM Enterprise (chat with published Notebooks only) and content generation. 
 No access to Pre-Built agents, media generation (e.g., image, video), or Agent creation / publishing
Eligibility: User must be in a non-information worker role - GCP has the right to determine if a role is deemed frontline (consistent  with GWS terms) 

Billing: Frontline Worker (FLW) is available in CPQ for quoting and contracting.  

                          For additional details, please see the attached FAQ







Q: How can we monitor data usage numbers in order to avoid overages? The concern being the Overages and Data Indexing clauses where we will be paying list price for overages.
Admin dashboard for overages and quota usage is on roadmap for end of July  delivery


Q:  What is the pricing structure for Agentspace SKU’s including NotebookLM Enterprise? 

Please refer to go/agentspace-pricingeditions for details.


Note:

1 - The size of the ingested data will count towards the 75 GiB pooled allocation, not the size of the resulting data index
2 - To further enable an AS + GWS better together story beyond discounting, GWS data (incl. Drive, Gmail, Calendar) does not count toward the data indexing allocation. 
3 - Data indexing overages for Agentspace are charged at the standard VAIS data indexing rate of $5 per GiB/month (SKU ID: BC7D-6A97-90F8)


Please share the pricing with the customer at the appropriate time and utilize the tiered discounts available to you, detailed below. 


Q: What type of discounts are available?

Agentspace discounting has been designed to reward long-term, broad adoption by publicly offering deep discounts for commitments, stackable with GWS preferred pricing and/or enterprise-wide coverage. 

Below is the discounting structure that sellers can take advantage of: 



Agentspace Enterprise
Agentspace Enterprise Plus
Customer visibility
Commit discount
1 year: 15%; 3 years: 30%
Public (programmatic)
Enterprise-wide coverage 1
15%
Public (programmatic)
GWS Preferred Pricing 2
15%
Contact sales
Field empowerment 3
Standard
Contact sales


Defined as a minimum of 85% of an enterprise’s information workers holding valid Business or Enterprise productivity suite licenses, including but not limited to M365 Business, O365 E3 or E5, Google Workspace Business or Enterprise. 
2 Eligible for new and existing GWS Enterprise and Business (excl. Starter) annual commit customers only (GWS monthly customers not eligible); Agentspace license discounts available for up to the total number of eligible GWS licenses.
3 Authorized PM Approver: Vlad Vuskovic (vuskovic@)



Discount TermsDiscounts are stackable (deal pricing discount supersedes field empowerment)
Discounts only eligible for customers that purchase 1yr or 3yr commit SKUs 
Excluded from EDP, aligned with Gen AI products

Commitment Terms
Programmatically offered in Monthly, 1 year and 3 year commitments
Commitments auto-renew by default 
Pricing for custom terms available other than 1 month, 1yr, or 3yr (to support co-term with existing GCP contract) - no approval necessary (more details in License Terms)
Customers may cancel but are responsible for remaining payments on commitment period



Agentspace can also be used to burn down existing commits.


If you need any help with discounts / pricing, please use the following  Agentspace - Field Deal Guidance as guidance. For additional details contact Natasha Zagorodnyaya at the deal desk.  

Q: Is there a trial period for Agentspace? And how many times can it be renewed?

To promote usage and early adoption in the GA Allowlist period, we are offering 0/30/60/90 days of free based on seller and their manager approval, promotional usage for an unlimited number of licenses for Search + Assistant. The trial begins on the date the project is allowlisted. 


Note: Please mention the trial period you choose in the go/agentspace-ttp form. 

Q: How do I extend my Free trial? 

Please submit this form (go/extend-agentspace-trial) with the relevant details. Account information as in Vector, connectors and use cases and the Project Number for which you are requesting the extension is critical. Requests will be reviewed weekly and approved requests will receive communications on a weekly basis. 

For questions, please contact agentspace-trial@google.com or for escalations email katecummings@google.com . Please note that emails will not be monitored regularly, and the form submission will be critical to reviewing your extension request.


Q: Is there a trial period for NotebookLM? 

NotebookLM is GA. Customers get a free trial of 14 days when they purchase the standalone NotebookLM SKU.

Agentspace customers allowlisted in TTP get NotebookLM Enterprise for the same trial period as Agentspace. 

Q: Is there a minimum number of seats or term? Can they start with less than 10 employees or cancel after a few months?

No. There is no minimum seat required for Agentspace Enterprise SKU.
Agentspace FLW Sku has a  minimum of 150 Enterprise Plus licenses.  
Cancellation - Customers may cancel at any time but are still responsible for remaining payments on existing commitment periods


Q: Since data coverages will apply for Pricing, how do I calculate the approximate data size for 3p data? 

Please use this calculator to calculate data size for top connectors. We will keep adding other popular connectors to this calculation. 


Q: How do I calculate the TCO for my customer to estimate costs? 

Please use this calculation for TCO.- go/Agentspace-business-value
If you have a qualified customer opportunity and need dedicated TCO or business value support? Please reach out directly to opeterson@ and aliu@ to discuss


Q. Will there be any specific quotas or rate limits in Agentspace? If so, how will they work?


Yes, see details here: Agentspace_NewPricingStrategy_June2025

Q: How do I generate a quote, approvals, contract and close my deal?

Sellers should follow the CPQ steps outlined in Agentspace Quoting in CPQ.  

In that process you may also request Deal Assistance which will be routed to DSS.  

DPO will follow the process outlined in the DealKB “Agentspace”  
Note that DPO requires your customer to be in TTP for order form creation: 

Q: Who do I contact for issues with Agentspace order forms?
A: Please reach out to https://moma.corp.google.com/person/enichols 


Who are my DPO leads? 
EMEA https://moma.corp.google.com/person/vdh

LATAM https://moma.corp.google.com/person/pauloramos

NorthAM    https://moma.corp.google.com/person/markev?hq=type%3Apeople&q=DPO%20Northam

APAC

https://moma.corp.google.com/person/shankarsraman?prev_event_id=nNXdZ53vN5_i3vEPu-X18Qc&hq=type%3Apeople


Q: : Why are customers being billed for the same user multiple times across different projects?
Answer: Billing is based on user access per project. If a user is a member of multiple projects, they occupy a licensed seat in each one. Because licenses are allocated on a per-project basis, a charge is incurred for each project to which a user is assigned. 
Q: We have purchased a bulk number of licenses. Can we reallocate or move these licenses between our various projects as our user needs change?
Answer: At this time, it is not possible to move licenses between projects once they have been assigned. However, this feature is planned for a future update and is expected to be available by August 2025.
Q: Can customers use cloud console to purchase more seats?
Answer: Customers can purchase additional licenses using Cloud Console only if the original subscription was purchased using Cloud Console. For all subscriptions completed via offline order form, the order form needs to be updated with the new license count. 

Q: Can Organizations upgrade or downgrade Agentspace editions for specific users or projects? 
Answer: Customers can upgrade licenses using Cloud Console only if the original subscription was purchased using Cloud Console. For all subscriptions completed via offline order form, the order form needs to be updated with the new license distribution. Downgrades are not supported at this time. The customer must wait until their existing subscription expires. 
Q: when users leave a project or the organization, customers could be billed for inactive users.
Answer: It is up to the customer to manage licenses in their organization. The customer must deprovision licenses from former employees so that licenses can be reassigned to other users. 

Q:  Many organizations setup separate production and non-production engines. Will the users be billed twice and how does data overages apply in this case?  
Answer:  We recommend 2 options:
Option 1:
Customer have their prod and staging engines in the same GCP project. This way the customer only needs to ingest data once and only needs 1 set of licenses as we allow datastores to be linked to multiple Agentspace engines.

Option 2:
Customer has separate Prod and Staging projects. The customer purchases a handful of licenses in their staging project (20-50 licenses or so). They ingest a subset of production data to do their testing


Q:  Many organizations will want their users to immediately start using Agentspace while connectors are being configured. How would a customer go about doing this?  
Answer: This is the model we propose to customers for managing production and non-production environments. 
Customers create 2 Agentspace engines in the same project - EngineA and EngineB (the customer would not be billed twice)
EngineA is their production one where users are using the chat first experience. EngineB is where they're setting up connectors. 
Once a customer has set up connectors and is happy with the experience in EngineB, they start linking those data stores into EngineA. 
End users now have access to the connector experience



Deal Management Team FAQ: 
Q:  Can we reference a KB for Agentspace?  
Agentspace KB is now available here:  https://support.google.com/dealkb/answer/15757285?hl=en

Q: Is Google Agentspace available in Vector to quote? 
Agentspace is now quotable in Vector CPQ.  (Agentspace Quoting in CPQ see above)

Q: Does Google Agentspace have an available option to “prepay”? 
At this time there is not an option to prepay. In general, Argentum SKUs and no other subscription products through Argentum (like Support) offer prepay.

Q: Can GCP credits be used against Agentspace subscriptions.  Sellers are messaging that this is possible? 
No, Agentspace SKUs aren't in the general purpose GCP SKU Group (since there are not external IDs), BUT if you select Agentspace SKUs specifically in a commit quote, then they can be credited (same way as Support and every other GCP subscription we've sold). 
At this time, you can't offer GCP Credits on Agentspace in the quoting flow, so assume that there is no pricing policy that allows this at this time. (ie., we don't offer credits for Workspace licenses - value is built into the discounted pricing not the credits).

Q: Can the PM team approve an ‘annual subscription fee’ instead of a ‘per user/seat’ licensing approach?
The PM team cannot approve this material pricing exception, please work with your Pricing team for further questions. 

NotebookLM Enterprise Questions 
NotebookLM Enterprise FAQ
Partner FAQ: 
Q: What is the guidance and associated resources and POCs for the various partner paths for Agentspace?

GSIs/RSIs
Sell to
If you are selling to a partner please follow the normal recommended guidance for selling Agentspace that is detailed out on go/agentspace-gtm.  Note allowlist priority will be given to priority GSI/SIs that are willing to self-onboard using the partner facing Agentspace accelerator methodology.  

Sell with
Involving a Regional or Global SI who will perform onboarding autonomously: We published on Delivery Navigator and in partnership with GCC the methodology which a G/RSI can follow to onboard a customer. We strongly recommend that a given partner self-onboards their own organization to Agentspace prior to onboarding one of their clients, which by the way should be a great productivity booster for them internally. To allowlist one of your partners who is willing to self-onboard, secure regional support with your Partner team, and then please reach out to Sergio Villani. Any partner willing to self-onboard and approved by your regional partner organization will be allowlisted. To date, this list includes details on partners that have completed or planned to complete their self-onboarding.

Sell thru
See reseller ordering


Technology Partners
Partners looking to build connectors
Building connectors for other customers: The current priority is for Google Cloud product teams to build our own official connectors for agentspace due to overall product experience and quality dependency on connectors.
Partners can always build custom connectors for a specific customer but this is not considered an official connector 
Interested in improving your company's Agentspace connector: If you have an ISV that has an enterprise platform that you feel needs a connector built for or improved please contact Grace Wu. Example: the Asana team wants to make sure our Agentspace connector is optimized. 	

Partners looking to build agents for Agentspace
Building agents for a specific customer: A partner such as a GSI/RSI is always welcome to build an agent for any customer.  These agents would be specific to that given customer’s Agentspace instance. We are NOT currently looking for partners to create agents which will be shipped with the OOTB Agentspace product right now.
Building Agents for all customers (Marketplace): There are many partners, usually ISVs, that want to build agents for Agentspace to be broadly available to companies. We are working on an integration between Agentspace and our Google Cloud Partner Marketplace. Partners such as ISVs will be able to publish Agentspace specific agents on Marketplace. In turn, Agentspace admin users can go to the Marketplace and search for agents that may be useful to their organization and provision through Marketplace. Once provisioned in Marketplace, the agents become available in the given customers Agentspace instance. There are a few ISVs we are actively partnering with for launch. For more information see this presentation and recording



Q: How do I get my partner allowlisted?

Reminder: allowlisting for Agentspace Enterprise is subject to validation and a partner should be involved for actual onboarding so the onboarding is more involved. On the other hand, allowlisting for NoteBookLM Enterprise is expected to be granted with some rare exceptions and is all self-onboarding.

With regards to Agentspace Enterprise:

This is strictly related to the sell-to motion: Once you have completed the Pitch and Demo to the partner, please fill out the nomination at go/agentspace-ttp. Account team and customer/ partner will be informed via email from TTP about the acceptance into the program with details on the 0/30/60/90 days promotion period based on the seller and their manager approval.

If you want to allow list an implementation partner (GSI, RSI) to get them familiar with the product prior to introducing it to their own clients (sell-with motion), they can self-onboard using the published Agentspace accelerator, pending approval which you can request by reaching out to Sergio Villani.

To date, this list includes details on partners that have completed or planned to complete their self-onboarding.


Q: Where can I check the status of my partner that was submitted for allowlist?
Please check the status of your nomination at go/spark-ttp-status. 

Q: Can we fast track a standard customer (non-partner) for allowlist if they agree to self-onboard, they are very technical?
At this point to ensure a positive customer experience we are not opening up Agentspace for self-service customer onboarding unless you are a priority GSI/SI.

Q: Who is funding customer onboardings and what is the expectation for onboardings going forward?

We currently rely on partner led onboardings and PSO led onboardings, where funding will be provided via canonical partner funding programs (BIF, PSF etc.). 

Reseller Ordering:  

During the Allowlist GA phase (current phase) the reseller will need to work with the Google Cloud Field to create a Vector Opportunity and basically follow the same process as used for Direct deals except that the Partner Discount will be listed as a discount, in addition to any customer specific discounts (if applicable).  The contract will be created by CPQ.  

Once CPQ is live sellers can follow this CPQ process to create quotes, contracts and close deals:  CPQ Flow-Agentspace.

Consult with DSS and DPO for all Deal Related Questions.  They are reached by requesting Deal Assistance in the Vector / CPQ process.  

Product Documentation

[External][CustomerName]Spark Setup Instructions
Connector Docs: go/es-connectors
CUSTOM CONNECTOR FOR AGENTSPACE 
Note: Documentation will be launched on the website with the product launch. 
Need help? 
Q: What if I encounter product issues? 

Please use these links: 

Connectors: 
go/Agentspace-connector-doc-issue for reporting documentation related issues/requests go/Agentspace-connector-issue for reporting Connector issues
go/Agentspace-connector-feature-request for a new connector not included in go/es-connectors
go/spark-connector-doc-issue for reporting documentation related issues/request
Search Quality: go/spark-quality-issue
Feedback about NotebookLM Enterprise: go/nblm-product-feedback
Feature Requests: 
go/spark-search-fr
go/spark-assistant-fr
go/notebooklmplus-fr

Q: If I have questions about Agentspace, where can I get them addressed? 

Please visit Vertex Search Office hours held weekly on Fridays at 9:00 am PST. 
Weekly Search E2E Applications + Vertex AI Agent Builder Office Hours. We will schedule Office hours for Agentspace as well starting Jan ‘25. 

Q: Can I get a specialist / technical help to get started with Agentspace? 

If you need technical / onboarding assistance for Agentspace, please fill this http://goto.google.com/vais-onboarding-intake


Q: Can I get PM help to get started with Agentspace? 

Please email on this Agentspace_pmhelp@google.com if you need PM help for pitch/ customer meeting.

Q: How can I share customer feedback for Agentspace with the PM/ Eng teams? 
We actively seek feedback from account teams in the TTP program here. But please share your feedback about the product/ onboarding process etc even if your account is not in the TTP. 

Q: How do I make sure that bugs are raised with the information required by Engineering for triage/resolution

Work with your partners to capture the following:
Include Project Name, ID and # on all bugs
For any datastore issues, include the the data store ID, ie 'Collection ID' , location (global, us, eu) and tenant project
For any search result issues, include the data store ID, ie 'Collection ID', location (global, us. eu) and tenant project, and search token, ie 'attiributionToken'
For any generated answer issues, include the data store ID, ie 'Collection ID', location (global, us. eu) and tenant project, and answer token, ie 'uToken'
Collect all relevant screenshots and include in the bug

This link provides information on how to  get the Attribution and Assistant tokens that helps Engineering reproduce issues and resolve




Additional Questions

Questions to ask your customers on their Security and Compliance setup

Q: Are there any 
Data Residency requirements?
     (Region for your data to be processed (global, EU, US))

Q: What Identity Provider does the customer use? 
      (Active Directory Entra ID, LDAP, OKTA or other systems.)

Q: How do they currently manage user identities and access controls? Are they using any other systems?

Q: How important is access controlled search? Is content available to all users?

Q: Do you manage all users and user groups in your Identity Provider (IDP)?

Q: Are all of your SaaS instances and home-grown applications leveraging your IDP-defined user groups?









QUESTIONS FROM THE FIELD





<< Please add additional questions you hear from your customers here and we will answer them for you>>

<ldap> <Question> 

May 9, 2025
We are responding to a comprehensive RFI and have a number of ourstanding questions we’d like guidance on; 

For agentspace and agent engine; What are your SLAs for latency and throughput, and how are they measured and enforced?

Does either agentspace or ADK agents educate itself and improve, will there be agent created agents?


Esther Arribas García ACLs granularity question

Does Agentspace handle granular permissions like who can perform CRUD operations on Databases or BQ at column or row level? I mean things like: "being able to read entity 'employee' inside HR area, limited to the hotel where the user works and the organization to which the user belongs."






 



What is the resell motion for MSP/Resellers? Specifically will Agentspace fall under the current Google Cloud product schedule or will an addendum be needed? If an addendum is needed, how can I get access to a copy early? How will this be transacted - online through the partner sales console or offline (PST)? Will partner discount follow the Google Cloud Partner Advantage guide ex 12% for premier partners? Not PST;  follows same process as Direct but the template is created with the reseller terms. The contract will be created by CPQ once CPQ is live on 2/24/2025.  
During the Allowlist GA phase (current phase) the reseller will need to work with the Google Cloud Field to create a Vector Opportunity and basically follow the same process as used for Direct deals except that the Partner Discount will be listed as a discount, in addition to any customer specific discounts (if applicable).  The contract will be created by CPQ once CPQ is live on 2/24/2025.  

Once CPQ is live (2/24/2025) sellers can follow this CPQ process to create quotes, contracts and close deals:  CPQ Flow-Agentspace.

Consult with DSS and DPO for all Deal Related Questions.  They are reached by requesting Deal Assistance in the Vector / CPQ process.  


Resellers:  

Reseller Enablement:  

[input from Rachel Lunt ] 
Reseller Materials:   

[input from Rachel Lunt ] 




@ HSBC is asking for an overview of how we address the following situations:

1. Responsible AI controls
a. Offensive language protection
b. Users introducing biases into the LLM responses
c. Handling of PII in prompts
2. How to implement preventative, detective and corrective controls to monitor effectiveness of the solution in place for the concerns above and to test these controls at audit time

We have shared that 1.a will be handled as part of Responsible AI (RAI) controls built in Gemini. 1.b We don’t use user prompts for training so biases will not be introduced in the LLM. c is on the roadmap.  

Questions: 
1. Can we use security filters present in gemini in Spark 
2. How do we handle PII in prompts
3. Is there a security architecture for answer gen api which addresses RAI concerns  
4. Details for question 2 on audit compliance


Dambo Ren - We have a customers currently in allowlist of Agentspace, they are excited about the capabilities allowing customer create their own agent business workflow, and they are keen to find out if we will provide a private marketplace for org to publish/share/collaborate with the different agents, which is a common requirement for many companies based on our engagement across the region, I once saw the slide deck about it in preview version of Spark, however now it was gone with Agentspace, would like to hear your feedback & comments, thanks! 




Ken Zhang- Please share the configuration doc with steps to secure the setup and security features (APIs, buttons, sliders etc.) available in Agentspace. Besides, please also share the NDA only doc or slides on 1) how the data & sync work between agentspace and the connected apps/products; 2) the API flow and mechanism how Agentspace honours the connected SaaS/Prodcuts’ own ACLs; 3) how do customer protect, update and remove the synced data and ACL

Is AgentSpace available to internal Google Teams?

Not deployed as a tool however to test, yes u can get your pwn project allowlisted. Pls refer to go/agentspace-GTM for bug
There is also dogfood version in go/spark-dogfood

Questions 1/17 (skchennuri@) for customer Sabre
Nested Folders in cloud storage bucket: Does AgentSpace support nested folders without metadata.json files? If not, is there a workaround for organizing files beyond a single directory in cloud storage bucket?
Model Customization: How can I change the underlying AI model used by the Agent Assist app within AgentSpace?
Playbooks & Tooling: Does AgentSpace support features like Playbooks, function calling, and tool utilization for building more complex agent workflows?
ServiceNow Resource Modification: Can AgentSpace modify resources within ServiceNow (SNOW)? Ex: editing fields of SNOW incident, if yes, when is this feature expected?
Workday extend Licensing: Do customers need Workday extend licenses (e.g., for publishing APIs) to use AgentSpace's Workday integration features?
Demos: Customers wants to see demos on Workday, SNOW and Salesforce integration, do we have any, They'd also like to see how multiple agents can collaborate and work together within the AgentSpace platform. 
Pricing: pricing is pmpu based, so it does not matter how many agentspace apps are being used by a user, right? 
Agentspace is providing high accuracy compared to agent builder, my understanding is Agent builder uses RAG and AS uses KG, is there any plan to use KG in AB?
can we have one end point for all languages instead of multiple end points for each language?
Agents Quesitons 1/17 (blairgosselin@)
Will their be the ability to create no-code Agents (similar to GEMS in Gemini for Workspace)
Will they be publishable/sharable to other people within the org. 
RK> Low Code/ NoCode Framework to be launched sometime around Feb, at which point their will be an updated list of OOTB Agents 
Will we support the Vertex Agents API to “Code” agents
	RK> Priorities, Low Code (Playbooks) available now; No Code (new framework) available Q1; High 
Code (Vertex Agents API) available H2
Looking for more details around the Agentspace API. USE CASE: Workday would like to pull in valuable information in the workday hub. Image a links or a side panel to the relevant content in Drive or in Sharepoint - a widget with displays this info and takes the user off the WD Hub to the relevant page. 
APIs are available 

Agents Quesitons 1/17 (stuartgano@)

Are there use cases where someone would expose an agentspace app externally to their customers? 

	Agentspace is an internal use case wherein customer will buy licenses for each employee. 


What is the OEM model for agentspace apps for resellers or VARS 
How does networking work in this model? Where does the GCP project live. How does the billing work? 

Reselling options have been launched. 

Can I create notebook LM Enterprise Apps for a specific use case and is their a s
separate pricing model for this? 
IE. My customer wants an app that does only this one thing and Notebook LM provides that functionality.  Like a healthcare patient search app for clinicians 
Broadly are we only selling this as an enterprise knowledge management platform or can it be a point solution too. And is pricing different? 


Are agentspace Actions the differentiator here? That only this tier is allowed to call the calendar api? To create a new event? What is stopping me from doing this with webhook fulfillment in dialogflow? 
Is this Api Documented in detail somewhere? 


Agentspace Differentiations over other tiers are 
Access to 3p data sources 
Actions on 1p and 3p data sources
Agentic framework (coming soon)
Prebuilt Agents (starting with Research Agent) 

curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -H "X-Goog-User-Project: 1234" \
"https://discoveryengine.googleapis.com/v1alpha/projects/1234/locations/global/collections/default_collection/engines?engineId=myapp" \
-d '{
  "name": "projects/1234/locations/global/collections/default_collection/engines/myapp",
  "displayName": "myapp",
  "solutionType": "SOLUTION_TYPE_SEARCH",
  "searchEngineConfig": {
    "required_subscription_tier": "SUBSCRIPTION_TIER_SEARCH_AND_ASSISTANT",
    "searchTier": "SEARCH_TIER_ENTERPRISE",
    "searchAddOns": ["SEARCH_ADD_ON_LLM"]
  },
  "commonConfig": {
    "companyName": "My Company"
  },
  "industryVertical": "GENERIC",
  "dataStoreIds": ["mydatastore1", "mydatastore2"]
}'
------------------
{
  "name": "projects/1234/locations/global/collections/default_collection/operations/create-engine-877224357326871179",
  "done": true,
  "response": {
    "@type": "type.googleapis.com/google.cloud.discoveryengine.v1alpha.Engine",
    "name": "projects/1234/locations/global/collections/default_collection/engines/myapp",
    "displayName": "myapp",
    "dataStoreIds": [
      "mydatastore1",
      "mydatastore2"
    ],
    "solutionType": "SOLUTION_TYPE_SEARCH",
    "searchEngineConfig": {
      "searchTier": "SEARCH_TIER_ENTERPRISE",
      "searchAddOns": [
        "SEARCH_ADD_ON_LLM"
      ]
    },
    "commonConfig": {
      "companyName": "My Company"
    },
    "industryVertical": "GENERIC"
  }
}








Can we request Hyland OnBase as a datasource integration?  Once approved, how long would it take for the connector to be available?
My customer is very interested by AgentSpace and is starting a POC in one entity. However my customer is a very large worldwide group, each entity is expecting to get a tailored configuration especially for dedicated agents while the connectors will target global datasources. Is it a setup that would be possible with the future agents part in AgentSpace ?
What outputs can a “bring your own agent” render (markup tables, images, html) for example DFCX support rich responses components and agent kit supports buttons for user selection…
When bringing your own embeddings, how does agent space call the right embedding model for the user queries when using the datastore with the custom embeddings? is there a way to tell agent space or the datastore engine  to use a custom model - endpoint/model garden/anything else… 
What options do we have for customizing AgentSpace UI?
How can users disable/enable 1P agents (deep research is the only one for now) for a specific agent space instance?
Reshma Kapadia: While discussion with customer one of the question came in , if we build our own agent using agent builder & import it in agent space, how orchastration will happen. For ex: In insurance demo if he ask question show me Claim X & I have built a agent to validate documents, will it automatically trigger that agent with required parameter or it has to be done manually via @ command as it is shown in demo.


1. What's the document recall strategy in Agentspace?
2. What's the ratio between vector search and full-text search?
3. Do we have ranking model in Agentspace? If we have, what model is used?
4. How is the preprocessing of query text implemented in Agentspace?Or the overages apply after entire 50GiB * Number of Users is reached, so say if the subscription is for 1000 users then overages will apply only after 50,000 GiB?


Question 40. a) Agentspace has an announcements section as per UI. Can we use sharepoint news as well as custom data stores as source for this announcements section? 


In the announcements section, you can add data 


b) Additionally, Can we use dogfood to demo to customers?
We do not recommend doing a demo using dogfood since you sign in using google credentials and you might inadvertently display personal data 




Reshma Kapadia Payam Mousavi Rohini Goyal Stuart Moncada Sri Lingamneni


Question 41: Can you point me to documentation on how to create an Agent that displays a form? Not launched yet, will add details close to the launch


Should i assume that agentspace follows google chat card v2 for interface?Neil Sugden My Partner has asked:
Can they access the Sales Playbook & Business Case Calculator? And will they or a partner version be on  Partner Advantage (like these other resources)? Many thanks https://www.partneradvantage.goog/GCPPRM/s/global-search/agentspace?tabset-8713e=1


Neil Sugden My partner has asked is there any research that shows why Sharepoint is being highlighted as the main sales target market or is this just anecdotal feedback from the field? In addition they have asked what would be the 3 connectors they should position first.
Answer given below: Google Cloud Consulting/Professional Services has seen these connectors requested and installed during their work so far. So the top three connectors (by volume deployed) are: Sharepoint, Jira, Slack.



.


Rachel Lam Hi team, I have couple of questions while working with a customer:
1- Do we need Agentspace licenses for every employee in the organizations, or can the customer choose which user they want to purchase the license to? 
2- How do we move forward for Startup customers that need a fairly small amount of licenses (less than 10 users), likely a partner would even do this. 




Sanskriti Pattanayak 1) How does multiple versions/ historical version for a certain document(eg. on Google docs) get handled? Would we be able to query and search information on older versions (eg. use case being contract documents and understanding historical contract terms)
2) Can we restrict access to some sensitive information which the end user/ employee has access to as per ACLs but we do not want the same to get indexed/ingested by Agentspace and to be surfaced in any agentspace results.
3) Can we manage per user budgeting wrt to how many queries they can perform on Agentspace?
4) Successful Customer Usecases
5) Can the enterprise wide org setting for agent restriction be made such that certain users have access to only certain agents and data sources irrespective of if they have ACL level access to the data otherwise
6) Customer query: Best practice on how to structure employee productivity so that we can define success metrics and make a case for this platform -how are customers/partners measuring success with agentspace.
7) How does the data sync in the background while adhering to ACLs. 






Nora Filali 1) Do we have more information on when Sensitive Data Protection API / Model Armor will be integrated to AgentSpace, and we can get customers allowlisted? Customer would like to be able to scan uploaded attachements for PII/PHI during a session. https://sites.google.com/corp/google.com/agentspace/home#h.3f4tphhd9pn8 this mentions dogfood in (May/June), but was looking into more details


2) Within the same agentspace app for a customer, can a group of user be only be able to see agent A while another group can only see Agent B? I know ACLs can be configured for data, but what about the agents?


3) When integrating an ADK agent into agentspace, can the ADK agent have access to the new documents uploaded by the user during a session through the AgentSpace UI?
4) In the reporting features, is there a plan / a way to download a pdf report of the dashboard? Customer would like to be able to export the reports in multiple formats, including PDF, Excel, and CSV.
5) In the reporting features, is it possible to get metrics by groups of users? Customer wants this feature to be able to derive usage statistics by department 
6) Can the application scale up to 1000 concurrent simultaneous users without performance degradation?
7) Is there a plan to send notifications to admins when feedback is received
8) Customer wants to upload xlsx documents into gcs, so that they are consumed by agentspace. What is the best practice for ingesting it? Layout parser in agentspace when uploading unstructured data doesn’t seem to support xlsx. 


Samar Bhat
Does Agentspace license cost count towards computing the variable support fee since the product is covered as part of GCP Premium Support?




Neil Sugden
Question from partner on Agentspace and NotebookLM:
NotebookLM Enterprise doesn't appear to be part of Google Vault or any kind of enterprise security led search tool (i.e. what if there's a security hold on an employee, how can an employer get info back on their use of the tool?). 
Concern on how to sell it if this is not available. There appear to be a number of Buganizer tickets suggesting similar for Starling Bank, Atom bank, Revolut NotebookLM - Integrate NotebookLM into Vault [Cloud Blocker] [Kpmg LLP] Legal Hold and eDiscovery for Agentspace and NotebookLM


Magdalena Platter Question from my customer, a Private Bank:
My customer wants to know if the Google Search grounding within Agentspace is using the Web Grounding for Enterprise behind the scene or Grounding with Google Search. This is important information for them to have as per our Service Terms (Section 19.k GCP Service Specific Terms, Service Terms) we store prompts, contextual information that Customer may provide, and Generated Output for thirty (30) days as they can be used for debugging and testing. Is there any additional information available on how Google Search grounding is used within Agentspace? My customer is very concerned about their data like prompts potentially even CID data being persisted and being logged. Thanks






I have a partner implementing a Agentspace to a customer and they do have a blocker, quoting:


1. We have agentspace application with one connector and one agent (as on diagram).
2. we want to use data from connector inside agent. 
3. Agent created with agent designer (no-code) can use data from connectors by default. 
4. But we have to create custom full-code agent (Vertex AI) and register it to the application. 
5. What should we do to enable similar behaviour for custom agent - be able to use data from connector?

