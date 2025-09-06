import re
# Helper to extract JSON from LLM response
def extract_json_from_response(text):
    # Try to find a code block with json
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    # Try to find the first {...}
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1)
    raise ValueError("No JSON object found in response")
import os
import json
import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
from signal_monitor import search_web_for_signals
from lead_tracker import get_prior_leads, store_new_leads

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load secrets; in GCP, use Secret Manager
if 'GOOGLE_CLOUD_PROJECT' in os.environ:  # Detect if in Cloud Functions
    from google.cloud import secretmanager
    client = secretmanager.SecretManagerServiceClient()
    project_id = os.getenv("GCP_PROJECT_ID")
    secret_name = f"projects/{project_id}/secrets/OPENAI_API_KEY/versions/latest"
    response = client.access_secret_version(name=secret_name)
    api_key = response.payload.data.decode("UTF-8")
else:
    api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o", api_key=api_key)

# Define pipeline configs
PIPELINES = [
    {
        "role": "fractional_cto",
        "search_query": "businesses hiring fractional CTO",
        "relevance_prompt_template": """Evaluate signal for fractional CTO hiring intent (score 0-1). If >0.7, extract JSON: {{"company": str, "profile": str, "contacts": [str], "context": str, "timestamp": float, "source_url": str, "status": str}}.
        Signal: {signal}""",
        "novelty_prompt_template": """For company {company}, compare new signal {new_signal} to priors {priors}. Output JSON: {{"novelty_score": float, "reason": str}}.""",
        "collection_name": "leads_fractional_cto"
    },
    {
        "role": "part_time_software_engineer",
        "search_query": "businesses hiring part time software engineers",
        "relevance_prompt_template": """Evaluate signal for part-time software engineer hiring intent (score 0-1). If >0.7, extract JSON: {{"company": str, "profile": str, "contacts": [str], "context": str, "timestamp": float, "source_url": str, "status": str}}.
        Signal: {signal}""",
        "novelty_prompt_template": """For company {company}, compare new signal {new_signal} to priors {priors}. Output JSON: {{"novelty_score": float, "reason": str}}.""",
        "collection_name": "leads_part_time_software_engineer"
    },
    {
        "role": "part_time_data_engineer",
        "search_query": "businesses hiring part time data engineers analytics big data",
        "relevance_prompt_template": """Evaluate signal for part-time data engineer hiring intent with focus on analytics, big data, etc. (score 0-1). If >0.7, extract JSON: {{"company": str, "profile": str, "contacts": [str], "context": str, "timestamp": float, "source_url": str, "status": str}}.
        Signal: {signal}""",
        "novelty_prompt_template": """For company {company}, compare new signal {new_signal} to priors {priors}. Output JSON: {{"novelty_score": float, "reason": str}}.""",
        "collection_name": "leads_part_time_data_engineer"
    }
]

# Define state for LangGraph
class AgentState(TypedDict):
    signal: Dict[str, Any]
    config: Dict[str, Any]
    leads: List[Dict[str, Any]]
    step: str  # Track relevance or novelty phase

# Define LangGraph nodes
def relevance_node(state: AgentState) -> AgentState:
    logger.debug(f"Processing signal: {state['signal']['content']}")
    prompt = ChatPromptTemplate.from_template(state["config"]["relevance_prompt_template"])
    response = llm([HumanMessage(content=prompt.format(signal=state["signal"]["content"]))])
    logger.debug(f"LLM response: {response.content}")
    try:
        json_str = extract_json_from_response(response.content)
        data = json.loads(json_str)
        logger.debug(f"Parsed JSON: {data}")
        if "company" in data and "context" in data:
            data["timestamp"] = state["signal"]["timestamp"]
            data["source_url"] = state["signal"]["source_url"]
            data["role"] = state["config"]["role"]
            data["status"] = data.get("status", "NEW")
            state["leads"].append(data)
            logger.debug(f"Lead appended: {data}")
            state["step"] = "novelty"
        else:
            logger.debug(f"Lead not appended: company={'company' in data}, context={'context' in data}")
            state["step"] = "done"
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error(f"Error processing LLM response: {e}")
        state["step"] = "done"
    return state

def novelty_node(state: AgentState) -> AgentState:
    if not state["leads"]:
        logger.debug("No leads to process in novelty_node")
        state["step"] = "done"
        return state
    
    lead = state["leads"][-1]
    company = lead.get("company")
    if not company:
        logger.debug("No company in lead, removing")
        state["leads"].pop()
        state["step"] = "done"
        return state

    prior_leads = get_prior_leads(state["config"]["collection_name"])
    priors = prior_leads.get(company, [])
    
    if not priors:
        logger.debug(f"No priors for {company}, proceeding to store")
        state["step"] = "store"
        return state
    
    logger.debug(f"Checking novelty for {company} with priors: {priors}")
    prompt = ChatPromptTemplate.from_template(state["config"]["novelty_prompt_template"])
    response = llm([HumanMessage(content=prompt.format(
        company=company,
        new_signal=state["signal"]["content"],
        priors=[p["context"] for p in priors]
    ))])
    try:
        json_str = extract_json_from_response(response.content)
        novelty_data = json.loads(json_str)
        logger.debug(f"Novelty response: {novelty_data}")
        if novelty_data.get("novelty_score", 0) <= 0.5:
            logger.debug("Lead is duplicate, removing")
            state["leads"].pop()
        state["step"] = "store"
    except (json.JSONDecodeError, ValueError):
        logger.error("Error parsing novelty response, removing lead")
        state["leads"].pop()
        state["step"] = "done"
    return state

def store_node(state: AgentState) -> AgentState:
    if state["leads"]:
        logger.debug(f"Storing leads: {state['leads']}")
        store_new_leads(state["leads"], state["config"]["collection_name"])
    state["step"] = "done"
    return state

# Create LangGraph workflow
def create_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("relevance", relevance_node)
    workflow.add_node("novelty", novelty_node)
    workflow.add_node("store", store_node)
    
    workflow.set_entry_point("relevance")
    workflow.add_edge("relevance", "novelty")
    workflow.add_edge("novelty", "store")
    workflow.add_edge("store", END)
    
    return workflow.compile()

def process_for_role(config) -> list:
    workflow = create_workflow()
    raw_signals = search_web_for_signals._run(config["search_query"])
    new_leads = []
    
    for signal in raw_signals:
        state = {"signal": signal, "config": config, "leads": [], "step": "relevance"}
        final_state = workflow.invoke(state)
        new_leads.extend(final_state["leads"])
    
    logger.debug(f"Processed role {config['role']}: {len(new_leads)} leads")
    return new_leads

def process_all_pipelines() -> list:
    all_leads = []
    for config in PIPELINES:
        leads = process_for_role(config)
        all_leads.extend(leads)
    logger.debug(f"Total leads across all pipelines: {len(all_leads)}")
    return all_leads