import os
import json
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from .signal_monitor import search_web_for_signals
from .lead_tracker import get_prior_leads, store_new_leads

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

llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

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

agent = initialize_agent(
    tools=[search_web_for_signals],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def process_for_role(config) -> list:
    relevance_prompt = PromptTemplate.from_template(config["relevance_prompt_template"])
    novelty_prompt = PromptTemplate.from_template(config["novelty_prompt_template"])
    
    raw_signals = search_web_for_signals(config["search_query"])
    new_leads = []
    prior_leads = get_prior_leads(config["collection_name"])

    for signal in raw_signals:
        relevance_response = agent.run(relevance_prompt.format(signal=signal['content']))
        try:
            data = json.loads(relevance_response)
        except json.JSONDecodeError:
            continue
        if data.get('score', 0) <= 0.7:
            continue

        data['timestamp'] = signal['timestamp']
        data['source_url'] = signal['source_url']
        data['role'] = config["role"]
        data['status'] = "NEW"  # Default status
        company = data.get('company')
        if not company:
            continue

        priors = prior_leads.get(company, [])
        if not priors:
            new_leads.append(data)
        else:
            novelty_response = agent.run(novelty_prompt.format(
                company=company,
                new_signal=signal['content'],
                priors=[p['context'] for p in priors]
            ))
            try:
                novelty_data = json.loads(novelty_response)
            except json.JSONDecodeError:
                continue
            if novelty_data.get('novelty_score', 0) > 0.5:
                new_leads.append(data)

    if new_leads:
        store_new_leads(new_leads, config["collection_name"])
    return new_leads

def process_all_pipelines() -> list:
    all_leads = []
    for config in PIPELINES:
        leads = process_for_role(config)
        all_leads.extend(leads)
    return all_leads