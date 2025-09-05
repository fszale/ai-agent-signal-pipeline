import os
from dotenv import load_dotenv
from ai_agent import process_all_pipelines

load_dotenv()  # For local development

def run_pipeline(event=None, context=None):  # Cloud Function entry point
    all_leads = process_all_pipelines()
    return f"Pipeline complete: {len(all_leads)} leads saved"

if __name__ == "__main__":
    run_pipeline()