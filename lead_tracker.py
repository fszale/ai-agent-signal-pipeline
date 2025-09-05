import os
import hashlib
from google.cloud import firestore

# In local, use emulator if set; else real Firestore
project_id = os.getenv("GCP_PROJECT_ID")
db = firestore.Client(project=project_id)

def get_prior_leads(collection_name: str):
    collection = db.collection(collection_name)
    docs = collection.stream()
    priors = {}
    for doc in docs:
        company = doc.id
        priors[company] = doc.to_dict().get('signals', [])
    return priors

def store_new_leads(new_leads, collection_name: str):
    collection = db.collection(collection_name)
    for lead in new_leads:
        company = lead['company']
        doc_ref = collection.document(company)
        existing = doc_ref.get()
        if existing.exists:
            existing_data = existing.to_dict()
            signals = existing_data.get('signals', [])
        else:
            signals = []
        signals.append({
            'context': lead['context'],
            'timestamp': lead['timestamp'],
            'source_url': lead['source_url'],
            'status': lead['status'],  # Store status
            'hash': hashlib.sha256(lead['context'].encode()).hexdigest()
        })
        doc_ref.set({'signals': signals})