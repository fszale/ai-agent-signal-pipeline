import os
import hashlib
from google.cloud import firestore

# In local, use emulator if set; else real Firestore
project_id = os.getenv("GCP_PROJECT_ID")
dn_name = os.getenv("FIRESTORE_DB", "(default)")
db = firestore.Client(project=project_id,database=dn_name)

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
        
        # Check for duplicate signal based on hash
        new_signal_hash = hashlib.sha256(lead['context'].encode()).hexdigest()
        if any(signal['hash'] == new_signal_hash for signal in signals):
            continue  # Skip duplicate signal
        
        signals.append({
            'context': lead['context'],
            'timestamp': lead['timestamp'],
            'source_url': lead['source_url'],
            'status': lead['status'],
            'hash': new_signal_hash
        })
        doc_ref.set({'signals': signals})