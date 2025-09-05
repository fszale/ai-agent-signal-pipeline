import requests
from langchain.tools import tool

@tool
def search_web_for_signals(query: str) -> list:
    """Search the web for signals about businesses hiring for specific roles."""
    url = f"https://www.reddit.com/search.json?q={query}&sort=new"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    data = response.json()
    signals = [
        {
            "content": post['data']['title'] + ": " + post['data']['selftext'],
            "timestamp": post['data']['created_utc'],
            "source_url": post['data']['url']
        } for post in data['data']['children']
    ]
    return signals[:10]  # Limit for free tier