import requests
from langchain_core.tools import BaseTool
from typing import Any, Dict, List

class SearchWebForSignals(BaseTool):
    name: str = "search_web_for_signals"
    description: str = "Search the web for signals about businesses hiring for specific roles."

    def _run(self, query: str) -> List[Dict[str, Any]]:
        """Synchronous implementation of the search tool."""
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

search_web_for_signals = SearchWebForSignals()