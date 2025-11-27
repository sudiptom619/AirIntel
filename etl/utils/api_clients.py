# etl/utils/api_clients.py
import requests
import time

class SimpleRequestClient:
    """
    Very small wrapper for GET/POST with retry and basic backoff.
    Use this for Overpass/OpenAQ calls early in the project.
    """
    def __init__(self, retries=3, backoff=1.0, timeout=30):
        self.retries = retries
        self.backoff = backoff
        self.timeout = timeout
        self.session = requests.Session()

    def get(self, url, params=None, headers=None):
        for attempt in range(1, self.retries + 1):
            try:
                r = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt == self.retries:
                    raise
                time.sleep(self.backoff * attempt)

    def post(self, url, data=None, headers=None):
        for attempt in range(1, self.retries + 1):
            try:
                r = self.session.post(url, data=data, headers=headers, timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt == self.retries:
                    raise
                time.sleep(self.backoff * attempt)
