"""Smoke-test the prediction API."""

import json
import sys

import requests

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:9696/predict"

payload = {
    "hour": 10,
    "day_of_week": 0,
    "month": 3,
    "cash_type": "card",
}

response = requests.post(URL, json=payload, timeout=10)
print(response.status_code)
print(json.dumps(response.json(), indent=2))