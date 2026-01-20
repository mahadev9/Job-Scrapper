import os
from typing import List

from serpapi import GoogleSearch


def create_google_search(query, location="United States", next_page_token=None):
    params = {
        "engine": "google_jobs",
        "q": query,
        "location": location,
        "hl": "en",
        "no_cache": True,
        "api_key": os.environ["SERPAPI_KEY"],
    }

    if next_page_token:
        params["next_page_token"] = next_page_token

    return GoogleSearch(params)


def fetch_google_jobs(query, pages, location="United States") -> List:
    jobs = create_google_search(query, location)
    results = jobs.get_dict()

    all_jobs = [] + results.get("jobs_results", [])
    next_page_token = results.get("serpapi_pagination", {}).get("next_page_token")

    while next_page_token and pages > 0:
        pages -= 1
        jobs = create_google_search(query, location, next_page_token)
        results = jobs.get_dict()
        all_jobs += results.get("jobs_results", [])
        next_page_token = results.get("serpapi_pagination", {}).get("next_page_token")

    return all_jobs
