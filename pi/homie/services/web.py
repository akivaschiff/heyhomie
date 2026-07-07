"""Web services: Tavily search, URL fetch, and a Claude-backed recipe extractor.
These are the integration seams the recipe tool depends on; injected via ToolContext
so tools stay unit-testable without network."""

import json
import os
import re

import requests


def tavily_search(query: str, max_results: int = 3) -> list[dict]:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return []
    resp = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": api_key, "query": query, "max_results": max_results},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {"url": r.get("url"), "title": r.get("title"), "content": r.get("content", "")}
        for r in data.get("results", [])
    ]


def fetch_url(url: str, max_chars: int = 12000) -> str:
    resp = requests.get(url, timeout=20, headers={"User-Agent": "Homie/1.0"})
    resp.raise_for_status()
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", resp.text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars]


def make_recipe_extractor(anthropic_client, model: str):
    """Returns (source_url, raw_text) -> {name, ingredients, steps} via Claude."""

    def extract(source_url: str, raw_text: str) -> dict:
        prompt = (
            "Extract the recipe from this page text as JSON with keys "
            '"name" (string), "ingredients" (array of strings), "steps" (array of '
            "strings). Output only the JSON.\n\n" + raw_text
        )
        resp = anthropic_client.messages.create(
            model=model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        match = re.search(r"\{.*\}", text, flags=re.S)
        payload = match.group(0) if match else text
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return {"name": "", "ingredients": [], "steps": []}
        return {
            "name": data.get("name", ""),
            "ingredients": data.get("ingredients", []) or [],
            "steps": data.get("steps", []) or [],
        }

    return extract
