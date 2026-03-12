import os
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP


API_BASE = "https://api.semanticscholar.org/graph/v1"
DEFAULT_FIELDS = [
    "title",
    "year",
    "abstract",
    "venue",
    "citationCount",
    "authors",
    "externalIds",
    "url",
]


def _headers() -> dict[str, str]:
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    headers: dict[str, str] = {"User-Agent": "appliedIntelligence-semantic-scholar-mcp/1.0"}
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _request(path: str, params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(
        f"{API_BASE}/{path}",
        params=params,
        headers=_headers(),
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def _normalize_paper(paper: dict[str, Any]) -> dict[str, Any]:
    authors = [author.get("name", "") for author in paper.get("authors", [])]
    return {
        "paper_id": paper.get("paperId"),
        "title": paper.get("title"),
        "year": paper.get("year"),
        "venue": paper.get("venue"),
        "citation_count": paper.get("citationCount"),
        "authors": authors,
        "url": paper.get("url"),
        "abstract": paper.get("abstract"),
        "external_ids": paper.get("externalIds", {}),
    }


mcp = FastMCP(
    name="semantic-scholar-mcp",
    instructions=(
        "Searches Semantic Scholar for real academic papers. "
        "Prefer this tool whenever literature claims need verification."
    ),
)


@mcp.tool(description="Search papers from Semantic Scholar with optional year filters.")
def search_papers(
    query: str,
    year_from: int = 2024,
    year_to: int = 2026,
    limit: int = 10,
) -> dict[str, Any]:
    params = {
        "query": query,
        "limit": max(1, min(limit, 25)),
        "fields": ",".join(DEFAULT_FIELDS),
        "year": f"{year_from}-{year_to}",
    }
    payload = _request("paper/search", params)
    papers = [_normalize_paper(item) for item in payload.get("data", [])]
    return {
        "query": query,
        "year_range": [year_from, year_to],
        "total": payload.get("total"),
        "returned": len(papers),
        "papers": papers,
    }


@mcp.tool(description="Fetch a paper by Semantic Scholar paper ID, DOI, Corpus ID, ArXiv ID, MAG ID, ACL ID, PMID, PMCID, or URL.")
def get_paper(identifier: str) -> dict[str, Any]:
    params = {"fields": ",".join(DEFAULT_FIELDS)}
    payload = _request(f"paper/{identifier}", params)
    return _normalize_paper(payload)


if __name__ == "__main__":
    mcp.run("stdio")
