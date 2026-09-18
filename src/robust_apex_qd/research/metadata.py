"""Publication identifiers for a provenance audit, independent of database row IDs."""

import hashlib
from typing import Any


def publication_keys(articles: list[dict[str, Any]]) -> set[str]:
    keys = set()
    for article in articles:
        pubmed = article.get("pubmed") or {}
        identifier = str(pubmed.get("pubmedId") or "").strip()
        if identifier and identifier.isdigit():
            keys.add(f"pubmed:{identifier}")
        elif article.get("title") and article.get("year"):
            title = " ".join(str(article["title"]).lower().split())
            key = hashlib.sha256(f"{article['year']}:{title}".encode()).hexdigest()
            keys.add(f"title_year:{key}")
    return keys
