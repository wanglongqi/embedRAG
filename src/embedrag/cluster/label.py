"""Cluster labeling: keyword labels by default, optional LLM summaries.

The keyword label is free (derived from c-TF-IDF keywords). When an
OpenAI-compatible chat endpoint is configured, each cluster's keywords +
representative texts are summarized into a short natural-language topic name.
"""

from __future__ import annotations

import json

from embedrag.cluster.types import ClusterInfo
from embedrag.logging_setup import get_logger

logger = get_logger(__name__)


def keyword_label(cluster: ClusterInfo, max_terms: int = 3) -> str:
    """A compact label from the cluster's top distinctive keywords."""
    terms = [k for k in cluster.keywords[:max_terms] if k]
    if terms:
        return " / ".join(terms)
    return f"cluster {cluster.cluster_id}"


def apply_keyword_labels(clusters: list[ClusterInfo]) -> None:
    """Fill in ``label`` for every cluster using keywords (in place)."""
    for c in clusters:
        if not c.label:
            c.label = keyword_label(c)


async def label_clusters_llm(
    clusters: list[ClusterInfo],
    chat_url: str,
    model: str = "",
    api_key: str = "",
    language: str = "auto",
    timeout_seconds: int = 60,
) -> None:
    """Generate a short topic name + summary per cluster via an LLM (in place).

    Falls back to keyword labels for any cluster the model fails to label.
    """
    import aiohttp

    apply_keyword_labels(clusters)  # ensure a fallback label exists
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for c in clusters:
            prompt = _build_prompt(c, language)
            payload: dict = {
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            }
            if model:
                payload["model"] = model
            try:
                async with session.post(chat_url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = _parse_label(content)
                if parsed.get("label"):
                    c.label = parsed["label"]
                if parsed.get("summary"):
                    c.summary = parsed["summary"]
            except Exception as exc:
                logger.warning("llm_label_failed", cluster_id=c.cluster_id, error=str(exc))


_SYSTEM_PROMPT = (
    "You name clusters of similar short texts. Reply ONLY with compact JSON "
    '{"label": "<=6 word topic name", "summary": "one sentence"}.'
)


def _build_prompt(cluster: ClusterInfo, language: str) -> str:
    examples = "\n".join(f"- {t[:200]}" for t in cluster.representative_texts[:5])
    kws = ", ".join(cluster.keywords[:10])
    lang_hint = ""
    if language and language != "auto":
        lang_hint = f" Respond in {language}."
    return (
        f"Cluster of {cluster.size} items.{lang_hint}\n"
        f"Keywords: {kws}\n"
        f"Representative examples:\n{examples}\n\n"
        "Give a short topic label and a one-sentence summary as JSON."
    )


def _parse_label(content: str) -> dict:
    content = content.strip()
    # strip markdown code fences if present
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:]
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            pass
    # fallback: treat the whole content as the label
    return {"label": content[:60]}
