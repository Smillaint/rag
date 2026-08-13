# -*- coding: utf-8 -*-
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

HYDE_MAX_TOKENS = 256
HYDE_TEMPERATURE = 0.7

SYSTEM_PROMPT = (
    "You are a technical document author. Given a question, write a concise "
    "passage (3-6 sentences, roughly 120-180 words) that directly answers it "
    "as if extracted from a reference document. Be specific about names, "
    "terms, and steps. Do not write introductions, conclusions, or "
    "meta-commentary. Output only the passage."
)

USER_TEMPLATE = "Question: {query}\n\nPassage:"


@dataclass
class HyDEResult:
    """Result of HyDE hypothetical document generation."""

    document: str | None
    used: bool
    elapsed_ms: float
    error: str | None = None


def generate_hypothetical_document(
    client: "OpenAI",
    model: str,
    query: str,
    max_tokens: int = HYDE_MAX_TOKENS,
    temperature: float = HYDE_TEMPERATURE,
) -> HyDEResult:
    """Generate a hypothetical answer document for HyDE vector retrieval.

    The generated document is used as the vector-search query so that its
    embedding lands closer to real answer chunks than the short question
    would. On any failure the document is None and callers fall back to the
    original query.
    """
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(query=query)},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        doc = (response.choices[0].message.content or "").strip()
        elapsed = (time.perf_counter() - started) * 1000
        if not doc:
            return HyDEResult(
                document=None, used=False, elapsed_ms=elapsed, error="empty_response"
            )
        logger.info("HyDE generated hypothetical document (%d chars)", len(doc))
        return HyDEResult(document=doc, used=True, elapsed_ms=elapsed)
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000
        logger.warning("HyDE generation failed, falling back to raw query: %s", exc)
        return HyDEResult(
            document=None,
            used=False,
            elapsed_ms=elapsed,
            error=f"{type(exc).__name__}: {exc}",
        )
