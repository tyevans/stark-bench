"""Every knob that changes a number, as a value.

The resolved contents travel in `raw` and are embedded verbatim in the
results file. A number whose config is not recorded is not re-runnable.

The *loader* lives in `adapters.config_file`. Splitting them is not
bookkeeping: reading a file and parsing YAML are I/O, and a domain that
imports `yaml` cannot be constructed in a test without one.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunConfig:
    name: str
    dataset: str
    split: str
    chunker: str
    embeddings: str
    dimension: int
    aggregation: str
    agent: str
    k: int
    raw: str
    #: The chat model an LLM agent talks to, as the server's own model id.
    #: Optional: `dense` and `hybrid` make no LLM call at all, so a config
    #: for either would have nothing to say here. `None` means the CLI's
    #: `DEFAULT_CHAT_MODEL`.
    chat_model: str | None = None
    #: Asymmetric-model task prefixes, seated on `LangChainEmbeddingProvider`.
    #: `nomic-embed-text-v1.5` wants `search_document: ` on stored text and
    #: `search_query: ` on a query; ada-002 wants neither, which is why both
    #: default to empty rather than to nomic's values.
    #:
    #: These are part of the store's identity, not decoration -- a corpus
    #: embedded with a prefix and the same corpus embedded without it are not
    #: comparable vectors (redstring ADR 0043). `_table_for` in the CLI
    #: derives the table name from them for that reason, so changing one here
    #: lands in a different table rather than mixing two vector spaces.
    document_prefix: str = ""
    query_prefix: str = ""
