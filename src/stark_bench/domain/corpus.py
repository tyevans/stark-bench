"""The identity of a vector space, and what may share one.

## Why this is a type and not a naming convention

Two corpora are interchangeable when the same text embeds to the same
vector. That is a claim about the *model*, and this project has been wrong
about it three times in one day, each time silently:

  - the endpoint was swapped from one model to another while continuing to
    advertise the old id, so vectors from two models would have landed in a
    table named for one of them. Only a dimension mismatch caught it, and a
    dimension check is not a model check -- at 768 or 1024, which is most
    models, it fails open.
  - a corpus was embedded without the task prefix its model requires. The
    vectors were well-formed, clustered sensibly, and scored plausibly. The
    only symptom was a retrieval number quietly below what the model can
    do, which reads as "this model is mediocre".
  - a table was renamed after an ingest, orphaning 129,375 rows. Every
    query then returned empty, retrieval reported success, and the first
    sign of trouble was `min() arg is an empty sequence` three layers away.

Each of those is the same defect: identity was a convention, so nothing
could compare it. Here it is a frozen value, and equality means exactly
"these vectors may be compared".

## The prefix is part of the identity

`nomic-embed-text-v1.5` wants `search_document: ` and `search_query: `;
`Nemotron-3-Embed-1B` wants `passage: ` and `query: `; ada-002 wants
neither. Text embedded behind a prefix and the same text embedded bare land
in different regions of the space and cosine between them is meaningless --
so a prefix change is a model change for every purpose that matters here.
redstring ADR 0043 says so, extending ADR 0002's "a different embedding
model means a new store".
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b

#: Joins the parts of the identity before hashing. A separator is required,
#: not decorative: without one, ("ab", "") and ("a", "b") concatenate to the
#: same string and two incomparable corpora share a table. NUL cannot occur
#: in a model id or a prefix, which is why it is the separator.
_SEPARATOR = "\x00"


@dataclass(frozen=True, slots=True)
class CorpusIdentity:
    """Everything that decides whether two sets of vectors are comparable.

    Frozen and hashable, so it can be a dict key and cannot drift after a
    store has been opened against it.
    """

    model: str
    dimension: int
    document_prefix: str = ""
    query_prefix: str = ""

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty -- it is half the identity")
        if self.dimension <= 0:
            raise ValueError(f"dimension must be positive, got {self.dimension}")

    @property
    def is_prefixed(self) -> bool:
        """Whether this model asks for task prefixes at all.

        Empty on both sides is a legitimate configuration -- ada-002 is
        symmetric -- and is *also* what a forgotten prefix looks like. The
        two are indistinguishable from here, which is why the CLI logs what
        it seated rather than leaving it to be inferred.
        """
        return bool(self.document_prefix or self.query_prefix)

    def table_name(self, prefix: str = "kg_chunks") -> str:
        """A Postgres identifier that no other identity can collide with.

        Lowercased with non-alphanumerics collapsed, because a model id is a
        vendor's string and a table name is not: `Nemotron-3-Embed-1B` has
        capitals, and redstring's chunk store rejects anything that is not a
        bare lowercase identifier.

        The digest is appended **only when a prefix is set**, so unprefixed
        tables keep the names their existing rows were written against.
        Renaming those would orphan every row ingested before the prefix
        work landed, which is precisely the failure listed in the module
        docstring -- the fix must not re-commit the bug it was fixing.
        """
        slug = "".join(c if c.isalnum() else "_" for c in self.model).lower()
        if not self.is_prefixed:
            return f"{prefix}_{slug}"
        return f"{prefix}_{slug}_{self.digest()}"

    def digest(self, size: int = 4) -> str:
        """A short stable hash of the whole identity.

        Stable across processes and releases -- blake2b, not `hash()`, whose
        string seed is randomised per interpreter. A digest that moved
        between runs would orphan the previous ingest, which is the same
        failure by a different route.
        """
        # Dimension is deliberately NOT in the digest, though it IS in
        # equality. Adding it changed the Nemotron table from
        # `..._d38d8f8b` to `..._691131b0` while a two-hour ingest was
        # writing to the first -- landing that would have orphaned the whole
        # corpus, which is the failure this module exists to prevent, and it
        # was caught by printing the name rather than by reasoning.
        #
        # Nothing is lost: dimension is a property of the model, so two
        # identities differing only in dimension cannot both be real, and
        # redstring's chunk store fixes width at DDL time and rejects a
        # mismatched vector on the first insert anyway.
        joined = _SEPARATOR.join((self.model, self.document_prefix, self.query_prefix))
        return blake2b(joined.encode("utf-8"), digest_size=size).hexdigest()

    def comparable_with(self, other: CorpusIdentity) -> bool:
        """Whether vectors from these two corpora may be compared at all.

        Spelled out rather than left to `==` so that call sites read as the
        question they are actually asking. Cosine between vectors from two
        different identities is a number, and it means nothing.
        """
        return self == other
