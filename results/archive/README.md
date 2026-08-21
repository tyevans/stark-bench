# Superseded reports

Reports here were measured against a configuration that no longer exists.
They are kept because they record something no current run can: what the
harness scored *before* a defect was fixed.

`scripts/results_table.py` does not read this directory.

## `nomic-no-prefixes/`

`redstring-native` scored against `nomic-embed-text-v1.5` with **no task
prefixes seated** -- the defect described in CLAUDE.md's prefix section.
nomic is asymmetric and wants `search_document: ` on corpus text and
`search_query: ` on queries; neither was applied.

    dense    mrr 0.1948  hit@1 0.1321  hit@5 0.2643  recall@20 0.3111
    hybrid   mrr 0.2196  hit@1 0.1429  hit@5 0.3000  recall@20 0.3151

Not comparable to anything current: different model, different dimension
(768 vs 2048), different chunk table, and a pre-wipe corpus. Its only use is
as the "before" half of a before-and-after on prefixes, and even that is
weak, because the model changed at the same time. **Do not put these numbers
in a table beside a Nemotron row.**

The reason they are worth keeping at all: 0.1948 and 0.2196 are entirely
plausible numbers. Nothing about them looks broken. That is the whole point
of the prefix warning -- an unprefixed run does not fail, it just quietly
underperforms, and these are what that looks like.
