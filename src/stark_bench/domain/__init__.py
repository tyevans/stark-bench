"""Values with meaning in the benchmark, and no knowledge of how it runs.

Nothing here imports anything else from `stark_bench`, opens a socket, reads
a file, or knows that Postgres exists. That is what makes these the things
worth reasoning about: a rule expressed here is true of the benchmark, not
true of one code path through it.

The layer exists because of what went wrong without it. Every silent defect
this project has shipped lived in the gap between a concept we held in our
heads and a type the code could check:

  - a corpus embedded by one model landed in a table named for another,
    because "which corpus is this" was a string convention;
  - the same corpus was ingested with and without a task prefix, and the two
    were comparable only by accident of which table they reached;
  - a report outlived the config that produced it, and nothing could tell.

`CorpusIdentity` is the answer to all three: it is the *identity* of a
vector space, and two of them are interchangeable when and only when it is
equal.
"""

from stark_bench.domain.corpus import CorpusIdentity
from stark_bench.domain.cost import Cost, ToolCall
from stark_bench.domain.ingest import IngestOutcome
from stark_bench.domain.query import Query, Ranked

__all__ = ["CorpusIdentity", "Cost", "IngestOutcome", "Query", "Ranked", "ToolCall"]
