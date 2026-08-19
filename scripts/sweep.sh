#!/usr/bin/env bash
#
# Every remaining number, in one server configuration.
#
# Three phases, ordered so the cheapest and most-wanted land first:
#   A  ingest all three arms, edges included (deep needs them to traverse)
#   B  dense + hybrid on each -- no LLM call
#   C  zero_shot + deep on each -- one to eight LLM calls per query
#
# ## Why this checks exit codes so insistently
#
# Its first version did not, and the shared embedding endpoint went down
# four minutes into a five-hour run. All three ingests died -- `connection
# refused`, then `503 Loading model` -- and the script moved straight on to
# SCORING, against three corpora holding ~16k chunks of an expected 535k.
# Twelve plausible-looking numbers, every one of them garbage, and nothing
# in the output would have said so.
#
# So: `set -euo pipefail`, an explicit check after every stage, and an
# assertion on the DATA (node count) rather than on the exit code alone --
# an ingest can exit 0 having embedded nothing.
#
# ## Why it retries
#
# The endpoint is shared and is restarted by hand sometimes. Ingest is
# resumable -- `skb/ingest.py` skips a node whose chunks are already stored
# -- so a retry after a blip costs the work already done and nothing more.
# The FIRST attempt passes --no-resume to guarantee a clean corpus; retries
# drop it, so they resume rather than start over. Losing 70 minutes to a
# five-second network hiccup is the failure this prevents.
set -euo pipefail

cd "$(dirname "$0")/.."
ARMS="native-wholedoc redstring-native native-sliding1k"
EXPECTED_NODES=129375
# The embedding peer died of `CUDA error: out of memory` four times: twice
# at client concurrency 64, then twice more at 16. **Concurrency was not the
# variable** -- backing off looked like it helped and did not. The card had
# 373 MiB free with a multimodal 27B at `--ctx-size 65536 -np 4` beside the
# embedder, and `alloc` failed on a transient buffer.
#
# Fixed on the server: a text-only 16k `-np 1` chat build freed 4.2 GB, so
# there is now 4577 MiB of headroom rather than 373. 32 rather than 64
# because the measured gain from 16 to 64 was 2.7% at -np 1 (1745 -> 1792
# nodes/min), which is not worth re-approaching an edge that has already
# cost four crashes.
CONCURRENCY="${CONCURRENCY:-4}"   # >= the server's -np; see ingest.py
# Texts per embedding request. This and CONCURRENCY together decide almost
# nothing: measured on 3000 nodes, every setting from 2-in-flight to
# 512-in-flight lands between 1312 and 1618 nodes/min. The endpoint is at a
# hardware ceiling around 1615 and the knobs move a 23% band around it.
#
# Recorded because the opposite was believed for an hour, on the strength of
# a standalone probe that used urllib without connection reuse and was
# therefore measuring TCP handshakes rather than the model.
EMBED_BATCH="${EMBED_BATCH:-128}"
# The peer is restarted by hand and has crashed on its own. Over a run this
# long, three attempts is optimistic.
MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"
# RESUME=1 makes even the FIRST attempt resume rather than starting the arm
# clean. Use it to continue an interrupted sweep: four relaunches in one
# session each discarded ~20k already-embedded chunks before this existed.
#
# Guarded, not trusted -- see scripts/resume_is_safe.py for why a changed
# chunker makes resuming actively wrong rather than merely stale.
RESUME="${RESUME:-0}"
LOG_DIR="${LOG_DIR:-/tmp/stark-sweep}"
mkdir -p "$LOG_DIR"

run_cli() {  # run_cli <logfile> <args...>
  local log="$1"; shift
  if ! uv run python -m stark_bench.harness.cli "$@" >"$log" 2>&1; then
    echo "!!! FAILED: cli $*"
    grep -v "HTTP Request" "$log" | tail -15
    return 1
  fi
  grep -v "HTTP Request" "$log" | tail -1
}

ingest_arm() {  # ingest_arm <config>
  local cfg="$1" attempt
  local first_extra=(--no-resume)
  if [ "$RESUME" = "1" ]; then
    if python3 scripts/resume_is_safe.py "$cfg"; then
      echo "=== RESUME $cfg: recorded config matches, continuing existing corpus"
      first_extra=()
    else
      echo "=== RESUME refused for $cfg: config differs from the recorded ingest,"
      echo "===   or none was recorded. Re-ingesting clean -- a changed chunker"
      echo "===   leaves stale chunk ids behind that still answer queries."
    fi
  fi
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    local extra=("${first_extra[@]}")
    [ "$attempt" -gt 1 ] && extra=()   # retries always resume; see header
    echo "=== INGEST $cfg attempt=$attempt $(date -u +%H:%M:%S)"
    if run_cli "$LOG_DIR/$cfg.ingest.log" \
         --config "config/$cfg.yaml" --ingest --ingest-edges \
         "${extra[@]}" --embed-concurrency "$CONCURRENCY" --embed-batch "$EMBED_BATCH"; then
      break
    fi
    [ "$attempt" -eq "$MAX_ATTEMPTS" ] && { echo "=== ABORT: $cfg ingest failed ${MAX_ATTEMPTS}x"; exit 1; }
    echo "=== retrying $cfg in 60s"; sleep 60
  done

  # The exit code is not the evidence. An ingest that embedded nothing can
  # still exit 0, and this project has shipped exactly that shape before.
  local nodes
  nodes=$(python3 -c "
import json,sys
print(json.load(open('results/$cfg.ingest.json')).get('nodes', 0))")
  if [ "$nodes" -ne "$EXPECTED_NODES" ]; then
    echo "=== ABORT: $cfg ingested $nodes nodes, expected $EXPECTED_NODES"
    exit 1
  fi
  echo "=== INGEST-OK $cfg nodes=$nodes"
}

for cfg in $ARMS; do ingest_arm "$cfg"; done
echo "=== PHASE-A DONE $(date -u +%H:%M:%S)"

score() {  # score <config> <agent>
  echo "=== RUN $1/$2 $(date -u +%H:%M:%S)"
  run_cli "$LOG_DIR/$1.$2.log" --config "config/$1.yaml" --run --agent "$2" \
    || { echo "=== $1/$2 failed; continuing to the next cell"; return 0; }
}

for agent in dense hybrid; do
  for cfg in $ARMS; do score "$cfg" "$agent"; done
done
echo "=== PHASE-B DONE $(date -u +%H:%M:%S)"
uv run python scripts/results_table.py || true

for agent in zero_shot deep; do
  for cfg in $ARMS; do score "$cfg" "$agent"; done
done
echo "=== SWEEP DONE $(date -u +%H:%M:%S)"
uv run python scripts/results_table.py || true
