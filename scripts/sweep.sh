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
CONCURRENCY="${CONCURRENCY:-64}"
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
  for attempt in 1 2 3; do
    local extra=(--no-resume)
    [ "$attempt" -gt 1 ] && extra=()   # retries resume; see header
    echo "=== INGEST $cfg attempt=$attempt $(date -u +%H:%M:%S)"
    if run_cli "$LOG_DIR/$cfg.ingest.log" \
         --config "config/$cfg.yaml" --ingest --ingest-edges \
         "${extra[@]}" --embed-concurrency "$CONCURRENCY"; then
      break
    fi
    [ "$attempt" -eq 3 ] && { echo "=== ABORT: $cfg ingest failed 3x"; exit 1; }
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
