#!/usr/bin/env bash
# Run a sharded job to completion under the memory watchdog, restarting it when it asks for that.
#   caffeinate -is runner/run_sharded.sh runner/toy.yaml            # whole run, resumable; Ctrl-C or CONTROL=pause stops it
#   runner/run_sharded.sh configs/my_run.yaml --max-shards 3       # extra arguments go to shard_runner.py
# Settings by environment variable: PYTHON (default python), MIN_AVAIL_GB (default 12.5: the free memory a job needs at
# start, about its peak + 3.5 GB), BLOCK_ON (regex of other GPU jobs to wait for), MLX_PATTERN (replaces the watchdog's
# own pattern for other MLX jobs).
# One GPU job at a time: waits while the config's control_file says "pause". Every attempt gets its own guard record and
# log (outputs/logs/<config>_<time>.*), so a failed start is never overwritten. Exit codes of shard_runner.py:
#   0 done | 10 paused | 11 shard limit | 12 memory recycle | 20 canary mismatch | 30 config/design problem
cd "$(dirname "$0")/.." || exit 2
PY=${PYTHON:-python}
CFG=$1; [ -n "$CFG" ] || { echo "usage: runner/run_sharded.sh CONFIG [shard_runner args]"; exit 2; }
shift
NAME=$(basename "$CFG" .yaml)
CONTROL=$($PY -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1])).get('control_file') or '')" "$CFG")
STATE=$($PY -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1])).get('state_file') or '')" "$CFG")
mkdir -p outputs/logs
CHAIN=outputs/logs/${NAME}_chain.log
fails=0
while true; do
  while [ -n "$CONTROL" ] && grep -q '^pause' "$CONTROL" 2>/dev/null; do
    [ -n "$STATE" ] && echo "waiting for the GPU $(date +%FT%T)" > "$STATE"; sleep 60
  done
  ts=$(date +%Y%m%dT%H%M%S)
  $PY scripts/run_guarded.py --name "${NAME}_$ts" --min-avail-gb "${MIN_AVAIL_GB:-12.5}" \
      ${CONTROL:+--control-file "$CONTROL"} ${BLOCK_ON:+--block-on "$BLOCK_ON"} ${MLX_PATTERN:+--mlx-pattern "$MLX_PATTERN"} \
      -- $PY runner/shard_runner.py "$CFG" "$@"
  rc=$?
  echo "$(date +%FT%T) ${NAME}_$ts rc=$rc" >> "$CHAIN"
  [ $rc -ne 0 ] && tail -n 5 "outputs/logs/${NAME}_$ts.log" >> "$CHAIN" 2>/dev/null
  case $rc in
    0)        echo "complete"; break ;;
    10|11|12) fails=0; sleep 5 ;;              # paused, shard limit reached, or memory recycle: loop on
    3)        sleep 120 ;;                     # watchdog refused (not enough free memory): wait and try again
    20|30)    echo "stopped: rc=$rc, see outputs/logs/${NAME}_$ts.log and the run's CANARY.log"; break ;;
    *)        fails=$((fails + 1)); [ $fails -ge 6 ] && { echo "6 failures in a row: stopping" >> "$CHAIN"; break; }; sleep 300 ;;
  esac
  case " $* " in *" --list "*) break ;; esac
done
