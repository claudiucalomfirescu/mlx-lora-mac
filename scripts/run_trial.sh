#!/bin/zsh
# One full trial for one config, one memory-guarded job at a time:
# template check -> data -> 4-step smoke -> training -> base eval -> adapter eval -> server check.
# Usage (from the repo root, venv active): scripts/run_trial.sh configs/recipe.yaml
set -e
CFG=$1
[ -n "$CFG" ] || { echo "usage: scripts/run_trial.sh configs/<name>.yaml"; exit 2; }
PY=${PYTHON:-python}
NAME=$($PY -c "import sys; sys.path.insert(0,'scripts'); from common import load_cfg; print(load_cfg('$CFG')['name'])")
G() { n=$1; shift; $PY scripts/run_guarded.py --name ${NAME}_$n -- $PY "$@"; }
$PY scripts/check_template.py $CFG
G data      scripts/make_data.py $CFG
$PY scripts/check_rows.py $CFG
G smoke     scripts/train.py $CFG --smoke
G train     scripts/train.py $CFG
RUN=$(ls -d outputs/runs/*_${NAME} | tail -1)
G eval_base  scripts/evaluate.py $CFG --tag ${NAME}_base
G eval_tuned scripts/evaluate.py $CFG --tag ${NAME}_tuned --adapter $RUN/adapters
G serve      scripts/server_check.py $CFG --adapter $RUN/adapters --ref ${NAME}_tuned --base ${NAME}_base
$PY scripts/summarize.py ${NAME}
