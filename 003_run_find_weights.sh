#!/usr/bin/env bash

### --- 0) ENV & Basics ---
# sinnvolle ENV-Variablen
if [ -f .env ]; then
  source .env
else
  source default.env
fi

# Logs
logdir="$BASE_LOG_DIR/run_weights$(date +%F_%H%M)"
mkdir -p "$logdir"

echo "[Info] Starte Gewichtssuche Run. Logs: $logdir"

for model_name in "qwen" "jina"; do
  cd "$CODE_DIR"

  echo "[Info]Starte Gewichtssuche fuer ${model_name}..."
  poetry run python find_best_weights.py \
    --model_name "${model_name}" \
    --modes both \
    --methods 2 3 \
    --frac_queries 0.05 \
    --ndcg_k 10 \
    2>&1 | tee "$logdir/find_best_weights_${model_name}.log"
  echo "[Info] Modell ${model_name} fertig."
done

echo "[Info] Run komplett fertig."
