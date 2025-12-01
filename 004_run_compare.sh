#!/usr/bin/env bash

### --- 0) ENV & Basics ---
# sinnvolle ENV-Variablen
if [ -f .env ]; then
  source .env
else
  source default.env
fi

# Logs
logdir="$BASE_LOG_DIR/run_comparisons$(date +%F_%H%M)"
mkdir -p "$logdir"

echo "[Info] Starte Vergleichs-Run. Logs: $logdir"

for model_name in "Qwen" "jina"; do
  cd "$CODE_DIR"

  echo "[Info] Starte Methodenvergleich fuer ${model_name}..."
  poetry run python method_comparison.py \
    --model_name "${model_name}" \
    --run_test False \
    2>&1 | tee "$logdir/method_comp_${model_name}.log"

  echo "[Info] Modell ${model_name} fertig."
done

echo "[Info] Run komplett fertig."
