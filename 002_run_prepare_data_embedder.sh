#!/usr/bin/env bash

### --- 0) ENV & Basics ---
# sinnvolle ENV-Variablen
if [ -f .env ]; then
  source .env
else
  source default.env
fi

# Logs
logdir="$BASE_LOG_DIR/run_embed$(date +%F_%H%M)"
mkdir -p "$logdir"


echo "[Info] Führe prepare_data.py aus ..."
poetry run python prepare_data.py 2>&1 | tee "$logdir/01_prepare_data.log"
echo "[Info] Datenvorbereitung abgeschlossen."

echo "[Info] Starte Run. Logs: $logdir"

### --- 1) Modelle nacheinander laufen lassen ---
for model_name in "qwen" "jina"; do
  cd "$CODE_DIR"
  echo "[Info] === Starte Embedding für Modell: ${model_name} ==="
  poetry run python embedder.py "${model_name}" \
    2>&1 | tee "$logdir/embed_${model_name}.log"

  echo "[Info] Embedding ${model_name} fertig."
done