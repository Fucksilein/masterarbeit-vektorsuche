#!/usr/bin/env bash

### --- 0) ENV & Basics ---
# sinnvolle ENV-Variablen
if [ -f .env ]; then
  source .env
else
  source default.env
fi

# Logs
logdir="$BASE_LOG_DIR/install_$(date +%F_%H%M)"
mkdir -p "$logdir"

echo "[Info] Starte Install-Setup. Logs: $logdir"

### --- 1) Systempakete & Tools ---
sudo apt-get update -y
sudo apt-get install -y \
  git tmux htop nvtop unzip \
  gcc g++ make \
  python3.12 python3.12-venv python3.12-distutils \
  pipx awscli

pipx install poetry || true
pipx ensurepath || true
export PATH="$HOME/.local/bin:$PATH"

echo "[Info] GPU / Treiber:"
nvidia-smi || echo "[Warnung] nvidia-smi fehlgeschlagen (Treiber prüfen)"

### --- 2) Poetry-Umgebung
poetry env use python3.12
poetry install

### --- 3) Torch/CUDA-Test & Systemreport ---
cd $CODE_DIR
echo "[Info] Teste torch.cuda.is_available() ..."
poetry run python -c "import torch; print('torch version:', torch.__version__); print('cuda available:', torch.cuda.is_available())" \
  2>&1 | tee "$logdir/01_torch_cuda_check.log"

echo "[Info] Erzeuge system_report.tex ..."
poetry run python system_report.py 2>&1 | tee "$logdir/02_system_report.log"
cd ..

### --- 4) ESCI-Daten laden & vorbereiten ---
echo "[Info] Downloade ESCI-Daten ..."
mkdir -p $CODE_DIR/shopping_queries_dataset
cd $CODE_DIR/shopping_queries_dataset

curl -L -o shopping_queries_dataset_examples.parquet \
  https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset/shopping_queries_dataset_examples.parquet

curl -L -o shopping_queries_dataset_products.parquet \
  https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset/shopping_queries_dataset_products.parquet

cd $BASE_DIR

### --- 5) Modelle vorab downloaden (Qwen + Jina) ---
echo "[Info] Lade Qwen/Qwen3-Embedding-0.6B ..."
poetry run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer.from_pretrained('Qwen/Qwen3-Embedding-0.6B')" \
  2>&1 | tee "$logdir/03_download_qwen.log" || echo "[Warnung] Download Qwen fehlgeschlagen"

echo "[Info] Lade jinaai/jina-embeddings-v2-small-en ..."
poetry run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer.from_pretrained('jinaai/jina-embeddings-v2-small-en')" \
  2>&1 | tee "$logdir/03_download_jina.log" || echo "[Warnung] Download Jina fehlgeschlagen"


echo "[Info] Install-Fertig. Logs in $logdir"
