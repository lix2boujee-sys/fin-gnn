#!/bin/bash
set -e
# Cloud GPU setup script for E5-Mistral experiments
# Run once after uploading the project to /root/fin-gnn

if [ -d "/root/fin-gnn" ]; then
  PROJECT_DIR="/root/fin-gnn"
elif [ -d "/root/autodl-tmp/fin-gnn" ]; then
  PROJECT_DIR="/root/autodl-tmp/fin-gnn"
else
  echo "Project directory not found. Expected /root/fin-gnn or /root/autodl-tmp/fin-gnn"
  exit 1
fi
cd "$PROJECT_DIR"

echo "============================================"
echo "  E5-Mistral Cloud GPU Setup"
echo "============================================"

echo ""
echo "=== 1. Install dependencies ==="
pip install -r requirements.txt -q

echo ""
echo "=== 2. Download MiniLM (~90MB) ==="
mkdir -p cache/models/all-MiniLM-L6-v2
python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
m.save('cache/models/all-MiniLM-L6-v2')
print('MiniLM OK')
"

echo ""
echo "=== 3. Download E5-Mistral (~14GB, may take 5-10 min) ==="
mkdir -p cache/models/e5-mistral-7b-instruct
python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('intfloat/e5-mistral-7b-instruct')
m.save('cache/models/e5-mistral-7b-instruct')
print('E5-Mistral OK')
"

echo ""
echo "=== 4. Fix Windows paths to Linux paths ==="
sed -i "s|D:/fin-gnn|$PROJECT_DIR|g" configs/*.yaml
echo "Paths fixed."

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
nvidia-smi
echo ""
echo "Ready: cd $PROJECT_DIR && tmux new -s e5_exp"
