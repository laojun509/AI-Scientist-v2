#!/bin/bash
cd /root/.openclaw/workspace/AI-Scientist-v2
source venv/bin/activate
export DEEPSEEK_API_KEY="sk-1d66962246ee43eea270c57f2df793a9"
# 跳过 Semantic Scholar 来避免限流
export SKIP_SEMANTIC_SCHOLAR=1

python ai_scientist/perform_ideation_temp_free.py \
  --workshop-file "ai_scientist/ideas/my_first_research.md" \
  --model deepseek-chat \
  --max-num-generations 3 \
  --num-reflections 2
