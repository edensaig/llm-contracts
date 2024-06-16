#! /bin/bash

output_dir=~/Documents/data/llm_contracts


if [ ! -d $output_dir ]; then
  echo "Output directory doesn't exist: $output_dir"
  echo "Please create it in order to proceed."
  exit 1
fi

cd $output_dir


# Chatbot Arena MT-Bench dataset
# Leaderboard: https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard
# Dataset URL taken from: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/download_mt_bench_pregenerated.py
curl -LO https://huggingface.co/spaces/lmsys/mt-bench/resolve/main/data/mt_bench/model_judgment/gpt-4_single.jsonl

# HumanEval and MBPP datasets
curl https://raw.githubusercontent.com/evalplus/evalplus.github.io/main/results.json > "$output_dir/heval_and_mbpp.json"

echo "Done!"