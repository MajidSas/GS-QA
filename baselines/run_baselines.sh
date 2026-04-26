python baselines.py --model ministral-3:14b-cloud --parser-model qwen3-coder-next:cloud --baseline direct
python baselines.py --model ministral-3:14b-cloud --parser-model qwen3-coder-next:cloud --baseline shuffled
python baselines.py --model sonnet4.6 --parser-model qwen3-coder-next:cloud --baseline direct
python baselines.py --model gpt4o --parser-model qwen3-coder-next:cloud --baseline direct

python baselines.py --model ministral-3:14b-cloud --parser-model qwen3-coder-next:cloud --baseline text2sql
python baselines.py --model sonnet4.6 --parser-model qwen3-coder-next:cloud --baseline text2sql
python baselines.py --model gpt4o --parser-model qwen3-coder-next:cloud --baseline text2sql

python baselines.py --model ministral-3:14b-cloud --parser-model qwen3-coder-next:cloud --embeddings nomic --baseline rag
python baselines.py --model sonnet4.6 --parser-model qwen3-coder-next:cloud --embeddings nomic --baseline rag
python baselines.py --model gpt4o --parser-model qwen3-coder-next:cloud --embeddings nomic --baseline rag


python summary.py --models "ministral-3:14b-cloud" sonnet4.6 gpt4o --rename ministral-3:14b-cloud_shuffled=shuffled

python clean_benchmark.py --verify-geo-wkts --models "ministral-3:14b-cloud" sonnet4.6 gpt4o --rename ministral-3:14b-cloud_shuffled=shuffled