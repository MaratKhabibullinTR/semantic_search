## Dataset
https://www.kaggle.com/datasets/jeet2016/us-financial-news-articles

## Install
``` bash
poetry install
```

## Reindex
```bash
poetry run python -m server_cli reindex --input data/corpus --out data/index --model "all-MiniLM-L12-v2"
```

## Benchamrk - Index All
```bash
poetry run python -m server_cli index-all --config benchmark_config.yaml
```

## Benchmark - Query
```bash
poetry run python -m server_cli query --config benchmark_config.yaml --query "revenue growth"
```

## Benchmark - Report
```bash
poetry run python -m server_cli report --config benchmark_config.yaml --qrels qrels.jsonl
```

## Search
```bash
poetry run python -m server_cli search --index data/index --query "revenue growth" --k 8
```

## Build dataset
```bash
poetry run python -m server_cli convert-json-corpus-to-txt
```

## MCP stdio server
```bash
poetry run python -m server_mcp_stdio
```
Then hook it into your MCP host configuration. Tools: `reindex`, `search`.


## Streamlit App

The project includes a Streamlit web application for testing text splitters/sentencizers.

```bash
# Run the Streamlit app
poetry run streamlit run streamlit_app.py
```
