## Requirements
- poetry: https://python-poetry.org/docs/#installation
- python: >=3.11

## Install
``` bash
poetry install
```

## Dataset
https://www.kaggle.com/datasets/jeet2016/us-financial-news-articles

### Convert json dataset to text
```bash
poetry run python -m server_cli convert-json-corpus-to-txt
```

## Benchmark

### Index
```bash
poetry run python -m server_cli index-all --config benchmark_config.yaml
```

### Query
```bash
poetry run python -m server_cli query --config benchmark_config.yaml --query "revenue growth"
```

### Report
```bash
poetry run python -m server_cli report --config benchmark_config.yaml --qrels qrels.jsonl
```

## Experiments
### Reindex
```bash
poetry run python -m server_cli reindex --input data/corpus --out data/index --model "all-MiniLM-L12-v2"
```

### Query
```bash
poetry run python -m server_cli search --index data/index --query "revenue growth" --k 8
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
