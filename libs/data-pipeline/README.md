# Data Pipeline

## Purpose
This component handles data ingestion (ETL), normalization, and preparation for the API or training.

## Workflow

1.  **Ingestion**: Scripts in `src/data_pipeline/ingest/` fetch data from external sources (GCS, S3, APIs).
2.  **Processing**: Clean and normalize data using Pandas/Polars.
3.  **Loading**: Load data into the main database or vector store.

## How to Run
Always use `uv run` to execute scripts to ensure dependencies are loaded.

```bash
uv run python src/data_pipeline/ingest/load_data.py --source ./data
```

## Testing
Write unit tests for your transformation logic in `tests/`.
