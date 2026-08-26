# Cost Tracking & Logging

Log requests and bodies asynchronously for analytics and auditing.

```python
from majordomo_llm import get_llm_instance
from majordomo_llm.logging import LoggingLLM, SqliteAdapter, FileStorageAdapter

llm = get_llm_instance("anthropic", "claude-sonnet-5")

# Local dev: SQLite + local files
db = await SqliteAdapter.create("llm_logs.db")
storage = await FileStorageAdapter.create("./request_logs")
logged = LoggingLLM(llm, db, storage)
resp = await logged.get_response("Hello!")
await logged.close()
```

Cloud example (Postgres + S3):

```python
from majordomo_llm.logging import LoggingLLM, PostgresAdapter, S3Adapter

db = await PostgresAdapter.create(host="localhost", port=5432, database="llm_logs", user="postgres", password="password")
storage = await S3Adapter.create(bucket="my-llm-logs", prefix="requests")
logged = LoggingLLM(llm, db, storage)
```

Tips
- Redact secrets in prompts/outputs when necessary.
- Track `api_key_hash`/`api_key_alias` for attribution.
- See README for the database schema and adapters.
