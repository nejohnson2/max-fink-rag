# Chat Interaction Logging

This document describes the data collection system for chat interactions between users and the RAG system.

## Overview

The system logs every query and response in a lightweight JSONL (JSON Lines) format. Each interaction is appended to a single log file, making it easy to analyze user behavior, system performance, and conversation patterns.

## Configuration

Set the log file path in your `.env` file:

```env
# Data Collection Configuration
# Path to store chat interaction logs in JSONL format
CHAT_LOG_PATH = "logs/chat_interactions.jsonl"
```

The log directory will be created automatically if it doesn't exist.

## Log Format

Each line in the log file is a complete JSON object with the following fields:

```json
{
  "timestamp": "2026-01-15T10:30:45.123456+00:00",
  "session_id": "session_1736934645_abc123def",
  "question": "What was Max Fink's role at Stony Brook?",
  "answer": "According to the biographical files, Max Fink was...",
  "sources": [
    {
      "parent_id": "item_1234",
      "title": "Max Fink Biography",
      "source": "https://exhibits.library.stonybrook.edu/...",
      "collection": "Biographical Files"
    }
  ],
  "intent": "biographical",
  "retrieval_time_seconds": 2.34,
  "answer_time_seconds": 1.56,
  "total_time_seconds": 3.90,
  "excluded_parent_ids": [],
  "num_sources": 6
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | string | ISO 8601 UTC timestamp of the interaction |
| `session_id` | string | Unique identifier for the browser tab session |
| `question` | string | User's query text |
| `answer` | string | RAG system's response |
| `sources` | array | List of source documents with metadata |
| `intent` | string | Classified intent: biographical/research/correspondence |
| `retrieval_time_seconds` | float | Time spent on retrieval + reranking |
| `answer_time_seconds` | float | Time spent generating the answer |
| `total_time_seconds` | float | Total query processing time |
| `excluded_parent_ids` | array | List of excluded document IDs (if any) |
| `num_sources` | integer | Number of source documents used |

## Analysis Tools

### Quick Analysis with Python

Use the included analysis script:

```bash
# View summary statistics
python analyze_logs.py logs/chat_interactions.jsonl

# Export to CSV for Excel/pandas
python analyze_logs.py logs/chat_interactions.jsonl --csv output.csv
```

### Using `jq` (Command Line)

```bash
# Count total interactions
cat logs/chat_interactions.jsonl | wc -l

# View most recent query
tail -1 logs/chat_interactions.jsonl | jq '.'

# Extract all questions
jq -r '.question' logs/chat_interactions.jsonl

# Calculate average response time
jq -s 'map(.total_time_seconds) | add/length' logs/chat_interactions.jsonl

# Count queries by intent
jq -r '.intent' logs/chat_interactions.jsonl | sort | uniq -c

# Find slow queries (>5 seconds)
jq 'select(.total_time_seconds > 5)' logs/chat_interactions.jsonl

# Get unique session count
jq -r '.session_id' logs/chat_interactions.jsonl | sort -u | wc -l
```

### Using Python pandas

```python
import json
import pandas as pd

# Load logs into DataFrame
logs = []
with open('logs/chat_interactions.jsonl', 'r') as f:
    for line in f:
        logs.append(json.loads(line))

df = pd.DataFrame(logs)

# Basic analysis
print(df.describe())
print(df['intent'].value_counts())
print(df['total_time_seconds'].mean())

# Group by session
session_stats = df.groupby('session_id').agg({
    'question': 'count',
    'total_time_seconds': 'mean'
}).rename(columns={'question': 'num_queries', 'total_time_seconds': 'avg_time'})

print(session_stats)
```

## Performance Characteristics

- **Write Speed**: Append-only writes are extremely fast (< 1ms)
- **File Locking**: Not required due to atomic append operations
- **Disk Usage**: ~1-3 KB per interaction (varies with answer length)
- **Impact**: Negligible impact on query latency (< 0.1%)

## Privacy Considerations

The logs contain:
- ✅ Full query text and responses
- ✅ Source document metadata
- ✅ Timing and performance metrics
- ❌ No user IP addresses
- ❌ No personally identifiable information (unless in queries)

**Important**: Review your institution's data retention and privacy policies before deploying in production.

## Backup and Rotation

For production use, consider implementing log rotation:

```bash
# Example: Rotate logs weekly
mv logs/chat_interactions.jsonl logs/chat_interactions_$(date +%Y%m%d).jsonl
touch logs/chat_interactions.jsonl
```

Or use a tool like `logrotate`:

```
/path/to/logs/chat_interactions.jsonl {
    weekly
    rotate 12
    compress
    missingok
    notifempty
    create 0644 www-data www-data
}
```

## Example Queries

### Most Common Questions

```bash
jq -r '.question' logs/chat_interactions.jsonl | sort | uniq -c | sort -rn | head -10
```

### Sessions with Most Interactions

```bash
jq -r '.session_id' logs/chat_interactions.jsonl | sort | uniq -c | sort -rn | head -10
```

### Average Time by Intent

```bash
jq -s 'group_by(.intent) | map({intent: .[0].intent, avg_time: (map(.total_time_seconds) | add/length)})' logs/chat_interactions.jsonl
```

### Export Specific Session

```bash
jq 'select(.session_id == "session_123...")' logs/chat_interactions.jsonl > session_123.jsonl
```

## Troubleshooting

### Log file not being created

1. Check that the `CHAT_LOG_PATH` is set in `.env`
2. Verify write permissions on the logs directory
3. Check Flask logs for any error messages

### Logs growing too large

1. Implement log rotation (see above)
2. Consider archiving old logs to compressed storage
3. Aggregate statistics and delete raw logs after analysis

### Analysis script errors

Ensure Python dependencies are installed:
```bash
pip install pandas  # Only needed for CSV export
```
