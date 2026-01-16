#!/usr/bin/env python3
"""Example script for analyzing chat interaction logs.

This demonstrates how to read and analyze the JSONL log files created by
the RAG system. Each line in the log file is a complete JSON object.

Usage:
    python analyze_logs.py logs/chat_interactions.jsonl
"""

import json
import sys
from collections import Counter
from pathlib import Path
from datetime import datetime


def analyze_logs(log_path: str):
    """Load and analyze chat interaction logs."""
    log_file = Path(log_path)

    if not log_file.exists():
        print(f"Log file not found: {log_path}")
        return

    interactions = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                interactions.append(json.loads(line))

    if not interactions:
        print("No interactions found in log file.")
        return

    print(f"\n{'='*60}")
    print(f"Chat Interaction Log Analysis")
    print(f"{'='*60}\n")

    # Basic statistics
    print(f"Total interactions: {len(interactions)}")
    print(f"Date range: {interactions[0]['timestamp']} to {interactions[-1]['timestamp']}")

    # Unique sessions
    unique_sessions = set(i['session_id'] for i in interactions)
    print(f"Unique sessions: {len(unique_sessions)}")

    # Intent distribution
    intents = Counter(i.get('intent') for i in interactions if i.get('intent'))
    print(f"\nIntent distribution:")
    for intent, count in intents.most_common():
        print(f"  {intent}: {count} ({count/len(interactions)*100:.1f}%)")

    # Average timing statistics
    retrieval_times = [i['retrieval_time_seconds'] for i in interactions if i.get('retrieval_time_seconds')]
    answer_times = [i['answer_time_seconds'] for i in interactions if i.get('answer_time_seconds')]
    total_times = [i['total_time_seconds'] for i in interactions if i.get('total_time_seconds')]

    if retrieval_times:
        print(f"\nAverage retrieval time: {sum(retrieval_times)/len(retrieval_times):.2f}s")
    if answer_times:
        print(f"Average answer time: {sum(answer_times)/len(answer_times):.2f}s")
    if total_times:
        print(f"Average total time: {sum(total_times)/len(total_times):.2f}s")

    # Sources statistics
    source_counts = [i['num_sources'] for i in interactions if i.get('num_sources')]
    if source_counts:
        print(f"\nAverage sources per query: {sum(source_counts)/len(source_counts):.1f}")

    # Recent queries
    print(f"\n{'='*60}")
    print(f"Recent Queries (last 5):")
    print(f"{'='*60}\n")

    for i in interactions[-5:]:
        timestamp = i['timestamp']
        session_id = i['session_id'][:16] + "..."  # Truncate for display
        question = i['question'][:60] + "..." if len(i['question']) > 60 else i['question']
        intent = i.get('intent', 'unknown')
        total_time = i.get('total_time_seconds', 0)

        print(f"[{timestamp}] Session: {session_id}")
        print(f"  Q: {question}")
        print(f"  Intent: {intent} | Time: {total_time:.2f}s")
        print()


def export_to_csv(log_path: str, output_path: str = "chat_log_export.csv"):
    """Export logs to CSV format for further analysis."""
    import csv

    log_file = Path(log_path)
    if not log_file.exists():
        print(f"Log file not found: {log_path}")
        return

    interactions = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                interactions.append(json.loads(line))

    if not interactions:
        print("No interactions to export.")
        return

    # Flatten the data for CSV export
    fieldnames = [
        'timestamp', 'session_id', 'question', 'answer', 'intent',
        'retrieval_time_seconds', 'answer_time_seconds', 'total_time_seconds',
        'num_sources', 'excluded_parent_ids_count'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in interactions:
            writer.writerow({
                'timestamp': i['timestamp'],
                'session_id': i['session_id'],
                'question': i['question'],
                'answer': i['answer'],
                'intent': i.get('intent', ''),
                'retrieval_time_seconds': i.get('retrieval_time_seconds', ''),
                'answer_time_seconds': i.get('answer_time_seconds', ''),
                'total_time_seconds': i.get('total_time_seconds', ''),
                'num_sources': i.get('num_sources', ''),
                'excluded_parent_ids_count': len(i.get('excluded_parent_ids', [])),
            })

    print(f"Exported {len(interactions)} interactions to {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_logs.py <log_file_path> [--csv output.csv]")
        sys.exit(1)

    log_path = sys.argv[1]

    if len(sys.argv) > 2 and sys.argv[2] == '--csv':
        csv_output = sys.argv[3] if len(sys.argv) > 3 else "chat_log_export.csv"
        export_to_csv(log_path, csv_output)
    else:
        analyze_logs(log_path)
