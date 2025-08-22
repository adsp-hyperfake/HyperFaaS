#!/bin/bash

set -e

REMOTE_USER="ubuntu"
SUT_IP="10.0.0.3"
REMOTE_DB_PATH="~/HyperFaaS/benchmarks/metrics.db"

LOCAL_DIR="$HOME/HyperFaaS/benchmarks"
LOCAL_DB="$LOCAL_DIR/metrics.db"
SUT_DB="$LOCAL_DIR/metrics_sut.db"
SCHEMA_SQL="$LOCAL_DIR/schema_from_sut.sql"

echo "📥 Pulling metrics DB from SUT VM ($SUT_IP)..."
scp ${REMOTE_USER}@${SUT_IP}:${REMOTE_DB_PATH} ${SUT_DB}

echo "🔎 Checking if local DB has necessary tables..."
HAS_TABLES=$(sqlite3 "$LOCAL_DB" "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('status_updates', 'cpu_mem_stats') LIMIT 1;")

if [ -z "$HAS_TABLES" ]; then
  echo "📄 Local DB missing tables. Extracting schema from SUT DB..."
  sqlite3 "$SUT_DB" ".schema" | grep -v sqlite_sequence > "$SCHEMA_SQL"
  echo "📥 Creating schema in local DB..."
  sqlite3 "$LOCAL_DB" < "$SCHEMA_SQL"
  rm -f "$SCHEMA_SQL"
else
  echo "✅ Local DB already has required tables."
fi

echo "🔗 Merging data from SUT DB into local DB..."
sqlite3 "${LOCAL_DB}" <<EOF
ATTACH DATABASE '${SUT_DB}' AS sut;

INSERT INTO status_updates SELECT * FROM sut.status_updates;
INSERT INTO cpu_mem_stats SELECT * FROM sut.cpu_mem_stats;

DETACH DATABASE sut;
EOF

echo "🧹 Cleaning up..."
rm -f "${SUT_DB}"

echo "✅ Merge complete."