import csv
import sqlite3

def create_tables(conn):
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        metric_name TEXT,
        timestamp INTEGER,
        metric_value REAL,
        check_name TEXT,
        error TEXT,
        error_code TEXT,
        expected_response TEXT,
        group_name TEXT,
        method TEXT,
        name TEXT,
        proto TEXT,
        scenario TEXT,
        service TEXT,
        status INTEGER,
        subproto TEXT,
        tls_version TEXT,
        url TEXT,
        extra_tags TEXT,
        metadata TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()

def import_csv_to_sqlite(csv_file='test_results.csv', db_file='metrics.db'):
    conn = sqlite3.connect(db_file)
    create_tables(conn)
    cursor = conn.cursor()

    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            cursor.execute('''
            INSERT INTO metrics (
                metric_name, timestamp, metric_value, check_name, error,
                error_code, expected_response, group_name, method, name,
                proto, scenario, service, status, subproto, tls_version,
                url, extra_tags, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['metric_name'],
                row['timestamp'],
                row['metric_value'],
                row['check'],
                row['error'],
                row['error_code'],
                row['expected_response'],
                row['group'],
                row['method'],
                row['name'],
                row['proto'],
                row['scenario'],
                row['service'],
                row['status'],
                row['subproto'],
                row['tls_version'],
                row['url'],
                row['extra_tags'],
                row['metadata']
            ))
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Import metrics from CSV to SQLite')
    parser.add_argument('--csv', default='test_results.csv', help='Path to CSV file')
    parser.add_argument('--db', default='metrics.db', help='Path to SQLite database')
    
    args = parser.parse_args()
    import_csv_to_sqlite(args.csv, args.db)