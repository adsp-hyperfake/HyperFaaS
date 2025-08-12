

import csv
import sqlite3
from datetime import datetime
import argparse
from column_names import *

def create_tables(conn):
    cursor = conn.cursor()
    
    cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS {METRICS_TABLE} (
        {REQUEST_ID} INTEGER PRIMARY KEY AUTOINCREMENT,
        {TIMESTAMP} TEXT,  -- ISO 8601 timestamp
        {FUNCTION_ID} TEXT,
        {IMAGE_TAG} TEXT,
        {GRPC_REQ_DURATION} INTEGER,
        {STATUS} TEXT,
        {ERROR} TEXT,
        {REQUEST_SIZE_BYTES} INTEGER,
        {RESPONSE_SIZE_BYTES} INTEGER,
        {CALL_QUEUED_TIMESTAMP} INTEGER, --unix nanoseconds
        {GOT_RESPONSE_TIMESTAMP} INTEGER, --unix nanoseconds
        {INSTANCE_ID} TEXT,
        {LEAF_GOT_REQUEST_TIMESTAMP} INTEGER, --unix nanoseconds
        {LEAF_SCHEDULED_CALL_TIMESTAMP} INTEGER, --unix nanoseconds
        {FUNCTION_PROCESSING_TIME_NS} INTEGER -- nanoseconds
    )
    ''')
    
    conn.commit()


def import_csv_to_sqlite(csv_file='test_results.csv', db_file='metrics.db'):
    """Import the new simplified CSV format into SQLite database"""
    conn = sqlite3.connect(db_file)
    create_tables(conn)
    cursor = conn.cursor()
    
    imported_count = 0
    skipped_count = 0
    
    try:
        with open(csv_file, 'r') as file:
            csv_reader = csv.DictReader(file)
            
            for row_num, row in enumerate(csv_reader, start=2):  # Start at 2 since header is row 1
                try:
                    # Convert numeric fields, handling empty strings
                    def safe_int(value):
                        if not value or value == '':
                            return None
                        try:
                            return int(value)
                        except ValueError:
                            return None
                    
                    cursor.execute(f'''
                    INSERT INTO {METRICS_TABLE} (
                        {TIMESTAMP}, {FUNCTION_ID}, {IMAGE_TAG}, {GRPC_REQ_DURATION}, {STATUS}, {ERROR},
                        {REQUEST_SIZE_BYTES}, {RESPONSE_SIZE_BYTES}, {CALL_QUEUED_TIMESTAMP},
                        {GOT_RESPONSE_TIMESTAMP}, {INSTANCE_ID}, {LEAF_GOT_REQUEST_TIMESTAMP},
                        {LEAF_SCHEDULED_CALL_TIMESTAMP}, {FUNCTION_PROCESSING_TIME_NS}
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row.get(TIMESTAMP),
                        row.get(FUNCTION_ID),
                        row.get(IMAGE_TAG),
                        safe_int(row.get('latency_ms')),
                        row.get(STATUS),
                        row.get(ERROR) if row.get(ERROR) else None,
                        safe_int(row.get('request_size_bytes')),
                        safe_int(row.get('response_size_bytes')),
                        row.get(CALL_QUEUED_TIMESTAMP),
                        row.get(GOT_RESPONSE_TIMESTAMP),
                        row.get(INSTANCE_ID) if row.get(INSTANCE_ID) else None,
                        row.get(LEAF_GOT_REQUEST_TIMESTAMP),
                        row.get(LEAF_SCHEDULED_CALL_TIMESTAMP),
                        safe_int(row.get('function_processing_time_ns'))
                    ))
                    
                    imported_count += 1
                    
                except Exception as e:
                    print(f"Warning: Skipping row {row_num} due to error: {e}")
                    print(f"  Row data: {row}")
                    skipped_count += 1
                    continue
        
        conn.commit()
        print(f"Successfully imported {imported_count} requests")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} rows due to errors")
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found")
    except Exception as e:
        print(f"Error importing CSV: {e}")
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser(description='Import simplified metrics CSV to SQLite')
    parser.add_argument('--csv', default='test_results.csv', 
                       help='Path to CSV file (default: test_results.csv)')
    parser.add_argument('--db', default='metrics.db', 
                       help='Path to SQLite database (default: metrics.db)')
    
    args = parser.parse_args()
    
    print(f"Importing from {args.csv} to {args.db}")
    import_csv_to_sqlite(args.csv, args.db)


if __name__ == "__main__":
    main()