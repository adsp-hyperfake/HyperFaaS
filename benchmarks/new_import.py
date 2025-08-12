

import csv
import sqlite3
from datetime import datetime
import argparse

def create_tables(conn):
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS metrics (
        request_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,  -- ISO 8601 timestamp
        function_id TEXT,
        image_tag TEXT,
        latency_ms INTEGER,
        status TEXT,
        error TEXT,
        request_size_bytes INTEGER,
        response_size_bytes INTEGER,
        call_queued_timestamp INTEGER, --unix nanoseconds
        got_response_timestamp INTEGER, --unix nanoseconds
        instance_id TEXT,
        leaf_got_request_timestamp INTEGER, --unix nanoseconds
        leaf_scheduled_call_timestamp INTEGER, --unix nanoseconds
        function_processing_time_ns INTEGER, -- nanoseconds
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
                    
                    cursor.execute('''
                    INSERT INTO metrics (
                        timestamp, function_id, image_tag, latency_ms, status, error,
                        request_size_bytes, response_size_bytes, call_queued_timestamp,
                        got_response_timestamp, instance_id, leaf_got_request_timestamp,
                        leaf_scheduled_call_timestamp, function_processing_time_ns
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row.get('timestamp'),
                        row.get('function_id'),
                        row.get('image_tag'),
                        safe_int(row.get('latency_ms')),
                        row.get('status'),
                        row.get('error') if row.get('error') else None,
                        safe_int(row.get('request_size_bytes')),
                        safe_int(row.get('response_size_bytes')),
                        row.get('call_queued_timestamp'),
                        row.get('got_response_timestamp'),
                        row.get('instance_id') if row.get('instance_id') else None,
                        row.get('leaf_got_request_timestamp'),
                        row.get('leaf_scheduled_call_timestamp'),
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