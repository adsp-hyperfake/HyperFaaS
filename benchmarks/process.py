import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm
from column_names import *
class TrainingData:
    def __init__(self, db_path: str, active_calls_window_size: int):
        """Initialize the database processor with the path to the SQLite database."""
        self.db_path = db_path
        self.conn = None
        self.active_calls_window_size = active_calls_window_size

    def connect(self):
        """Connect to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            raise

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print("Database connection closed")

    def create_training_data_table(self):
        """Create the training_data table with the specified schema."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {REQUEST_ID} TEXT,
            {IMAGE_TAG} TEXT,
            {TIMESTAMP} INTEGER,
            {STATUS} TEXT,
            {REQUEST_SIZE_BYTES} INTEGER,
            {FUNCTION_INSTANCES_COUNT} INTEGER,
            {ACTIVE_FUNCTION_CALLS_COUNT} INTEGER,
            {WORKER_CPU_USAGE} REAL,
            {WORKER_RAM_USAGE} INTEGER,
            {FUNCTION_PROCESSING_TIME_NS} INTEGER,
            {FUNCTION_CPU_USAGE} REAL,
            {FUNCTION_RAM_USAGE} INTEGER
        );
        """

        try:
            cursor = self.conn.cursor()
            cursor.execute(create_table_sql)
            self.conn.commit()
            print("Training data table created successfully")
        except sqlite3.Error as e:
            print(f"Error creating training_data table: {e}")
            raise

    def get_metrics_data(self) -> pd.DataFrame:
        """Fetch all metrics data with optimized data types."""
        query = f"""
        SELECT
            {REQUEST_ID},
            {TIMESTAMP},
            {INSTANCE_ID},
            {IMAGE_TAG},
            {STATUS},
            {GRPC_REQ_DURATION},
            {CALL_QUEUED_TIMESTAMP},
            {GOT_RESPONSE_TIMESTAMP},
            {LEAF_GOT_REQUEST_TIMESTAMP},
            {LEAF_SCHEDULED_CALL_TIMESTAMP},
            {FUNCTION_PROCESSING_TIME_NS},
            {REQUEST_SIZE_BYTES},
            {RESPONSE_SIZE_BYTES}
        FROM {METRICS_TABLE}
        WHERE {REQUEST_ID} IS NOT NULL
        ORDER BY {TIMESTAMP}
        """

        try:
            df = pd.read_sql_query(query, self.conn)

            # Convert timestamp from ISO format to nanoseconds
            df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP]).astype(np.int64)

            df[CALL_QUEUED_TIMESTAMP] = pd.to_numeric(df[CALL_QUEUED_TIMESTAMP], downcast='float')
            df[GOT_RESPONSE_TIMESTAMP] = pd.to_numeric(df[GOT_RESPONSE_TIMESTAMP], downcast='float')
            df[LEAF_GOT_REQUEST_TIMESTAMP] = pd.to_numeric(df[LEAF_GOT_REQUEST_TIMESTAMP], downcast='float')
            df[LEAF_SCHEDULED_CALL_TIMESTAMP] = pd.to_numeric(df[LEAF_SCHEDULED_CALL_TIMESTAMP], downcast='float')

            df[FUNCTION_PROCESSING_TIME_NS] = pd.to_numeric(df[FUNCTION_PROCESSING_TIME_NS], downcast='float')
            df[REQUEST_SIZE_BYTES] = pd.to_numeric(df[REQUEST_SIZE_BYTES], downcast='integer', errors='coerce')
            df[RESPONSE_SIZE_BYTES] = pd.to_numeric(df[RESPONSE_SIZE_BYTES], downcast='integer', errors='coerce')

            print(f"Fetched {len(df)} metrics records")
            return df
        except Exception as e:
            print(f"Error fetching metrics data: {e}")
            raise

    def get_cpu_mem_stats(self) -> pd.DataFrame:
        """Fetch CPU and memory statistics with optimized data types."""
        query = f"""
        SELECT
            {INSTANCE_ID},
            {FUNCTION_ID},
            {IMAGE_TAG},
            {TIMESTAMP},
            {CPU_USAGE_PERCENT},
            {MEMORY_USAGE},
            {MEMORY_USAGE_LIMIT},
            {MEMORY_USAGE_PERCENT}
        FROM {CPU_MEM_STATS_TABLE}
        ORDER BY {TIMESTAMP}
        """

        try:
            df = pd.read_sql_query(query, self.conn)

            # everything to nanoseconds
            df[TIMESTAMP] = pd.to_numeric(df[TIMESTAMP], downcast='float') * 1e9

            df[CPU_USAGE_PERCENT] = pd.to_numeric(df[CPU_USAGE_PERCENT], downcast='float')
            df[MEMORY_USAGE] = pd.to_numeric(df[MEMORY_USAGE], downcast='integer', errors='coerce')
            df[MEMORY_USAGE_LIMIT] = pd.to_numeric(df[MEMORY_USAGE_LIMIT], downcast='integer', errors='coerce')
            df[MEMORY_USAGE_PERCENT] = pd.to_numeric(df[MEMORY_USAGE_PERCENT], downcast='float')

            print(f"Fetched {len(df)} CPU/memory stats records")
            return df
        except Exception as e:
            print(f"Error fetching CPU/memory stats: {e}")
            raise

    def precompute_active_function_calls(self, metrics_df: pd.DataFrame) -> Dict[float, int]:
        """
        Pre-calculate the number of active function calls for all timestamps.
        This is much more efficient than calculating for each timestamp individually (that was taking hours).
        """
        print("Pre-computing active function calls...")

        # What do we define as active function call?
        start_key = CALL_QUEUED_TIMESTAMP
        #end_key = GOT_RESPONSE_TIMESTAMP
        #start_key = LEAF_GOT_REQUEST_TIMESTAMP
        end_key = GOT_RESPONSE_TIMESTAMP

        # window size in milliseconds
        window_size = self.active_calls_window_size

        # Get all unique timestamps
        unique_timestamps = sorted(metrics_df[TIMESTAMP].unique())
        active_calls_dict = {}

        # Convert to numpy arrays for faster operations
        start_timestamps = metrics_df[start_key].values
        end_timestamps = metrics_df[end_key].values

        for timestamp in tqdm(unique_timestamps, desc="Computing active calls"):
            active_mask = (start_timestamps - window_size <= timestamp) & (timestamp <= end_timestamps + window_size)
            active_calls_dict[timestamp] = int(np.sum(active_mask))

        print(f"Pre-computed active calls for {len(unique_timestamps)} unique timestamps")
        return active_calls_dict



    def precompute_worker_stats_by_second(self, stats_df: pd.DataFrame) -> Dict[int, Dict]:
        """
        Pre-compute worker stats aggregated by second (timestamp).
        """
        print("Pre-computing worker stats by second...")

        # Convert timestamp from nanoseconds back to seconds for grouping
        stats_df_copy = stats_df.copy()
        stats_df_copy['timestamp_seconds'] = (stats_df_copy[TIMESTAMP] / 1e9).astype(int)

        # Group by second and compute aggregated stats
        worker_stats = {}
        for timestamp_sec, group in tqdm(stats_df_copy.groupby('timestamp_seconds'), desc="Computing worker stats"):
            unique_instances = group[INSTANCE_ID].nunique()
            avg_cpu_usage = group[CPU_USAGE_PERCENT].mean()
            avg_memory_usage = group[MEMORY_USAGE].mean()

            worker_stats[timestamp_sec] = {
                'function_instances_count': int(unique_instances),
                'worker_cpu_usage': float(avg_cpu_usage),
                'worker_ram_usage': int(avg_memory_usage) if not pd.isna(avg_memory_usage) else 0
            }

        print(f"Pre-computed worker stats for {len(worker_stats)} seconds")
        return worker_stats

    def precompute_instance_stats_by_second(self, stats_df: pd.DataFrame) -> Dict[Tuple[str, int], Tuple[float, int]]:
        """
        Pre-compute instance stats by second and instance_id
        (instance_id, timestamp_seconds) -> (cpu_usage, ram_usage)
        """
        print("Pre-computing instance stats by second...")

        # Convert timestamp from nanoseconds back to seconds
        stats_df_copy = stats_df.copy()
        stats_df_copy['timestamp_seconds'] = (stats_df_copy[TIMESTAMP] / 1e9).astype(int)

        instance_stats = {}
        for _, row in tqdm(stats_df_copy.iterrows(), total=len(stats_df_copy), desc="Computing instance stats"):
            instance_id = row[INSTANCE_ID]
            timestamp_sec = row['timestamp_seconds']
            cpu_usage = float(row['cpu_usage_percent'])
            ram_usage = int(row['memory_usage']) if pd.notna(row['memory_usage']) else 0

            instance_stats[(instance_id, timestamp_sec)] = (cpu_usage, ram_usage)

        print(f"Pre-computed instance stats for {len(instance_stats)} combinations")
        return instance_stats

    def process_training_data(self):
        """Main method to process and create training data."""
        print("Starting training data processing...")

        metrics_df = self.get_metrics_data()
        stats_df = self.get_cpu_mem_stats()

        if len(metrics_df) == 0:
            print("No metrics data found")
            return

        active_calls_lookup = self.precompute_active_function_calls(metrics_df)

        worker_stats_lookup = self.precompute_worker_stats_by_second(stats_df)
        instance_cpu_ram_lookup = self.precompute_instance_stats_by_second(stats_df)

        training_data = []
        total_requests = len(metrics_df)

        for _, row in tqdm(metrics_df.iterrows(), total=total_requests, desc="Processing requests"):
            # we insert 0 values for all metrics that are not available for now. I think this fks up the training so we should try to find a better way to handle this.
            try:
                request_id = row[REQUEST_ID]
                timestamp = row[TIMESTAMP]
                function_image_tag = row[IMAGE_TAG]
                instance_id = row[INSTANCE_ID]

                # Calculate request body size
                request_body_size = int(row[REQUEST_SIZE_BYTES])
                active_calls = active_calls_lookup.get(timestamp, 0)

                worker_stats = worker_stats_lookup.get(int(timestamp / 1e9), {
                    'function_instances_count': 0,
                    'worker_cpu_usage': 0.0,
                    'worker_ram_usage': 0
                })

                function_cpu_usage, function_ram_usage = instance_cpu_ram_lookup.get(
                    (instance_id, int(timestamp / 1e9)), (0.0, 0)
                )

                if pd.notna(row[FUNCTION_PROCESSING_TIME_NS]):
                    function_runtime = int(row[FUNCTION_PROCESSING_TIME_NS])
                else:
                    continue

                training_record = {
                    REQUEST_ID: request_id,
                    TIMESTAMP: int(timestamp),
                    STATUS: row[STATUS],
                    REQUEST_SIZE_BYTES: request_body_size,
                    IMAGE_TAG: function_image_tag,
                    FUNCTION_INSTANCES_COUNT: int(worker_stats['function_instances_count']),
                    ACTIVE_FUNCTION_CALLS_COUNT: int(active_calls),
                    WORKER_CPU_USAGE: float(worker_stats['worker_cpu_usage']),
                    WORKER_RAM_USAGE: int(worker_stats['worker_ram_usage']),
                    FUNCTION_PROCESSING_TIME_NS: int(function_runtime),
                    FUNCTION_CPU_USAGE: float(function_cpu_usage),
                    FUNCTION_RAM_USAGE: int(function_ram_usage)
                }

                training_data.append(training_record)

            except Exception as e:
                print(f"Error processing record {row.get(REQUEST_ID, 'unknown')}: {e}")
                continue

        print(f"Processed {len(training_data)} training records")
        return training_data


    def avg_one_to_n_relations(self):
        """Creates a new table training_data_avg with averaged one-to-n relations."""
        
        print("Averaging up 1 to n relations...")

        avg_sql = f"""
            INSERT INTO training_data_avg (
                id,
                {REQUEST_ID},
                {TIMESTAMP},
                {STATUS},
                
                {IMAGE_TAG},
                {REQUEST_SIZE_BYTES},
                {FUNCTION_INSTANCES_COUNT},
                {ACTIVE_FUNCTION_CALLS_COUNT},
                {WORKER_CPU_USAGE},
                {WORKER_RAM_USAGE},
                
                {FUNCTION_PROCESSING_TIME_NS},
                {FUNCTION_CPU_USAGE},
                {FUNCTION_RAM_USAGE}
            )
            SELECT
                -- use min for non-inputs
                MIN(id) as id,                      
                MIN({REQUEST_ID}) as {REQUEST_ID},  
                MIN({TIMESTAMP}) as {TIMESTAMP},     
                MIN({STATUS}) as {STATUS},          
                
                -- group by inputs
                {IMAGE_TAG},
                {REQUEST_SIZE_BYTES},
                {FUNCTION_INSTANCES_COUNT},
                {ACTIVE_FUNCTION_CALLS_COUNT},
                {WORKER_CPU_USAGE},
                {WORKER_RAM_USAGE},
                
                -- average output
                AVG(CAST({FUNCTION_PROCESSING_TIME_NS} AS REAL)) as {FUNCTION_PROCESSING_TIME_NS},  
                AVG({FUNCTION_CPU_USAGE}) as {FUNCTION_CPU_USAGE},                                  
                AVG(CAST({FUNCTION_RAM_USAGE} AS REAL)) as {FUNCTION_RAM_USAGE}                     
            FROM training_data
            GROUP BY
                {REQUEST_SIZE_BYTES},
                {IMAGE_TAG},
                {FUNCTION_INSTANCES_COUNT},
                {ACTIVE_FUNCTION_CALLS_COUNT},
                {WORKER_CPU_USAGE},
                {WORKER_RAM_USAGE}
            """

        try:
            cursor = self.conn.cursor()
            
            cursor.execute("CREATE TABLE IF NOT EXISTS training_data_avg AS SELECT * FROM training_data WHERE 0")
            cursor.execute("DELETE FROM training_data_avg")
            
            cursor.execute(avg_sql)
            rows_affected = cursor.rowcount

            self.conn.commit()
            print(
                f"Successfully averaged data - {rows_affected} grouped records created")
            return rows_affected
        
        except sqlite3.Error as e:
            print(f"Error averaging up 1 to n relations: {e}")
            self.conn.rollback()
            raise


    def insert_training_data(self, training_data: List[Dict]):
        """Insert the processed training data into the training_data table."""
        if not training_data:
            print("No training data to insert")
            return

        insert_sql = f"""
        INSERT INTO training_data (
            {REQUEST_ID},
            {TIMESTAMP},
            {STATUS},
            {REQUEST_SIZE_BYTES},
            {IMAGE_TAG},
            {FUNCTION_INSTANCES_COUNT},
            {ACTIVE_FUNCTION_CALLS_COUNT},
            {WORKER_CPU_USAGE},
            {WORKER_RAM_USAGE},
            {FUNCTION_PROCESSING_TIME_NS},
            {FUNCTION_CPU_USAGE},
            {FUNCTION_RAM_USAGE}
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        try:
            cursor = self.conn.cursor()

            # Clear existing data
            cursor.execute("DELETE FROM training_data")

            # Insert new data
            for record in training_data:
                cursor.execute(insert_sql, (
                    record[REQUEST_ID],
                    record[TIMESTAMP],
                    record[STATUS],
                    record[REQUEST_SIZE_BYTES],
                    record[IMAGE_TAG],
                    record[FUNCTION_INSTANCES_COUNT],
                    record[ACTIVE_FUNCTION_CALLS_COUNT],
                    record[WORKER_CPU_USAGE],
                    record[WORKER_RAM_USAGE],
                    record[FUNCTION_PROCESSING_TIME_NS],
                    record[FUNCTION_CPU_USAGE],
                    record[FUNCTION_RAM_USAGE]
                ))

            self.conn.commit()
            print(f"Successfully inserted {len(training_data)} training records")

        except sqlite3.Error as e:
            print(f"Error inserting training data: {e}")
            self.conn.rollback()
            raise

    def run(self):
        """Main execution method."""
        try:
            self.connect()
            self.create_training_data_table()
            training_data = self.process_training_data()
            if training_data:
                self.insert_training_data(training_data)
                print("Training data processing completed successfully")
            else:
                print("No training data was generated")
                
            self.avg_one_to_n_relations()
        except Exception as e:
            print(f"Error during processing: {e}")
            raise
        finally:
            self.close()

def main():
    """Main function to run the database processor."""
    parser = argparse.ArgumentParser(description='Process metrics database')
    parser.add_argument('--db-path', default='metrics.db', help='Path to SQLite database')
    parser.add_argument('--active-calls-window-size', type=int, default=100, help='Window size in milliseconds for active calls')
    args = parser.parse_args()
    try:
        processor = TrainingData(args.db_path, args.active_calls_window_size/2)
        processor.run()
        print("Processing completed successfully!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
