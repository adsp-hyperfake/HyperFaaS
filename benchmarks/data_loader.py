import sqlite3
import pandas as pd
import json
from column_names import *

class Data:
    def __init__(self):
        self.metrics = None
        self.cold_starts = None
        self.scenarios = None

    def load_metrics(self, db_path: str) -> pd.DataFrame:
        """Get request latency for each function from database."""
        
        conn = sqlite3.connect(db_path)
        
        query = f"""
        SELECT
            *
        FROM {METRICS_TABLE}
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Preprocess timestamps
        df[CALL_QUEUED_TIMESTAMP] = pd.to_datetime(df[CALL_QUEUED_TIMESTAMP], unit='ns')
        df[LEAF_GOT_REQUEST_TIMESTAMP] = pd.to_datetime(df[LEAF_GOT_REQUEST_TIMESTAMP], unit='ns')
        df[LEAF_SCHEDULED_CALL_TIMESTAMP] = pd.to_datetime(df[LEAF_SCHEDULED_CALL_TIMESTAMP], unit='ns')
        df[GOT_RESPONSE_TIMESTAMP] = pd.to_datetime(df[GOT_RESPONSE_TIMESTAMP], unit='ns')
        
        # Convert timestamp column
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
        
        # Add computed latency columns
        df['scheduling_latency_ms'] = (df[LEAF_SCHEDULED_CALL_TIMESTAMP] - df[LEAF_GOT_REQUEST_TIMESTAMP]).dt.total_seconds() * 1000
        df['leaf_to_worker_latency_ms'] = (df[CALL_QUEUED_TIMESTAMP] - df[LEAF_SCHEDULED_CALL_TIMESTAMP]).dt.total_seconds() * 1000
        df['function_processing_latency_ms'] = (df[GOT_RESPONSE_TIMESTAMP] - df[CALL_QUEUED_TIMESTAMP]).dt.total_seconds() * 1000
        
        self.metrics = df
        return df
    
    def load_cpu_mem_stats(self, db_path: str) -> pd.DataFrame:
        """Get CPU and memory usage stats for each worker."""
        conn = sqlite3.connect(db_path)
        query = """
        SELECT * FROM cpu_mem_stats
        """
        df = pd.read_sql_query(query, conn)
        # filter out negative timestamps . sometimes happens when memory is almost full
        df = df[df['timestamp'] >= 0]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        conn.close()
        return df
    
    def load_cpu_mem_stats_labeled(self, db_path: str, label: str) -> pd.DataFrame:
        """Convenience wrapper around load_cpu_mem_stats that adds a column identifying the worker implementation (e.g. 'original' or 'model')."""
        df = self.load_cpu_mem_stats(db_path).copy()
        df["worker_type"] = label
        return df

    def load_metrics_labeled(self, db_path: str, label: str) -> pd.DataFrame:
        """Convenience wrapper around load_metrics that adds a column identifying the worker implementation (e.g. 'original' or 'model')."""
        df = self.load_metrics(db_path).copy()
        df["worker_type"] = label
        return df

    def load_cold_start_times(self, db_path: str) -> pd.DataFrame:
        """Calculate cold start times for each function instance."""
            
        conn = sqlite3.connect(db_path)

        query = """
        WITH start_events AS (
            SELECT 
                instance_id,
                function_id,
                timestamp as start_time
            FROM status_updates
            WHERE event = 3 AND status = 0  -- EVENT_START, STATUS_SUCCESS
        ),
        running_events AS (
            SELECT 
                instance_id,
                function_id,
                timestamp as running_time
            FROM status_updates
            WHERE event = 6 AND status = 0  -- EVENT_RUNNING, STATUS_SUCCESS
        )
        SELECT 
            s.function_id as {FUNCTION_ID},
            s.instance_id as {INSTANCE_ID},
            fi.image_tag as {IMAGE_TAG},
            s.start_time as start_time,
            r.running_time as running_time,
            (julianday(r.running_time) - julianday(s.start_time)) * 24 * 60 * 60 * 1000 as cold_start_ms
        FROM start_events s
        JOIN running_events r ON s.instance_id = r.instance_id
        JOIN function_images fi ON s.function_id = fi.function_id
        ORDER BY s.function_id, s.instance_id
        """

        df = pd.read_sql_query(query, conn)
        conn.close()
        
        self.cold_starts = df
        return df

    def load_k6_scenarios(self, scenarios_path: str) -> pd.DataFrame:
        """Parse k6 scenarios JSON and create a timeline dataframe of expected RPS."""
        
        with open(scenarios_path, 'r') as f:
            data = json.load(f)
        
        scenarios = data['scenarios']
        
        total_duration_str = data['metadata']['totalDuration']
        if total_duration_str.endswith('m'):
            total_duration_seconds = int(total_duration_str.rstrip('m')) * 60
        elif total_duration_str.endswith('s'):
            total_duration_seconds = int(total_duration_str.rstrip('s'))
        else:
            raise ValueError(f"Invalid total duration: {total_duration_str}")
        
        timeline_data = []
        
        # Calculate the expected rps for each second
        for second in range(total_duration_seconds):
            rps_by_image = {}
            
            for scenario_name, scenario in scenarios.items():
                image_tag = scenario['tags']['image_tag']
                
                start_time_str = scenario['startTime']
                start_time = int(start_time_str.rstrip('s'))
                
                if scenario['executor'] == 'constant-arrival-rate':
                    duration = int(scenario['duration'].rstrip('s'))
                    end_time = start_time + duration
                    
                    # Check if this second falls within the scenario duration
                    if start_time <= second < end_time:
                        rate = scenario['rate']
                        if image_tag not in rps_by_image:
                            rps_by_image[image_tag] = 0
                        rps_by_image[image_tag] += rate
                        
                elif scenario['executor'] == 'ramping-arrival-rate':
                    # Calculate the rate for this specific second
                    start_rate = scenario['startRate']
                    stage = scenario['stages'][0]  # Important: we assume a single stage !
                    target_rate = stage['target']
                    stage_duration = int(stage['duration'].rstrip('s'))
                    end_time = start_time + stage_duration
                    
                    # Check if this second falls within the scenario duration
                    if start_time <= second < end_time:
                        progress = (second - start_time) / stage_duration
                        current_rate = start_rate + (target_rate - start_rate) * progress
                        
                        if image_tag not in rps_by_image:
                            rps_by_image[image_tag] = 0
                        rps_by_image[image_tag] += current_rate
            
            # Add entries for each image tag at this second
            for image_tag, rps in rps_by_image.items():
                timeline_data.append({
                    'second': second,
                    'image_tag': image_tag,
                    'expected_rps': rps
                })
            
            # If no scenarios are active for any image at this second, add zero entries
            if not rps_by_image:
                # Get all unique image tags from scenarios
                all_image_tags = set(scenario['tags']['image_tag'] for scenario in scenarios.values())
                for image_tag in all_image_tags:
                    timeline_data.append({
                        'second': second,
                        'image_tag': image_tag,
                        'expected_rps': 0
                    })
        
        df = pd.DataFrame(timeline_data)
        
        # Fill missing combinations with 0
        if not df.empty:
            all_seconds = range(total_duration_seconds)
            all_image_tags = df[IMAGE_TAG].unique()
            
            # Create complete index
            complete_index = pd.MultiIndex.from_product([all_seconds, all_image_tags], 
                                                    names=['second', 'image_tag'])
            complete_df = pd.DataFrame(index=complete_index).reset_index()
            
            # Merge with actual data
            df = complete_df.merge(df, on=['second', 'image_tag'], how='left')
            df['expected_rps'] = df['expected_rps'].fillna(0)
        
        self.scenarios = df
        return df