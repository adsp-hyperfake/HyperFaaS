import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
from collections.abc import Mapping
import time

IMAGE_PALETTE = {
    "hyperfaas-bfs-json:latest": "blue",
    "hyperfaas-echo:latest": "red",
    "hyperfaas-thumbnailer-json:latest": "green",
}

# Consistent palette for run/worker type labels in multi-run comparisons
RUN_PALETTE = {
    "linear": "#1f77b4",      # blue
    "rf": "#ff7f0e",     # orange  
    "nn": "#2ca02c",        # green
    "nn-2": "#d62728",        # red
    "real": "#9467bd",  # purple
    "optimized": "#8c564b", # brown
    "test": "#e377c2",      # pink
    "control": "#7f7f7f",   # gray
    "exp": "#bcbd22",       # olive
    "new": "#17becf",       # cyan
    "metrics": "#1f77b4",   # blue , for when just one single db
}

def measure_time(func):
    """Decorator to measure the time taken to execute a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time taken by {func.__name__}: {end_time - start_time:.3f} seconds")
        return result
    return wrapper

class Plotter:
    def __init__(self, show: bool = True, save_path=None, prefix: str = None, normalize_time: bool = False):
        self.show = show
        self.save_path = save_path
        self.save = bool(self.save_path)
        self.prefix = prefix + "_" if prefix else ""
        self.normalize_time = normalize_time

    def _prepare_multi_run_data(self, runs: Mapping[str, pd.DataFrame], apply_time_normalization: bool | None = None) -> pd.DataFrame:
        """
        Prepare multi-run data by adding run labels and optionally normalizing time.
        
        Args:
            runs: Dict mapping run_label → DataFrame
            apply_time_normalization: If None, uses self.normalize_time. If specified, overrides it.
        
        Returns:
            Combined DataFrame with worker_type column and optionally normalized timestamps
        """
        tagged = []
        should_normalize = apply_time_normalization if apply_time_normalization is not None else self.normalize_time
        
        if should_normalize:
            print("Normalizing time...")
            # First, convert all timestamps to datetime and find global minimum
            baseline_time = None
            converted_runs = []
            
            for label, df in runs.items():
                df2 = df.copy()
                # Convert timestamp to datetime first
                df2["timestamp"] = pd.to_datetime(df2["timestamp"], errors="coerce")
                # Find minimum timestamp for this run
                run_min = df2["timestamp"].min()
                if pd.notna(run_min):
                    if baseline_time is None or run_min < baseline_time:
                        baseline_time = run_min
                converted_runs.append((label, df2))
            
            # Use epoch as baseline if no valid timestamps found
            if baseline_time is None or pd.isna(baseline_time):
                baseline_time = pd.Timestamp('1970-01-01')
            
            # Now normalize each run to start from the baseline
            for label, df2 in converted_runs:
                # Calculate time difference and add to baseline
                run_min = df2["timestamp"].min()
                if pd.notna(run_min):
                    time_offset = run_min - baseline_time
                    df2["timestamp"] = baseline_time + (df2["timestamp"] - run_min)
                
                if "worker_type" in df2:
                    df2["worker_type"] = label
                else:
                    df2 = df2.rename_axis(None, axis=1)
                    df2["worker_type"] = label
                tagged.append(df2)
        else:
            for label, df in runs.items():
                df2 = df.copy()
                if "worker_type" in df2:
                    df2["worker_type"] = label
                else:
                    df2 = df2.rename_axis(None, axis=1)
                    df2["worker_type"] = label
                tagged.append(df2)
        
        return pd.concat(tagged, ignore_index=True, sort=False)

    def _filter_valid_latency_data(self, df: pd.DataFrame, ensure_datetime: bool = True) -> pd.DataFrame:
        """
        Filter for valid latency data and ensure proper data types.
        
        Args:
            df: Input DataFrame
            ensure_datetime: Whether to convert timestamp to datetime (only if not normalized)
        
        Returns:
            Filtered DataFrame with valid grpc_req_duration as float
        """
        df_filtered = df[df["grpc_req_duration"].notna()].copy()
        df_filtered["grpc_req_duration"] = df_filtered["grpc_req_duration"].astype(float)
        
        # Only convert timestamp if not already normalized and requested
        if ensure_datetime and not self.normalize_time:
            df_filtered["timestamp"] = pd.to_datetime(df_filtered["timestamp"], errors="coerce")
        
        return df_filtered

    @measure_time
    def plot_throughput_leaf_node(self, df: pd.DataFrame):
        """Plot the number of requests that completed processing per second at the leaf node level.
        We use requests fully completed as the basis for throughput here.
        """
        print("Plotting requests processed per second at the leaf node level...")
        
        df_copy = df.copy()
        df_copy['second'] = df_copy['timestamp'].dt.floor('s')
        
        start_time = df['timestamp'].min().floor('s')
        end_time = df['timestamp'].max().ceil('s')
        
        print(f"Time range: {start_time} to {end_time}")
        print(f"Duration: {end_time - start_time}")
        
        # Derive plotting buckets from the existing status column
        # Status values observed in DB: OK, DeadlineExceeded, Internal
        status_map = {
            'OK': 'successful',
            'DeadlineExceeded': 'timeout',
        }
        df_copy['status_bucket'] = df_copy['status'].map(status_map).fillna('error')
        
        counts_by_second = (
            df_copy.groupby(['second', 'status_bucket'])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=['successful', 'timeout', 'error'], fill_value=0)
        )
        
        time_range = pd.date_range(start=start_time, end=end_time, freq='s')
        plot_data = counts_by_second.reindex(time_range, fill_value=0)
        plot_data.index.name = 'time'
        plot_data = plot_data.reset_index()
        
        plot_data['total'] = plot_data['successful'] + plot_data['timeout'] + plot_data['error']
        
        # summary statistics based on status
        total_requests = len(df)
        total_successful = (df['status'] == 'OK').sum()
        total_timeouts = (df['status'] == 'DeadlineExceeded').sum()
        total_errors = total_requests - total_successful - total_timeouts
        if total_requests > 0:
            print(f"Success rate: {total_successful / total_requests * 100:.1f}%")
            print(f"Timeout rate: {total_timeouts / total_requests * 100:.1f}%")
            print(f"Error rate: {total_errors / total_requests * 100:.1f}%")
        
        # 2 plots in 1
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        ax1.fill_between(plot_data['time'], 0, plot_data['successful'], 
                        label='Successful', color='green', alpha=0.7)
        ax1.fill_between(plot_data['time'], plot_data['successful'], 
                        plot_data['successful'] + plot_data['timeout'],
                        label='Timeout', color='orange', alpha=0.7)
        ax1.fill_between(plot_data['time'], plot_data['successful'] + plot_data['timeout'],
                        plot_data['total'],
                        label='Error', color='red', alpha=0.7)
        
        """ ax1.plot(plot_data['time'], plot_data['total'], 
                color='black', linestyle='-', linewidth=2, label='Total Requests') """
        
        ax1.set_title('Leaf Node Throughput: Successful Requests, Timeouts, and Errors Over Time', 
                    fontsize=14, fontweight='bold')
        ax1.set_ylabel('Requests Per Second', fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
    
        # only successful requests for latency scatter
        successful_df = df[df['status'] == 'OK'].copy()
        if not successful_df.empty:
            sns.scatterplot(data=successful_df, x='timestamp', y='grpc_req_duration', 
                        hue='image_tag', palette=IMAGE_PALETTE, alpha=0.6, ax=ax2)
        ax2.set_title("Request Latency Over Time (Successful Requests Only)", fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Latency (ms)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if self.show:
            plt.show()
        if self.save:
            plt.savefig(os.path.join(self.save_path, f"{self.prefix}throughput_leaf_node.png"))
        plt.close()

    @measure_time
    def plot_decomposed_latency(self, df: pd.DataFrame):
        """Plot the distribution of latency sources decomposed by image tag using boxplots"""
        print("Plotting decomposed latency distribution...")
        
        df_clean = df.copy()
        
        if 'functionprocessingtime' in df_clean.columns:
            df_clean['function_processing_ms'] = pd.to_timedelta(df_clean['functionprocessingtime'], errors='coerce').dt.total_seconds() * 1000
        else:
            df_clean['function_processing_ms'] = 0
        
        required_cols = ['scheduling_latency_ms', 'leaf_to_worker_latency_ms', 'function_processing_latency_ms', 'image_tag']
        df_clean = df_clean.dropna(subset=required_cols)
        
        if df_clean.empty:
            print("No data available for latency decomposition plot")
            return
        
        latency_cols = ['scheduling_latency_ms', 'leaf_to_worker_latency_ms', 'function_processing_latency_ms']
        df_melted = df_clean.melt(
            id_vars=['image_tag'], 
            value_vars=latency_cols,
            var_name='latency_type', 
            value_name='latency_ms'
        )
        
        latency_labels = {
            'scheduling_latency_ms': 'Scheduling',
            'leaf_to_worker_latency_ms': 'Leaf-to-Worker',
            'function_processing_latency_ms': 'Function Processing'
        }
        df_melted['latency_type'] = df_melted['latency_type'].map(latency_labels)
        
        plt.figure(figsize=(14, 8))
        sns.boxplot(
            data=df_melted,
            x='image_tag',
            y='latency_ms',
            hue='latency_type',
            palette='Set2'
        )
        
        plt.title('Latency Distribution by Image Tag and Component', fontsize=14, fontweight='bold')
        plt.xlabel('Image Tag', fontsize=12)
        plt.ylabel('Latency (ms)', fontsize=12)
        plt.ylim(0,df_melted['latency_ms'].quantile(0.9))
        plt.legend(title='Latency Component', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        print("\nLatency Distribution Summary:")
        summary_stats = df_melted.groupby(['image_tag', 'latency_type'])['latency_ms'].agg(['count', 'mean', 'median', 'std']).round(2)
        print(summary_stats)
        
        if self.show:
            plt.show()
        if self.save:
            os.makedirs(self.save_path, exist_ok=True)
            plt.savefig(os.path.join(self.save_path, f"{self.prefix}decomposed_latency.png"))
        plt.close()

    @measure_time
    def plot_expected_rps(self, df: pd.DataFrame, scenarios_path: str = None):
        """Plot expected RPS over time for each function image."""
        if df.empty:
            print("No data to plot")
            return
        
        plt.figure(figsize=(15, 8))
        
        # Create the plot for individual functions
        sns.lineplot(data=df, x='second', y='expected_rps', hue='image_tag', 
                    marker='o', markersize=4, palette=IMAGE_PALETTE)
        
        # Add total RPS line
        total_rps = df.groupby('second')['expected_rps'].sum().reset_index()
        sns.lineplot(data=total_rps, x='second', y='expected_rps', 
                    color='black', linestyle='--', label='Total RPS',
                    linewidth=2)
        
        plt.title('Expected Request Rate Over Time (from k6 Scenarios)', fontsize=14, fontweight='bold')
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Expected Requests Per Second', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Function Image', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add scenario information as subtitle if path provided
        if scenarios_path:
            with open(scenarios_path, 'r') as f:
                metadata = json.load(f)['metadata']
            plt.suptitle(f"Total Duration: {metadata['totalDuration']}, Seed: {metadata['seed']}", 
                        fontsize=10, y=0.98)
        
        plt.tight_layout()
        if self.show:
            plt.show()
        if self.save:
            plt.savefig(os.path.join(self.save_path, f"{self.prefix}expected_rps.png"))
        plt.close()
        
        # Print summary statistics
        print("\nExpected RPS Summary by Image Tag:")
        summary = df.groupby('image_tag')['expected_rps'].agg(['mean', 'max', 'sum']).round(2)
        summary.columns = ['avg_rps', 'peak_rps', 'total_requests']
        print(summary)
        print(f"\nTotal expected requests across all functions: {df['expected_rps'].sum():.0f}")
    
    def _save_or_show(self, filename: str):
        """Helper to either display or save the current matplotlib figure respecting class options."""
        if self.show:
            plt.show()
        if self.save:
            os.makedirs(self.save_path, exist_ok=True)
            plt.savefig(os.path.join(self.save_path, f"{self.prefix}{filename}.png"))
        plt.close()

    @measure_time
    def plot_latency_rps_comparison(
        self,
        runs: Mapping[str, pd.DataFrame]
    ):
        """
        Overlay latency CDF and per‐second throughput for an arbitrary number of runs.
        `runs` is a dict mapping run_label (e.g. "orig", "model", "v2") → DataFrame.
        """
        df_all = self._prepare_multi_run_data(runs)
        df_l = self._filter_valid_latency_data(df_all)

        plt.figure(figsize=(15, 6))
        sns.ecdfplot(
            data=df_l,
            x="grpc_req_duration",
            hue="worker_type",
            palette=RUN_PALETTE,
            linewidth=2
        )
        plt.xlabel("End-to-end latency (ms)")
        plt.ylabel("ECDF")
        plt.title("Latency distribution – " + ", ".join(runs.keys()))
        self._save_or_show("latency_comparison_multi")

        df_rps = (
            df_l
            .assign(second=lambda d: d["timestamp"].dt.floor("s"))
            .groupby(["worker_type", "second"])
            .size()
            .reset_index(name="rps")
        )

        plt.figure(figsize=(15, 6))
        sns.lineplot(
            data=df_rps,
            x="second",
            y="rps",
            hue="worker_type",
            palette=RUN_PALETTE,
            linewidth=2
        )
        plt.ylabel("Requests / s")
        plt.title("Throughput over time – " + ", ".join(runs.keys()))
        self._save_or_show("rps_comparison_multi")

    @measure_time
    def plot_latency_ecdf_per_image(
        self,
        runs: Mapping[str, pd.DataFrame],
        cols: int = 3
    ):
        """
        Small‐multiples ECDF per image_tag, comparing N runs.
        `cols` controls how many panels per row.
        """
        # This function uses "run" instead of "worker_type", so we need special handling
        tagged = []
        for label, df in runs.items():
            df2 = df.copy()
            df2["run"] = label
            tagged.append(df2)
        df_all = pd.concat(tagged, ignore_index=True, sort=False)
        df_all = self._filter_valid_latency_data(df_all, ensure_datetime=False)

        g = sns.FacetGrid(
            df_all,
            col="image_tag",
            hue="run",
            palette=RUN_PALETTE,
            col_wrap=cols,
            sharex=True,
            sharey=True,
            height=4,
            aspect=1.2
        )
        g.map(sns.ecdfplot, "grpc_req_duration")
        g.add_legend(title="Run")
        g.set_axis_labels("Latency (ms)", "ECDF")
        g.set_titles("{col_name}")
        plt.subplots_adjust(top=0.88)
        g.fig.suptitle("Latency ECDF per Image – " + ", ".join(runs.keys()), fontsize=14)

        if self.show:
            plt.show()
        if self.save:
            os.makedirs(self.save_path, exist_ok=True)
            out = os.path.join(self.save_path, f"{self.prefix}latency_ecdf_per_image_multi.png")
            g.fig.savefig(out)
        plt.close(g.fig)

    @measure_time
    def plot_cpu_usage_total(self, runs: Mapping[str, pd.DataFrame]):
        """Plot the total CPU usage of the system."""
        print("Plotting total CPU usage...")
        
        df_all = self._prepare_multi_run_data(runs)

        df_cpu = df_all[df_all["cpu_usage_total"].notna()].copy()
        df_cpu["cpu_usage_total"] = df_cpu["cpu_usage_total"].astype(float)

        df_cpu_sum = (
            df_cpu
            .assign(second=lambda d: d["timestamp"].dt.floor("s"))
            .groupby(["worker_type", "second"])
            .agg({"cpu_usage_total": "sum"})
            .reset_index()
            .rename(columns={"cpu_usage_total": "cpu_usage_total_sum"})
        )

        plt.figure(figsize=(15, 6))
        sns.lineplot(data=df_cpu_sum, x="second", y="cpu_usage_total_sum", hue="worker_type", palette=RUN_PALETTE, linewidth=2)
        plt.ylabel("CPU Usage (Nanoseconds)")
        plt.title("Total CPU Usage Over Time – " + ", ".join(runs.keys()))
        self._save_or_show("cpu_usage_total_multi")
       
    @measure_time
    def plot_memory_usage_total(self, runs: Mapping[str, pd.DataFrame]):
        """Plot the total memory usage of the system."""
        print("Plotting total memory usage...")
        
        df_all = self._prepare_multi_run_data(runs)

        df_memory = df_all[df_all["memory_usage"].notna()].copy()
        df_memory["memory_usage"] = df_memory["memory_usage"].astype(float) / 1024 / 1024 # convert to MB

        df_memory_sum = (
            df_memory
            .assign(second=lambda d: d["timestamp"].dt.floor("s"))
            .groupby(["worker_type", "second"])
            .agg({"memory_usage": "sum"})
            .reset_index()
            .rename(columns={"memory_usage": "memory_usage_sum"})
        )
        
        plt.figure(figsize=(15, 6))
        sns.lineplot(data=df_memory_sum, x="second", y="memory_usage_sum", hue="worker_type", palette=RUN_PALETTE, linewidth=2)
        plt.ylabel("Memory Usage (MB)")
        plt.title("Total Memory Usage Over Time – " + ", ".join(runs.keys()))
        self._save_or_show("memory_usage_total_multi")

    @measure_time
    def plot_latency_distribution(self, runs: Mapping[str, pd.DataFrame]):
        """Plot the latency distribution of the system."""
        print("Plotting latency distribution...")
        
        df_all = self._prepare_multi_run_data(runs)
        df_all = self._filter_valid_latency_data(df_all, ensure_datetime=False)
        
        plt.figure(figsize=(15, 6))
        sns.kdeplot(data=df_all, x="grpc_req_duration", hue="worker_type", palette=RUN_PALETTE, bw_adjust=0.25)

        plt.xlabel("Latency (ms)")
        plt.ylabel("Density")
        plt.title("Latency Distribution – " + ", ".join(runs.keys()))
        self._save_or_show("latency_distribution_multi")

    @measure_time
    def plot_latency_distribution_per_image(self, runs: Mapping[str, pd.DataFrame]):
        """Plot the latency distribution per image_tag."""
        print("Plotting latency distribution per image_tag...")

        df_all = self._prepare_multi_run_data(runs)
        df_all = self._filter_valid_latency_data(df_all, ensure_datetime=False)

        g = sns.FacetGrid(
            df_all,
            col="image_tag",
            hue="worker_type",
            palette=RUN_PALETTE,
            col_wrap=3,
            sharex=True,
            sharey=True,
            height=4,
            aspect=1.2,
            xlim=(0,100)
        )
        g.map(sns.kdeplot, "grpc_req_duration", bw_adjust=0.25)
        g.add_legend(title="Run")
        g.set_axis_labels("Latency (ms)", "Density")
        g.set_titles("{col_name}")
        plt.subplots_adjust(top=0.88)
        g.fig.suptitle("Latency Distribution per Image – " + ", ".join(runs.keys()), fontsize=14)

        if self.show:
            plt.show()
        if self.save:
            os.makedirs(self.save_path, exist_ok=True)
            out = os.path.join(self.save_path, f"{self.prefix}latency_distribution_per_image_multi.png")
            g.fig.savefig(out)
        plt.close(g.fig)

    @measure_time
    def plot_latency_time_comparison(
        self,
        runs: Mapping[str, pd.DataFrame]
    ):
        """
        Latency line plot over the time of the workload by image_tag.
        `runs` is a dict mapping run_label (e.g. "orig", "model", "v2") → DataFrame.
        """
        df_all = self._prepare_multi_run_data(runs)
        df_l = self._filter_valid_latency_data(df_all)
        if df_l.empty:
            print("[WARNING] No data to plot for latency time comparison (df_l is empty). Skipping plot.")
            return
        df_l = (
            df_l
            .assign(second=lambda d: d["timestamp"].dt.floor("s"))
            .groupby(["worker_type", "second", "image_tag"])
            .agg({"grpc_req_duration": "median"})
            .reset_index()
        )

        plt.figure(figsize=(15, 6))
        g = sns.FacetGrid(
            df_l,
            col="image_tag",
            hue="worker_type",
            palette=RUN_PALETTE,
            col_wrap=3,
            sharex=True,
            sharey=True,
            height=4,
            aspect=1.2
        )
        g.map(sns.lineplot, "second", "grpc_req_duration", linewidth=2)
        g.add_legend(title="Run")
        g.set_axis_labels("Time (s)", "Latency (ms)")
        g.set_titles("{col_name}")
        plt.subplots_adjust(top=0.88)
        g.fig.suptitle("Latency over time – " + ", ".join(runs.keys()), fontsize=14)
        self._save_or_show("latency_time_comparison_multi")