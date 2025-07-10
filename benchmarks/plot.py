import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
from collections.abc import Mapping

IMAGE_PALETTE = {
    "hyperfaas-bfs-json:latest": "blue",
    "hyperfaas-echo:latest": "red",
    "hyperfaas-thumbnailer-json:latest": "green",
}
class Plotter:
    def __init__(self, show: bool = True, save_path=None, prefix: str = None):
        self.show = show
        self.save_path = save_path
        self.save = bool(self.save_path)
        self.prefix = prefix + "_" if prefix else ""

    def plot_throughput_leaf_node(self, df: pd.DataFrame):
        """Plot the number of requests that completed processing per second at the leaf node level.
        We use requests fully completed as the basis for throughput here.
        """
        print("Plotting requests processed per second at the leaf node level...")
        
        # Define time range based on timestamps
        start_time = df['timestamp'].min().floor('s')
        end_time = df['timestamp'].max().ceil('s')
        time_range = pd.date_range(start=start_time, end=end_time, freq='s')
        
        print(f"Time range: {start_time} to {end_time}")
        print(f"Duration: {end_time - start_time}")
        
        successful_counts = []
        timeout_counts = []
        error_counts = []
        
        for t in time_range:
            requests_in_window = (df['timestamp'] >= t) & (df['timestamp'] < t + pd.Timedelta(seconds=1))
            
            successful_counts.append((requests_in_window & df['grpc_req_duration'].notna()).sum())
            timeout_counts.append((requests_in_window & df['timeout'].notna()).sum())
            error_counts.append((requests_in_window & df['error'].notna()).sum())
        
        plot_data = pd.DataFrame({
            'time': time_range,
            'successful': successful_counts,
            'timeout': timeout_counts,
            'error': error_counts
        })
        
        plot_data['total'] = plot_data['successful'] + plot_data['timeout'] + plot_data['error']
        
        total_sent = df['grpc_req_duration'].notna().sum()
        total_timeouts = df['timeout'].notna().sum()
        total_errors = df['error'].notna().sum()
        total_requests = len(df) 
        total_successful = total_sent - total_timeouts - total_errors
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
    
        successful_df = df[df['grpc_req_duration'].notna()].copy()
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

    def plot_decomposed_latency(self, df: pd.DataFrame):
        """Plot the source of latency of requests decomposed by image tag"""
        print("Plotting decomposed latency...")
        # Function processing time (from the function execution itself)
        df['function_processing_ms'] = pd.to_timedelta(df['functionprocessingtime']).dt.total_seconds() * 1000
        
        # Remove rows with missing data
        df = df.dropna(subset=['scheduling_latency_ms', 'leaf_to_worker_latency_ms', 'function_processing_latency_ms', 'image_tag'])
        
        import seaborn.objects as so
        
        df_melted = df.melt(
            id_vars=['image_tag'], 
            value_vars=['scheduling_latency_ms', 'leaf_to_worker_latency_ms', 'function_processing_latency_ms'],
            var_name='latency_type', 
            value_name='latency_ms'
        )
        p = (
            so.Plot(df_melted, x="image_tag", y="latency_ms", color="latency_type")
            .add(so.Bar(), so.Agg("sum"), so.Norm(func="sum", by=["x"]), so.Stack())
            .layout(size=(12, 6))
            .label(
                title="Latency Decomposition by Image Tag",
                x="Image Tag",
                y="Average Latency (ms)",
                color="Latency Component"
            )
        )
        if self.show:
            p.show()
        if self.save:
            os.makedirs(self.save_path, exist_ok=True)
            p.save(os.path.join(self.save_path, f"{self.prefix}decomposed_latency.png"))
        plt.close()

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

    def plot_latency_rps_comparison(
        self,
        runs: Mapping[str, pd.DataFrame]
    ):
        """
        Overlay latency CDF and per‐second throughput for an arbitrary number of runs.
        `runs` is a dict mapping run_label (e.g. "orig", "model", "v2") → DataFrame.
        """
        tagged = []
        for label, df in runs.items():
            df2 = df.copy()
            if "worker_type" in df2:
                df2["worker_type"] = label
            else:
                df2 = df2.rename_axis(None, axis=1)
                df2["worker_type"] = label
            tagged.append(df2)
        df_all = pd.concat(tagged, ignore_index=True, sort=False)

        df_l = df_all[df_all["grpc_req_duration"].notna()].copy()
        df_l["grpc_req_duration"] = df_l["grpc_req_duration"].astype(float)

        plt.figure(figsize=(15, 6))
        sns.ecdfplot(
            data=df_l,
            x="grpc_req_duration",
            hue="worker_type",
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
            linewidth=2
        )
        plt.ylabel("Requests / s")
        plt.title("Throughput over time – " + ", ".join(runs.keys()))
        self._save_or_show("rps_comparison_multi")

    def plot_latency_ecdf_per_image(
        self,
        runs: Mapping[str, pd.DataFrame],
        cols: int = 3
    ):
        """
        Small‐multiples ECDF per image_tag, comparing N runs.
        `cols` controls how many panels per row.
        """
        tagged = []
        for label, df in runs.items():
            df2 = df.copy()
            df2["run"] = label
            tagged.append(df2)
        df_all = pd.concat(tagged, ignore_index=True, sort=False)

        df_all = df_all[df_all["grpc_req_duration"].notna()].copy()
        df_all["grpc_req_duration"] = df_all["grpc_req_duration"].astype(float)

        g = sns.FacetGrid(
            df_all,
            col="image_tag",
            hue="run",
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
