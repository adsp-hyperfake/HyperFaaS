import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

IMAGE_PALETTE = {
    "hyperfaas-bfs-json:latest": "blue",
    "hyperfaas-echo:latest": "red",
    "hyperfaas-thumbnailer-json:latest": "green",
}

def prepare_dataframe(df: pd.DataFrame, use_gotresponsetimestamp=True):
    """Helper function to prepare dataframe with timestamp conversions"""
    # Drop rows with missing required values
    if use_gotresponsetimestamp:
        df = df[df['callqueuedtimestamp'].notna() & df['gotresponsetimestamp'].notna()].copy()
    else:
        df = df[df['callqueuedtimestamp'].notna() & df['functionprocessingtime'].notna()].copy()
    
    # Convert timestamps from nanoseconds to datetime
    df['callqueuedtimestamp'] = pd.to_datetime(df['callqueuedtimestamp'], unit='ns')
    df['leafgotrequesttimestamp'] = pd.to_datetime(df['leafgotrequesttimestamp'], unit='ns')
    df['leafscheduledcalltimestamp'] = pd.to_datetime(df['leafscheduledcalltimestamp'], unit='ns')
    df['gotresponsetimestamp'] = pd.to_datetime(df['gotresponsetimestamp'], unit='ns')
    # Add latency
    df['scheduling_latency_ms'] = (df['leafscheduledcalltimestamp'] - df['leafgotrequesttimestamp']).dt.total_seconds() * 1000
    df['leaf_to_worker_latency_ms'] = (df['callqueuedtimestamp'] - df['leafscheduledcalltimestamp']).dt.total_seconds() * 1000
    df['function_processing_latency_ms'] = (df['gotresponsetimestamp'] - df['callqueuedtimestamp']).dt.total_seconds() * 1000
    return df

def plot_requests_processed_per_second(df: pd.DataFrame):
    """Plot the number of requests that completed processing per second"""
    print("Plotting requests processed per second...")
    
    # Prepare dataframe
    df = prepare_dataframe(df, use_gotresponsetimestamp=True)
    
    # Debug: Print actual time range
    raw_start = df['callqueuedtimestamp'].min()
    raw_end = df['gotresponsetimestamp'].max()
    print(f"Processed start time: {raw_start}")
    print(f"Processed end time: {raw_end}")
    print(f"Time difference: {raw_end - raw_start}")
    print(f"Total requests: {len(df)}")
    
    # Define time range per second based on when requests completed
    start_time = df['gotresponsetimestamp'].min().floor('s')
    end_time = df['gotresponsetimestamp'].max().ceil('s')
    
    print(f"Floored start time: {start_time}")
    print(f"Ceiled end time: {end_time}")
    print(f"Expected duration: {end_time - start_time}")
    
    # Create time range
    time_range = pd.date_range(start=start_time, end=end_time, freq='s')
    print(f"Time range length: {len(time_range)} seconds")
    
    # Count requests that completed in each second
    processed_counts = [
        ((df['gotresponsetimestamp'] > t) & (df['gotresponsetimestamp'] <= t + pd.Timedelta(seconds=1))).sum()
        for t in time_range
    ]
    
    # Create and plot the series
    series = pd.Series(processed_counts, index=time_range, name='processed_rps')
    print(f"\nProcessed requests stats:")
    print(f"Min: {series.min()}, Max: {series.max()}, Mean: {series.mean():.2f}")
    print(f"Total processed: {series.sum()}")
    
    plt.figure(figsize=(12, 5))
    sns.lineplot(x=series.index, y=series.values)
    plt.title("Requests Processed Per Second")
    plt.xlabel("Time")
    plt.ylabel("Number of Requests Completed")
    plt.tight_layout()
    plt.show()

def plot_throughput_vs_latency_over_time(df: pd.DataFrame):
    """Plot requests processed per second and latency over time in stacked subplots"""
    print("Plotting throughput vs latency over time...")
    
    # Prepare dataframe
    df = prepare_dataframe(df, use_gotresponsetimestamp=True)
    
    # Calculate requests processed per second
    start_time = df['gotresponsetimestamp'].min().floor('s')
    end_time = df['gotresponsetimestamp'].max().ceil('s')
    time_range = pd.date_range(start=start_time, end=end_time, freq='s')
    
    processed_counts = [
        ((df['gotresponsetimestamp'] > t) & (df['gotresponsetimestamp'] <= t + pd.Timedelta(seconds=1))).sum()
        for t in time_range
    ]
    
    throughput_series = pd.Series(processed_counts, index=time_range, name='processed_rps')
    
    # Create stacked subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Top plot: Requests processed per second
    sns.lineplot(x=throughput_series.index, y=throughput_series.values, ax=ax1)
    ax1.set_title("Requests Processed Per Second")
    ax1.set_ylabel("Requests/Second")
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Latency scatter plot
    df['latency'] = df['grpc_req_duration']
    sns.scatterplot(data=df, x='callqueuedtimestamp', y='latency', hue='image_tag', 
                   palette=IMAGE_PALETTE, alpha=0.6, ax=ax2)
    ax2.set_title("Request Latency Over Time")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Latency (ms)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print correlation info
    print(f"\nThroughput Statistics:")
    print(f"Mean RPS: {throughput_series.mean():.2f}")
    print(f"Max RPS: {throughput_series.max()}")
    print(f"Min RPS: {throughput_series.min()}")

def plot_decomposed_latency(df: pd.DataFrame):
    """Plot the source of latency of requests decomposed by image tag"""
    print("Plotting decomposed latency...")
    
    # Prepare dataframe
    df = prepare_dataframe(df, use_gotresponsetimestamp=True)
    
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
    (
        so.Plot(df_melted, x="image_tag", y="latency_ms", color="latency_type")
        .add(so.Bar(), so.Agg("sum"), so.Norm(func="sum", by=["x"]), so.Stack())
        .layout(size=(12, 6))
        .label(
            title="Latency Decomposition by Image Tag",
            x="Image Tag",
            y="Average Latency (ms)",
            color="Latency Component"
        )
        .show()
    )

def plot_expected_rps(df: pd.DataFrame, scenarios_path: str = None):
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
    plt.show()
    
    # Print summary statistics
    print("\nExpected RPS Summary by Image Tag:")
    summary = df.groupby('image_tag')['expected_rps'].agg(['mean', 'max', 'sum']).round(2)
    summary.columns = ['avg_rps', 'peak_rps', 'total_requests']
    print(summary)
    print(f"\nTotal expected requests across all functions: {df['expected_rps'].sum():.0f}")