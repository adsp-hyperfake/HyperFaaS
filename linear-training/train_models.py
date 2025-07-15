#!/usr/bin/env python3

import sqlite3
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings

def main():
    parser = argparse.ArgumentParser(description='Train linear regression models for function performance prediction')
    parser.add_argument('--db', required=True, help='Path to SQLite database file')
    parser.add_argument('--output', default='models.json', help='Output path for trained models')
    args = parser.parse_args()
    
    # Connect to database
    conn = sqlite3.connect(args.db)
    
    try:
        # Get unique image tags
        image_tags = get_image_tags(conn)
        print(f"Found {len(image_tags)} unique image tags")
        
        models = {}
        
        for image_tag in image_tags:
            print(f"\nTraining model for image tag: {image_tag}")
            
            # Load training data
            data = load_training_data(conn, image_tag)
            
            if len(data) < 10:  # Need minimum samples
                print(f"  Warning: insufficient data for {image_tag} (only {len(data)} samples)")
                continue
            
            # Train models
            model = train_model(image_tag, data)
            if model:
                models[image_tag] = model
                print(f"  Successfully trained with {len(data)} samples")
                print(f"  Runtime R²: {model['runtime_r2']:.3f}")
                print(f"  CPU R²: {model['cpu_r2']:.3f}")
                print(f"  RAM R²: {model['ram_r2']:.3f}")
        
        # Save models
        save_models(models, args.output)
        print(f"\nSuccessfully trained {len(models)} models and saved to {args.output}")
        
    finally:
        conn.close()

def get_image_tags(conn):
    """Get unique function image tags from training data"""
    query = """
        SELECT DISTINCT function_image_tag 
        FROM training_data 
        WHERE function_image_tag IS NOT NULL AND function_image_tag != ''
    """
    cursor = conn.execute(query)
    return [row[0] for row in cursor.fetchall()]

def load_training_data(conn, image_tag):
    """Load training data for a specific image tag"""
    query = """
        SELECT 
            request_body_size,
            function_instances_count,
            active_function_calls_count,
            worker_cpu_usage,
            worker_ram_usage,
            function_runtime,
            function_cpu_usage,
            function_ram_usage
        FROM training_data 
        WHERE function_image_tag = ? 
        AND request_body_size IS NOT NULL 
        AND function_instances_count IS NOT NULL
        AND active_function_calls_count IS NOT NULL
        AND worker_cpu_usage IS NOT NULL
        AND worker_ram_usage IS NOT NULL
        AND function_runtime IS NOT NULL
        AND function_cpu_usage IS NOT NULL
        AND function_ram_usage IS NOT NULL
        AND function_runtime > 0
    """
    
    df = pd.read_sql_query(query, conn, params=(image_tag,))
    return df

def train_model(image_tag, data):
    """Train linear regression models for runtime, CPU, and RAM prediction"""
    try:
        # Features: [body_size, instances, active_calls, worker_cpu, worker_ram]
        feature_columns = [
            'request_body_size',
            'function_instances_count', 
            'active_function_calls_count',
            'worker_cpu_usage',
            'worker_ram_usage'
        ]
        
        X = data[feature_columns].values
        
        # Targets (convert runtime from nanoseconds to milliseconds)
        y_runtime = data['function_runtime'].values / 1e6  # ns to ms
        y_cpu = data['function_cpu_usage'].values
        y_ram = data['function_ram_usage'].values
        
        # Check for variance in features
        if np.var(X, axis=0).min() < 1e-10:
            print(f"    Warning: Some features have very low variance")
        
        # Scale features for better numerical stability
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train three separate models
        models = {}
        r2_scores = {}
        
        # Runtime model
        runtime_model = LinearRegression()
        runtime_model.fit(X_scaled, y_runtime)
        y_runtime_pred = runtime_model.predict(X_scaled)
        r2_scores['runtime'] = r2_score(y_runtime, y_runtime_pred)
        
        # CPU model  
        cpu_model = LinearRegression()
        cpu_model.fit(X_scaled, y_cpu)
        y_cpu_pred = cpu_model.predict(X_scaled)
        r2_scores['cpu'] = r2_score(y_cpu, y_cpu_pred)
        
        # RAM model
        ram_model = LinearRegression()
        ram_model.fit(X_scaled, y_ram)
        y_ram_pred = ram_model.predict(X_scaled)
        r2_scores['ram'] = r2_score(y_ram, y_ram_pred)
        
        # Convert back to unscaled coefficients
        # For scaled features: y = w_scaled @ X_scaled + b_scaled
        # For original features: y = w @ X + b where w = w_scaled / scale, b = b_scaled - w_scaled @ mean / scale
        
        runtime_coeffs = runtime_model.coef_ / scaler.scale_
        runtime_intercept = runtime_model.intercept_ - np.dot(runtime_model.coef_, scaler.mean_ / scaler.scale_)
        
        cpu_coeffs = cpu_model.coef_ / scaler.scale_
        cpu_intercept = cpu_model.intercept_ - np.dot(cpu_model.coef_, scaler.mean_ / scaler.scale_)
        
        ram_coeffs = ram_model.coef_ / scaler.scale_
        ram_intercept = ram_model.intercept_ - np.dot(ram_model.coef_, scaler.mean_ / scaler.scale_)
        
        # Create model dict compatible with Go structure
        model = {
            'image_tag': image_tag,
            'runtime_coeffs': list(runtime_coeffs) + [runtime_intercept],  # [body_size, instances, active_calls, worker_cpu, worker_ram, intercept]
            'cpu_coeffs': list(cpu_coeffs) + [cpu_intercept],
            'ram_coeffs': list(ram_coeffs) + [ram_intercept], 
            'sample_count': len(data),
            'runtime_r2': r2_scores['runtime'],
            'cpu_r2': r2_scores['cpu'],
            'ram_r2': r2_scores['ram']
        }
        
        return model
        
    except Exception as e:
        print(f"    Error training model for {image_tag}: {e}")
        return None

def save_models(models, output_path):
    """Save models to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(models, f, indent=2)

if __name__ == "__main__":
    main() 