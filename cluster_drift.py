import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_logs(filepath):
    """Loads the clustered JSONL file."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        return pd.DataFrame()

def main():
    print("1. Loading clustered logs...")
    df = load_logs('clustered_logs.jsonl')

    if df.empty:
        print("Exiting: No data to process.")
        return

    # Sort by timestamp so we can accurately split "Past" vs "Present"
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    print("2. Re-computing embeddings for precise centroid math...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Generate embeddings and attach them directly to the dataframe
    df['embedding'] = list(model.encode(df['question'].tolist()))

    print("3. Simulating Time Windows (Chronological Split)...")
    # We split the dataset down the middle to simulate Window A (Past) and Window B (Recent)
    midpoint = len(df) // 2
    window_a = df.iloc[:midpoint]
    window_b = df.iloc[midpoint:]

    print("\n==================================================")
    print(" 🚀 ADVANCED CLUSTER DRIFT DETECTION")
    print("==================================================\n")

    # If the drift score goes above this, we trigger an alert
    ALERT_THRESHOLD = 0.15 

    for cluster_id in sorted(df['cluster_id'].unique()):
        cluster_a = window_a[window_a['cluster_id'] == cluster_id]
        cluster_b = window_b[window_b['cluster_id'] == cluster_id]

        # We need at least a few questions in both time windows to do math
        if len(cluster_a) < 2 or len(cluster_b) < 2:
            print(f"🔹 CLUSTER {cluster_id}: Insufficient data across both time windows.")
            print("-" * 50)
            continue

        # 1. Calculate Centroids (The mathematical center of the cluster in each time window)
        centroid_a = np.mean(np.vstack(cluster_a['embedding'].values), axis=0)
        centroid_b = np.mean(np.vstack(cluster_b['embedding'].values), axis=0)

        # 2. Measure Cosine Similarity between the two centroids
        similarity = cosine_similarity([centroid_a], [centroid_b])[0][0]
        
        # 3. Drift is the opposite of similarity
        drift_score = 1 - similarity

        print(f"🔹 CLUSTER {cluster_id}: Drift Score = {drift_score:.4f}")
        
        if drift_score > ALERT_THRESHOLD:
            print(f"   🚨 ALERT: Significant behavioral drift detected in Topic {cluster_id}!")
            print(f"      Topic shifted from e.g., '{cluster_a['question'].iloc[0]}'")
            print(f"                        to '{cluster_b['question'].iloc[0]}'")
        else:
            print("   ✅ Stable (No significant semantic shift)")
            
        print("-" * 50)

if __name__ == "__main__":
    main()