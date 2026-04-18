import json
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

def load_logs(filepath):
    """Reads the JSONL file and converts it into a Pandas DataFrame."""
    data = []
    try:
        with open(filepath, 'r' , encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        return pd.DataFrame()

def main():
    print("1. Loading logs...")
    # Make sure this matches the exact name of your log file
    df = load_logs('qa_monitoring_logs.jsonl')
    
    if df.empty:
        print("Exiting: No data to process.")
        return

    print("2. Generating embeddings for questions...")
    # 'all-MiniLM-L6-v2' is a standard, fast local model for converting text to semantic vectors
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    
    # Extract just the questions into a list
    questions = df['question'].tolist()
    
    # Generate the embeddings (this turns text into a matrix of numbers)
    embeddings = model.encode(questions)
    
    print("3. Performing K-Means Clustering (K=5)...")
    # We use random_state=42 so you get the exact same clusters every time you run it (good for presentations!)
    kmeans = KMeans(n_clusters=5, random_state=42)
    
    # This assigns a cluster number (0, 1, 2, 3, or 4) to every single row
    df['cluster_id'] = kmeans.fit_predict(embeddings)
    
    print("4. Saving clustered data...")
    # We save this to a NEW file so we don't accidentally ruin Siddhesh's original logs
    df.to_json('clustered_logs.jsonl', orient='records', lines=True)
    
    print("\n✅ Success! Created 'clustered_logs.jsonl' with a new 'cluster_id' column.")
    print(df[['question', 'cluster_id']].head()) # Print a quick preview

if __name__ == "__main__":
    main()