import pandas as pd
import json

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
    print("Loading clustered logs...\n")
    df = load_logs('clustered_logs.jsonl')
    
    if df.empty:
        print("Exiting: No data to process.")
        return

    print("="*50)
    print(" 📊 TOPIC CLUSTERING & ANALYSIS REPORT ")
    print("="*50 + "\n")
    
    # Analyze each cluster one by one
    for cluster_id in sorted(df['cluster_id'].unique()):
        cluster_data = df[df['cluster_id'] == cluster_id]
        
        # Calculate metrics
        mean_unc = cluster_data['uncertainty_score'].mean()
        std_unc = cluster_data['uncertainty_score'].std() # Standard Deviation
        
        mean_con = cluster_data['consistency_score'].mean()
        std_con = cluster_data['consistency_score'].std()
        
        num_questions = len(cluster_data)
        
        print(f"🔹 CLUSTER {cluster_id} ({num_questions} interactions)")
        
        # We use .fillna(0) in case a cluster only has 1 question (which makes std NaN)
        print(f"   ➤ Mean Uncertainty: {mean_unc:.3f} (± {pd.Series([std_unc]).fillna(0)[0]:.3f})")
        print(f"   ➤ Mean Consistency: {mean_con:.3f} (± {pd.Series([std_con]).fillna(0)[0]:.3f})")
        
        # Find the worst questions (Highest uncertainty, lowest consistency)
        worst_df = cluster_data.sort_values(by=['uncertainty_score', 'consistency_score'], ascending=[False, True])
        
        print("   ⚠️ Highest Risk Questions in this Topic:")
        # Grab up to the top 2 unique worst questions
        worst_questions = worst_df['question'].unique()[:2]
        for q in worst_questions:
            print(f"      - \"{q}\"")
            
        print("-" * 50)

if __name__ == "__main__":
    main()