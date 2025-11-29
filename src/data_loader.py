import pandas as pd
import numpy as np

def load_and_synthesize_data(student_path, reddit_path):
    """
    Merges structured student data with unstructured Reddit text 
    based on implied risk labels to create a Hybrid Dataset.
    """
    # 1. Load Raw Data
    df_student = pd.read_csv(student_path)
    df_reddit = pd.read_csv(reddit_path)

    # 2. Standardize Labels (Example Mapping)
    # Assume Student dataset has 'Depression' (Yes/No) and Reddit has 'label' (0=Stress, 1=Depression)
    # We need them to match.
    
    # Filter Reddit for relevant classes (e.g., keep Depression and Anxiety posts)
    reddit_depressed = df_reddit[df_reddit['label'] == 1]['text'].values
    reddit_stress = df_reddit[df_reddit['label'] == 0]['text'].values
    
    # 3. Synthetic Merge Logic
    hybrid_data =
    
    for index, row in df_student.iterrows():
        # Logic: If student has depression, assign a "Depression" post. 
        # If not, assign a "Stress" or "Normal" post.
        
        if row == 'Yes':
            text_sample = np.random.choice(reddit_depressed)
            risk_level = 'High'
        else:
            text_sample = np.random.choice(reddit_stress)
            risk_level = 'Low' # or 'Moderate' based on other cols
            
        # Combine into one dictionary
        hybrid_row = row.to_dict()
        hybrid_row['journal_text'] = text_sample  # The Unstructured Feature
        hybrid_row = risk_level     # The Target
        
        hybrid_data.append(hybrid_row)
        
    df_hybrid = pd.DataFrame(hybrid_data)
    return df_hybrid

if __name__ == "__main__":
    # Test the function
    df = load_and_synthesize_data('../data/raw/student.csv', '../data/raw/reddit.csv')
    df.to_csv('../data/processed/hybrid_dataset.csv', index=False)
    print("Hybrid Data Synthesized Successfully!")