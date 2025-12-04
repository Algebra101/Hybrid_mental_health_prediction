# Pseudo-code for Hybrid Synthesis
import pandas as pd
import random
from nltk.corpus import wordnet

# 1. Load Data
students = pd.read_csv('data/raw/Student_Mental_Health.csv')
reddit = pd.read_csv('data/raw/Reddit_Mental_Health.csv')

# 2. Pre-Merge Cleaning
reddit = reddit.dropna(subset=['post_text']) # Remove 350 nulls

# 3. Define Risk Mapping Function
def map_student_risk(row):
    if row == 'Yes' and row['Anxiety'] == 'Yes':
        return 3 # High Risk
    elif row == 'Yes':
        return 2 # Moderate
    elif row['Anxiety'] == 'Yes':
        return 1 # Low
    else:
        return 0 # Healthy

def map_reddit_risk(row):
    if row['subreddit'] in:
        return 3
    elif row['subreddit'] == 'anxiety':
        return 2
    elif row['subreddit'] == 'stress':
        return 1
    else:
        return 0

# 4. Apply Mappings
students['risk_label'] = students.apply(map_student_risk, axis=1)
reddit['risk_label'] = reddit.apply(map_reddit_risk, axis=1)

# 5. Create Text Buckets
text_buckets = {
    0: reddit[reddit['risk_label'] == 0]['post_text'].tolist(),
    1: reddit[reddit['risk_label'] == 1]['post_text'].tolist(),
    2: reddit[reddit['risk_label'] == 2]['post_text'].tolist(),
    3: reddit[reddit['risk_label'] == 3]['post_text'].tolist()
}

# 6. Synonym Augmentation Function
def augment_text(text):
    words = text.split()
    # Randomly replace 1 word with a synonym
    idx = random.randint(0, len(words)-1)
    # (Implementation of synonym lookup via WordNet omitted for brevity)
    return " ".join(words)

# 7. The Synthesis Loop
hybrid_data =
for index, student in students.iterrows():
    risk = student['risk_label']
    
    # Stochastic Selection
    # Pick a random text from the matching bucket
    base_text = random.choice(text_buckets[risk])
    
    # Augmentation
    final_text = augment_text(base_text)
    
    # Create merged record
    new_record = student.to_dict()
    new_record['journal_text'] = final_text
    hybrid_data.append(new_record)

# 8. Save
pd.DataFrame(hybrid_data).to_csv('data/processed/hybrid_final.csv')