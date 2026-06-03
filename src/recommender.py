import pandas as pd
import torch
import os
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class MentalHealthRecommender:
    def __init__(self,model_path,resource_file="resources.csv"):
        # If resource_file is just a filename, force it to look in the same folder as this script
        if not os.path.isabs (resource_file):
            current_dir = os.path.dirname (os.path.abspath (__file__))
            resource_file = os.path.join (current_dir,resource_file)

        self.df = pd.read_csv(resource_file)
        # We load the model as a 'base' model to get embeddings, not classification
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.model = AutoModel.from_pretrained (model_path)
        # We pre-calculate the "meaning" vector for every resource description
        self.resource_embeddings = self._batch_encode (self.df["Description"].tolist())



    def _batch_encode(self,texts):
        # Converts a list of text strings into mathematical vectors (Embeddings).
        inputs = self.tokenizer (texts,return_tensors="pt",padding=True,truncation=True,max_length=128)

        with torch.no_grad ():
            outputs = self.model(**inputs)
        # the 'CLS' token (the first token) as the representation of the whole sentence
        embeddings = outputs.last_hidden_state[:,0,:]
        return embeddings.numpy()



    def get_recommendations(self,student_text,risk_score):
        results = []
        # SAFETY GUARDRAILS
        # If the student is high risk (> 0.90), we FORCE crisis resources.
        if risk_score > 0.90:
            print (" CRITICAL RISK DETECTED. Activating Safety Protocol.")
            # Filter: Only show resources with Risk_Level_Allowed = 1 (Crisis)
            crisis_df = self.df[self.df["Risk_Level_Allowed"] == 1]
            return crisis_df.to_dict (orient="records")

        # SEMANTIC MATCHING
        # If risk is manageable, we look for the best helpful content.
        # Filter: Only show safe resources (Risk_Level_Allowed = 0)
        safe_df = self.df[self.df["Risk_Level_Allowed"] == 0].copy()
        safe_indices = safe_df.index.tolist ()
        safe_vectors = self.resource_embeddings[safe_indices]

        # Vectorize the Student's specific complaint [cite: 25]
        student_vector = self._batch_encode ([student_text])

        # Calculate Similarity (How close is the student's problem to the solution?) [cite: 28]
        # We compare the student_vector to ALL safe resource vectors
        scores = cosine_similarity (student_vector,safe_vectors)[0]

        # Rank and Pick Top 3 [cite: 30]
        # Get the indices of the top 3 highest scores
        top_k_indices = scores.argsort ()[-3:][::-1]

        # Package the results
        recommendations = safe_df.iloc[top_k_indices].to_dict (orient="records")

        return recommendations


