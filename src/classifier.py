import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForSequenceClassification


class MentalHealthClassifier:
    def __init__(self,model_path):
        """
        Initializes the 'Doctor' (Classification Model).
        """
        self.device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
        # Load the model specifically for CLASSIFICATION (finding labels)
        self.tokenizer = AutoTokenizer.from_pretrained ("roberta-base")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to (self.device)
        self.model.eval ()  # Set to evaluation mode (no training)

        # labels
        self.labels = ["Normal", "Mental Health Concern"]

    def predict(self,text):
        """
        Analyzes the text and returns The predicted condition
        """
        inputs = self.tokenizer (text,return_tensors="pt",truncation=True,max_length=128,padding=True).to (self.device)

        with torch.no_grad ():
            outputs = self.model (**inputs)

        # Convert logits to probabilities (0% to 100%)
        probs = F.softmax (outputs.logits,dim=1)

        # Get the highest probability and its index
        risk_score,predicted_class_idx = torch.max (probs,dim=1)

        prediction = self.labels[predicted_class_idx.item ()]
        score = risk_score.item ()

        return prediction,score