import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification

MODEL_PATH = r"C:\Users\NUGGET\mental_health_prediction\models\final_unified_model"

print (f"🧠 Probing model dimensions at: {MODEL_PATH}")

try:
    # Load model
    tokenizer = AutoTokenizer.from_pretrained (MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained (MODEL_PATH)

    # Create a dummy input
    text = "I am feeling test data."
    inputs = tokenizer (text,return_tensors="pt")

    # Run prediction
    with torch.no_grad ():
        outputs = model (**inputs)

    # COUNT THE OUTPUTS
    num_labels = outputs.logits.shape[1]

    print ("\n" + "=" * 40)
    print (f"📊 YOUR MODEL HAS {num_labels} OUTPUTS")
    print ("=" * 40)

    if num_labels == 2:
        print ("Result: This is a BINARY model (likely 'Normal' vs 'Risk').")
        print ("Use: self.labels = ['Normal', 'Risk']")
    elif num_labels == 7:
        print ("Result: This is the 7-CLASS model!")
        print ("Since names weren't saved, standard training usually sorts Alphabetically:")
        print ("['Anxiety', 'Bipolar', 'Depression', 'Normal', 'Personality Disorder', 'Stress', 'Suicidal']")
    else:
        print (f"Result: It has {num_labels} classes. We need to figure out what they are.")

except Exception as e:
    print (f"❌ Error: {e}")