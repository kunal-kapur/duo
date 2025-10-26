from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

# Path to the model directory (update if your path is different)
model_dir = "/home/ubuntu/kkapur-v2/models/replaced_vocab_roberta_for_jigsaw"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Put model in evaluation mode
model.eval()

# Example sentences to classify
sentences = [
    "I don't like machine learning. Screw it",
    "machine learning is really cool"
]

# Tokenize the inputs
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)

# Extract logits
logits = outputs.logits

preds = torch.argmax(logits, dim=1)

probs = F.softmax(logits, dim=1)

for idx, sent in enumerate(sentences):
    print(f"Sentence: {sent}")
    print(f"  Logits: {logits[idx].tolist()}")
    print(f"  Probabilities: {probs[idx].tolist()}")
    print(f"  Predicted class: {preds[idx].item()}")
    print()


from datasets import load_dataset

ds = load_dataset("allenai/real-toxicity-prompts")

print(ds['train'][0])  # Print the first example from the training set
