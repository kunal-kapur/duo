from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
# # Path to the model directory (update if your path is different)
# model_dir = "/home/ubuntu/kkapur-v2/models/replaced_vocab_roberta_for_jigsaw"

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(model_dir)
# model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# # Put model in evaluation mode
# model.eval()

# # Example sentences to classify
# sentences = [
#     "I don't like machine learning. Screw it",
#     "machine learning is really cool"
# ]

# # Tokenize the inputs
# inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

# with torch.no_grad():
#     outputs = model(**inputs)

# # Extract logits
# logits = outputs.logits

# preds = torch.argmax(logits, dim=1)

# probs = F.softmax(logits, dim=1)

# for idx, sent in enumerate(sentences):
#     print(f"Sentence: {sent}")
#     print(f"  Logits: {logits[idx].tolist()}")
#     print(f"  Probabilities: {probs[idx].tolist()}")
#     print(f"  Predicted class: {preds[idx].item()}")
#     print()


from datasets import load_dataset

ds = load_dataset("allenai/real-toxicity-prompts")
print(ds.keys())
print(ds.valid)
train_ds = ds['train']

# 1. Set the dataset format to 'torch'
# This tells the dataset to return PyTorch tensors instead of lists
# train_ds.set_format()

# 2. Create a DataLoader
# This will automatically handle batching and shuffling (if you want)
# batch_size = 32
# train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# 3. Iterate over the DataLoader
for batch in train_ds:
    # 'batch' is a dictionary where each value is a tensor
    # of shape [batch_size, ...]
    
    # 'batch['prompt']' will be a list of strings (since text isn't a tensor yet)
    # 'batch['toxicity']' would be a tensor if it were a numeric feature
    
    # Print the first prompt in the batch
    print(batch['prompt']['text']) 
    print("---")
    break

# Don't forget to set the format back if you need to
train_ds.set_format(type=None)