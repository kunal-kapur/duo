from transformers import AutoModelForSequenceClassification, AutoTokenizer, RobertaModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# model_dir = "/Users/kunalkapur/Downloads/checkpoints/replaced_vocab_roberta_for_jigsaw"
# tokenizer = AutoTokenizer.from_pretrained(model_dir)

# # Print all special tokens (including mask, if present)
# print("All special tokens:", tokenizer.all_special_tokens)
# print("All special token IDs:", tokenizer.all_special_ids)

# # Check vocabulary size and example tokens
# print("Vocab size:", len(tokenizer))
# for i in range(len(tokenizer)):
#     print(f"{i}: {tokenizer.convert_ids_to_tokens(i)}")

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier')

# Print out the mask token and mask token id
print("Mask token:", tokenizer.mask_token)            # Output: '<mask>'
print("Mask token id:", tokenizer.mask_token_id)      # Example output: 50264

# Example: locate the mask token in a sentence
sentence = "The quick brown <mask> jumps over the lazy dog."
encoded = tokenizer(sentence, return_tensors="pt")
mask_positions = (encoded["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
print(f"Mask positions in input: {mask_positions}")

# Tokenize two sentences as a batch and return PyTorch tensors
batch = tokenizer(["You are amazing!", "You suck. Screw you"], return_tensors="pt", padding=True, truncation=True)
print("Batch", batch['input_ids'].shape)
# Pass model inputs as keyword arguments
output = model(**batch)
print("output:", output)

# text = "Toxic content can appear as <mask>."
# inputs = tokenizer([text], return_tensors='pt')
# positions = (inputs['input_ids'] == mask_token_id).nonzero(as_tuple=True)
# print("Mask positions:", positions)


# Put model in evaluation mode
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


# from datasets import load_dataset

# ds = load_dataset("allenai/real-toxicity-prompts")
# print(ds.keys())
# print(ds.valid)
# train_ds = ds['train']

# # 1. Set the dataset format to 'torch'
# # This tells the dataset to return PyTorch tensors instead of lists
# # train_ds.set_format()

# # 2. Create a DataLoader
# # This will automatically handle batching and shuffling (if you want)
# # batch_size = 32
# # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# # 3. Iterate over the DataLoader
# for batch in train_ds:
#     # 'batch' is a dictionary where each value is a tensor
#     # of shape [batch_size, ...]
    
#     # 'batch['prompt']' will be a list of strings (since text isn't a tensor yet)
#     # 'batch['toxicity']' would be a tensor if it were a numeric feature
    
#     # Print the first prompt in the batch
#     print(batch['prompt']['text']) 
#     print("---")
#     break

# # Don't forget to set the format back if you need to
# train_ds.set_format(type=None)