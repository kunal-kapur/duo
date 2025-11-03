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

# tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
# model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier')

model_dir = "/home/ubuntu/kkapur-v2/models/replaced_vocab_roberta_for_jigsaw"
tokenizer = AutoTokenizer.from_pretrained(model_dir)

print(tokenizer.vocab_size)
print(tokenizer.unk_token_id)
print("unk token", tokenizer.unk_token)
print(tokenizer.convert_ids_to_tokens(0))
print(tokenizer.convert_ids_to_tokens(50255))
print(tokenizer.convert_ids_to_tokens(50256))
print(tokenizer.convert_ids_to_tokens(50257))

# # Print out the mask token and mask token id
# print("Mask token:", tokenizer.mask_token)            # Output: '<mask>'
# print("Mask token id:", tokenizer.mask_token_id)      # Example output: 50264

# # Example: locate the mask token in a sentence
# sentence = "The quick brown <mask> jumps over the lazy dog."
# encoded = tokenizer(sentence, return_tensors="pt")
# mask_positions = (encoded["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
# print(f"Mask positions in input: {mask_positions}")

# # Tokenize two sentences as a batch and return PyTorch tensors
# batch = tokenizer(["You are amazing!", "You suck. Screw you"], return_tensors="pt", padding=True, truncation=True)
# print("Batch", batch['input_ids'].shape)
# # Pass model inputs as keyword arguments
# output = model(**batch)
# print("output:", output)



import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification

import torch
import torch.nn.functional as F

class Toxicity:
    def __init__(self, tokenizer=None , device='cuda'):
        if tokenizer is None:
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier')
        self.tokenizer = tokenizer
        self.device = device
        self.model = model.to(self.device)
        self.device = self.device
        self.vocab_size = tokenizer.vocab_size
        self.embed = self.model.get_input_embeddings()  # Roberta embeddings
        self.LEN = 512

    def compute_constraint_grad(self, inputs):
        """
        Compute gradient of the non-toxic (class 1) loss wrt one-hot token inputs.
        Expects pre-tokenized tensors (input_ids, attention_mask).
        """
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        batch_size, seq_len = input_ids.shape

        # Convert input_ids -> one-hot with gradient
        one_hot = F.one_hot(input_ids, num_classes=self.vocab_size).float()  # [B, L, V]
        one_hot = one_hot.clone().detach().requires_grad_(True)

        # Map one-hot to embeddings manually (so gradient flows through one-hot)
        embedded = torch.matmul(one_hot, self.embed.weight)  # [B, L, D]

        # Forward pass through model using hidden embeddings
        outputs = self.model(inputs_embeds=embedded, attention_mask=attention_mask)
        logits = outputs.logits  # [B, 2]

        # Probability of toxic class (index 0)
        probs = F.softmax(logits, dim=1)

        # Define target = non-toxic (class 1)
        targets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits, targets)

        # Gradient wrt one-hot token inputs
        grad = torch.autograd.grad(loss, one_hot, retain_graph=False)[0]  # [B, L, V]

        return grad, probs



# # ---- Example usage ----
# toxic_model = Toxicity(device='cpu')
# texts = ["You are amazing and I love you quite a bit!", "Fuck you, I hate you. You suck. Screw you"]

# tokenized = toxic_model.tokenizer(
#     texts,
#     return_tensors='pt',
#     padding=True,
#     truncation=True,
#     max_length=toxic_model.LEN
    
# )

# grads, probs = toxic_model.compute_constraint_grad(
# tokenized
# )

# print("MASK ID:", toxic_model.tokenizer.mask_token_id)
# print("MMask token", toxic_model.tokenizer.mask_token)
# print("Grads shape:", grads.shape)
# grad_mask = grads[:, :, toxic_model.tokenizer.mask_token_id]
# print("Grad mask", grad_mask, grad_mask.shape)
# print("Probs:", probs)

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