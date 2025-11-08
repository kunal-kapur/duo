import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer, RobertaModel
from torch.nn import functional as F
import torch.nn as nn

class Constraint:
    def __init__(self, tokenizer, config):
        self.constraint_function = None
        if config.algo.get('constraint_function', None) == 'toxicity':
            self.constraint_function= Toxicity(tokenizer)
        return None

    def compute_constraint_grad(self, input_ids, attention_mask):
        with torch.enable_grad():
          return self.constraint_function.compute_constraint_grad(input_ids, attention_mask)

    @torch.no_grad()
    def evaluate_constraint_text(self, text_chunks, device):
        return self.constraint_function.evaluate_constraint_text(text_chunks)



class Toxicity:
    def __init__(self, tokenizer, device='cuda'):
        model_dir = "/home/ubuntu/kkapur-v2/models/replaced_vocab_roberta_for_jigsaw"
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)

        self.tokenizer = tokenizer
        self.device = device
        self.LEN = 512
        self.mask_index = len(tokenizer)  # virtual token index (not in vocab)

        # TODO Pls fix this or fine tune a new model
        # HACKY way to get around adding mask token to embedding layer
        old_emb = model.roberta.embeddings.word_embeddings
        new_emb = nn.Embedding(old_emb.num_embeddings + 1, old_emb.embedding_dim)
        new_emb.weight.data[:-1] = old_emb.weight.data
        nn.init.normal_(new_emb.weight.data[-1:], mean=0, std=old_emb.weight.data.std())

        model.roberta.embeddings.word_embeddings = new_emb.to(device)
        self.model = model

    @torch.no_grad()
    def evaluate_constraint_text(self, text_chunks):

        # this should should add attention mask automatically
        tokenized = self.tokenizer(
            text_chunks,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.LEN
        ).to(self.device)
        return self.evaluate_constraint(tokenized)

    @torch.no_grad()
    def evaluate_constraint(self, input):
        """
        Compute constraint function on this
        """
        outputs = self.model(**input)
        logits = outputs.logits  # [B, 2]
        probs = F.softmax(logits, dim=1)
        return probs


    def compute_constraint_grad(self, input_ids, attention_mask):
        """
        Compute gradient of the non-toxic (class 1) loss wrt one-hot token inputs.
        Expects pre-tokenized tensors (input_ids, attention_mask).
        """
        # input_ids = input_ids.to(self.device)
        # attention_mask = attention_mask.to(self.device)
        batch_size, seq_len = input_ids.shape

        # Get current embedding weights dynamically (handles extended vocab)
        embed_layer = self.model.get_input_embeddings()
        vocab_size, embed_dim = embed_layer.weight.shape

        # Convert input_ids -> one-hot with gradient, move to same device
        one_hot = F.one_hot(input_ids, num_classes=vocab_size).float().to(self.device)
        one_hot = one_hot.clone().detach().requires_grad_(True)

        # Map one-hot to embeddings manually (so gradient flows through one-hot)
        embedded = torch.matmul(one_hot, embed_layer.weight.to(self.device))  # [B, L, D]

        # Forward pass through model using hidden embeddings
        outputs = self.model(inputs_embeds=embedded, attention_mask=attention_mask)
        logits = outputs.logits  # [B, 2]
        probs = F.softmax(logits, dim=1)

        # Define target = non-toxic (class 1)
        targets = torch.ones(batch_size, dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits, targets)

        # Gradient wrt one-hot token inputs
        grad = torch.autograd.grad(loss, one_hot, retain_graph=False)[0]  # [B, L, V]

        return grad, probs


