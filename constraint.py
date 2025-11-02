import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.nn import functional as F

class Constraint:
    def __init__(self, tokenizer, config):
        self.constraint_function = None
        if config.algo.get('constraint_function', None) == 'toxicity':
            self.constraint_function= Toxicity(tokenizer)
        return None

    @torch.no_grad()
    def compute_constraint_grad(self, text_chunks):
        return self.constraint_function.compute_toxicity_grad(text_chunks)
    
    @torch.no_grad()
    def evaluate_constraint_text(self, text_chunks, device):
        return self.constraint_function.evaluate_constraint_text(text_chunks, device)



class Toxicity:
    def __init__(self, tokenizer=None , device='cuda'):
        if tokenizer is None:
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier')
        self.tokenizer = tokenizer
        self.device = device
        self.model = model.to(self.device)
        self.device = self.device
        self.vocab_size = tokenizer.vocab_size
        self.embed = self.model.get_input_embeddings()  # Roberta embeddings
        self.LEN = 512

    @torch.no_grad()
    def evaluate_constraint_text(self, text_chunks, device):
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
        # Forward pass through model using hidden embeddings
        outputs = self.model(**input)
        logits = outputs.logits  # [B, 2]
        probs = F.softmax(logits, dim=1)
        return probs

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




