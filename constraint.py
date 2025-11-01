import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.nn import functional as F

class Constraint:
    def __init__(self, tokenizer, config):
        self.constraint_function = None
        if config.algo.constrain_function == 'toxicity':
            self.constraint_function= Toxicity(tokenizer)

    def compute_constraint_grad(self, text_chunks):
        return self.constraint_function.compute_toxicity_grad(text_chunks)


class Toxicity:
    def __init__(self, tokenizer):
        model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier')
        self.tokenizer = tokenizer
        self.model = model.to('cuda')
        self.LEN = 512
        self.mask_token = self.tokenizer.mask_token_id

    def compute_toxicity_grad(self, text_chunks):
        encoded = self.tokenizer(
            text_chunks,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=self.LEN
        )
        input_ids = encoded['input_ids'].to('cuda')
        attention_mask = encoded['attention_mask'].to('cuda')
        input_ids = input_ids.clone().detach().requires_grad_(True)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits      # [batch, 2]
        probs = F.softmax(logits, dim=1)[:, 0]  # Toxic = index 0

        targets = torch.ones(logits.shape[0], dtype=torch.long, device=logits.device)  # non-toxic=1
        loss = F.cross_entropy(logits, targets)
        grad = torch.autograd.grad(loss, input_ids, retain_graph=True)[0]

        return grad, probs



