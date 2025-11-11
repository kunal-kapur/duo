"""
Utilities for evaluating conditional language generation with proper masking.

This module provides functions to compute perplexity and loss for generated
continuations while using prompts as context, ensuring that loss is only
calculated over the generated tokens, not the prompt tokens.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
import transformers


def compute_conditional_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    reduction: str = 'none',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute loss for conditional generation where prompt is context.

    This function computes the negative log-likelihood (loss) only for tokens
    that were generated after the prompt, while still using the prompt as
    context for the model's predictions.

    Args:
        model: A language model (e.g., GPT-2, LLaMA) that returns logits
        input_ids: Token IDs of shape (batch_size, seq_len)
                   Format: [prompt_tokens | generated_tokens]
        prompt_lengths: Length of prompt for each sample, shape (batch_size,)
        attention_mask: Optional attention mask, shape (batch_size, seq_len)
                       If None, assumes all tokens are valid (no padding)
        reduction: How to reduce the loss. Options:
                   - 'none': Return per-token losses
                   - 'mean': Average loss over all generated tokens
                   - 'batch': Average loss per sample in batch

    Returns:
        losses: Per-token losses, shape (batch_size, seq_len-1)
        loss_mask: Boolean mask indicating which tokens to include in loss
                   Shape: (batch_size, seq_len-1)
        num_tokens: Number of valid generated tokens per sample, shape (batch_size,)

    Example:
        >>> model = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
        >>> tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
        >>>
        >>> # Prepare data
        >>> prompts = ["The weather is", "I love"]
        >>> continuations = [" very nice today", " coding in Python"]
        >>>
        >>> # Tokenize
        >>> prompt_ids = [tokenizer.encode(p) for p in prompts]
        >>> full_ids = [tokenizer.encode(p + c) for p, c in zip(prompts, continuations)]
        >>>
        >>> # Pad to same length
        >>> max_len = max(len(ids) for ids in full_ids)
        >>> input_ids = torch.tensor([
        >>>     ids + [tokenizer.pad_token_id] * (max_len - len(ids))
        >>>     for ids in full_ids
        >>> ])
        >>> prompt_lengths = torch.tensor([len(p) for p in prompt_ids])
        >>>
        >>> # Compute loss only on continuations
        >>> losses, mask, num_tokens = compute_conditional_loss(
        >>>     model, input_ids, prompt_lengths
        >>> )
        >>>
        >>> # Get perplexity per sample
        >>> ppls = torch.exp((losses * mask).sum(dim=1) / num_tokens)
    """
    batch_size, seq_len = input_ids.shape

    # Create attention mask if not provided
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    # Forward pass through model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

    # Shift logits and labels for next-token prediction
    # logits: (batch, seq_len, vocab_size) -> (batch, seq_len-1, vocab_size)
    # labels: (batch, seq_len) -> (batch, seq_len-1)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # Compute per-token cross-entropy loss
    # Shape: (batch_size, seq_len-1)
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='none'
    ).view(batch_size, seq_len - 1)

    # Create mask for generated tokens only (not prompt tokens)
    # We want to include tokens at positions [prompt_length, seq_len-1]
    # Since we shifted, position i in shift_labels corresponds to predicting token i+1
    # So we mask out positions [0, prompt_length-1] in the shifted view
    position_indices = torch.arange(seq_len - 1, device=input_ids.device).unsqueeze(0)  # (1, seq_len-1)
    prompt_lengths_expanded = prompt_lengths.unsqueeze(1)  # (batch_size, 1)

    # Mask: True for generated tokens (position >= prompt_length)
    # In shifted view, position i predicts token i+1, so:
    # - position 0 predicts token 1
    # - position prompt_length-1 predicts token prompt_length
    # - position prompt_length predicts token prompt_length+1 (first generated token)
    loss_mask = position_indices >= prompt_lengths_expanded  # (batch_size, seq_len-1)

    # Also mask out padding tokens in the shifted attention mask
    padding_mask = attention_mask[:, 1:].bool()  # (batch_size, seq_len-1)

    # Combine masks: only include generated, non-padding tokens
    loss_mask = loss_mask & padding_mask

    # Count valid tokens per sample
    num_tokens = loss_mask.sum(dim=1)  # (batch_size,)

    # Apply reduction if requested
    if reduction == 'mean':
        total_loss = (losses * loss_mask).sum()
        total_tokens = num_tokens.sum()
        return total_loss / total_tokens, loss_mask, num_tokens
    elif reduction == 'batch':
        # Average per sample
        batch_losses = (losses * loss_mask).sum(dim=1) / num_tokens.clamp(min=1)
        return batch_losses, loss_mask, num_tokens
    else:  # reduction == 'none'
        return losses, loss_mask, num_tokens


def compute_conditional_perplexity(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    per_sample: bool = True,
) -> Union[torch.Tensor, float]:
    """
    Compute perplexity for conditional generation.

    Args:
        model: A language model
        input_ids: Token IDs of shape (batch_size, seq_len)
        prompt_lengths: Length of prompt for each sample, shape (batch_size,)
        attention_mask: Optional attention mask
        per_sample: If True, return perplexity per sample. If False, return
                   overall perplexity across all samples.

    Returns:
        If per_sample=True: Tensor of shape (batch_size,) with perplexity per sample
        If per_sample=False: Scalar tensor with overall perplexity

    Example:
        >>> # Compute perplexity for each prompt-continuation pair
        >>> ppls = compute_conditional_perplexity(
        >>>     model, input_ids, prompt_lengths, per_sample=True
        >>> )
        >>> print(f"Sample perplexities: {ppls}")
        >>>
        >>> # Compute overall perplexity
        >>> overall_ppl = compute_conditional_perplexity(
        >>>     model, input_ids, prompt_lengths, per_sample=False
        >>> )
        >>> print(f"Overall perplexity: {overall_ppl.item()}")
    """
    losses, loss_mask, num_tokens = compute_conditional_loss(
        model, input_ids, prompt_lengths, attention_mask, reduction='none'
    )

    if per_sample:
        # Perplexity per sample
        masked_losses = losses * loss_mask
        avg_loss_per_sample = masked_losses.sum(dim=1) / num_tokens.clamp(min=1)
        perplexities = torch.exp(avg_loss_per_sample)
        return perplexities
    else:
        # Overall perplexity
        total_loss = (losses * loss_mask).sum()
        total_tokens = num_tokens.sum()
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(avg_loss)
        return perplexity


def batch_conditional_eval(
    model: torch.nn.Module,
    prompts: List[str],
    continuations: List[str],
    tokenizer: transformers.PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate conditional generation on a dataset with batching.

    Args:
        model: A language model
        prompts: List of prompt strings
        continuations: List of continuation strings (one per prompt)
        tokenizer: Tokenizer for the model
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        device: Device to run evaluation on

    Returns:
        perplexities: Tensor of shape (num_samples,) with perplexity per sample
        losses: Tensor of shape (num_samples,) with average loss per sample

    Example:
        >>> model = transformers.GPT2LMHeadModel.from_pretrained('gpt2').to('cuda')
        >>> tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
        >>> tokenizer.pad_token = tokenizer.eos_token
        >>>
        >>> prompts = ["The weather is", "I love", "Machine learning"]
        >>> continuations = [" nice today", " Python", " is fascinating"]
        >>>
        >>> ppls, losses = batch_conditional_eval(
        >>>     model, prompts, continuations, tokenizer
        >>> )
        >>>
        >>> for i, (prompt, cont, ppl, loss) in enumerate(
        >>>     zip(prompts, continuations, ppls, losses)
        >>> ):
        >>>     print(f"Sample {i}:")
        >>>     print(f"  Prompt: '{prompt}'")
        >>>     print(f"  Continuation: '{cont}'")
        >>>     print(f"  Perplexity: {ppl:.2f}")
        >>>     print(f"  Loss: {loss:.4f}")
    """
    model.eval()

    assert len(prompts) == len(continuations), \
        "Number of prompts and continuations must match"

    all_perplexities = []
    all_losses = []

    num_samples = len(prompts)
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        batch_prompts = prompts[start_idx:end_idx]
        batch_continuations = continuations[start_idx:end_idx]

        # Tokenize prompts and full sequences
        prompt_encodings = [
            tokenizer.encode(p, add_special_tokens=True)
            for p in batch_prompts
        ]
        full_encodings = [
            tokenizer.encode(p + c, add_special_tokens=True, max_length=max_length, truncation=True)
            for p, c in zip(batch_prompts, batch_continuations)
        ]

        # Get prompt lengths
        prompt_lengths = torch.tensor([len(p) for p in prompt_encodings], device=device)

        # Pad sequences to same length
        max_len = max(len(enc) for enc in full_encodings)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        input_ids = torch.tensor([
            enc + [pad_token_id] * (max_len - len(enc))
            for enc in full_encodings
        ], device=device)

        attention_mask = torch.tensor([
            [1] * len(enc) + [0] * (max_len - len(enc))
            for enc in full_encodings
        ], device=device)

        # Compute metrics
        batch_ppls = compute_conditional_perplexity(
            model, input_ids, prompt_lengths, attention_mask, per_sample=True
        )

        losses, loss_mask, num_tokens = compute_conditional_loss(
            model, input_ids, prompt_lengths, attention_mask, reduction='none'
        )
        batch_losses = (losses * loss_mask).sum(dim=1) / num_tokens.clamp(min=1)

        all_perplexities.append(batch_ppls.cpu())
        all_losses.append(batch_losses.cpu())

    return torch.cat(all_perplexities), torch.cat(all_losses)


def integrate_with_metrics_class(
    metrics_instance,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
):
    """
    Integrate conditional perplexity computation with your existing Metrics class.

    This function computes conditional loss and updates the perplexity metrics
    in your codebase's Metrics class.

    Args:
        metrics_instance: Instance of your Metrics class (from metrics.py)
        model: Language model to evaluate with
        input_ids: Token IDs including both prompt and generation
        prompt_lengths: Length of prompt for each sample
        attention_mask: Optional attention mask

    Example:
        >>> from metrics import Metrics
        >>>
        >>> # Initialize metrics
        >>> metrics = Metrics(
        >>>     gen_ppl_eval_model_name_or_path='gpt2',
        >>>     eval_ppl_batch_size=8,
        >>>     tokenizer=tokenizer
        >>> )
        >>>
        >>> # Compute and record conditional perplexity
        >>> integrate_with_metrics_class(
        >>>     metrics, model, input_ids, prompt_lengths, attention_mask
        >>> )
        >>>
        >>> # Get the computed perplexity
        >>> ppl = metrics.gen_ppl.compute()
        >>> print(f"Conditional perplexity: {ppl:.2f}")
    """
    losses, loss_mask, num_tokens = compute_conditional_loss(
        model, input_ids, prompt_lengths, attention_mask, reduction='none'
    )

    # Update metrics using the same pattern as in your codebase
    # See metrics.py line 262: self.gen_ppl.update(nlls * valid_tokens, valid_tokens)
    metrics_instance.gen_ppl.update(losses * loss_mask, loss_mask)


if __name__ == "__main__":
    """Example usage demonstrating the conditional evaluation."""
    import transformers

    print("=" * 80)
    print("Example: Conditional Language Generation Evaluation")
    print("=" * 80)

    # Load model and tokenizer
    model_name = 'gpt2'
    print(f"\nLoading model: {model_name}")
    model = transformers.GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Prepare example data
    prompts = [
        "The weather today is",
        "I love programming in",
        "Machine learning is",
    ]
    continuations = [
        " sunny and warm with clear skies",
        " Python because it's versatile",
        " transforming how we build software",
    ]

    print("\n" + "-" * 80)
    print("Dataset:")
    print("-" * 80)
    for i, (p, c) in enumerate(zip(prompts, continuations)):
        print(f"Sample {i+1}:")
        print(f"  Prompt: '{p}'")
        print(f"  Continuation: '{c}'")
        print()

    # Method 1: Batch evaluation
    print("-" * 80)
    print("Method 1: Batch Evaluation")
    print("-" * 80)
    ppls, losses = batch_conditional_eval(
        model, prompts, continuations, tokenizer,
        batch_size=2, device='cpu'
    )

    for i, (p, c, ppl, loss) in enumerate(zip(prompts, continuations, ppls, losses)):
        print(f"Sample {i+1}:")
        print(f"  Text: '{p}{c}'")
        print(f"  Perplexity: {ppl:.3f}")
        print(f"  Avg Loss: {loss:.4f}")
        print()

    # Method 2: Manual batching with compute_conditional_loss
    print("-" * 80)
    print("Method 2: Manual Computation")
    print("-" * 80)

    # Tokenize
    prompt_encodings = [tokenizer.encode(p, add_special_tokens=True) for p in prompts]
    full_encodings = [tokenizer.encode(p + c, add_special_tokens=True)
                      for p, c in zip(prompts, continuations)]
    prompt_lengths = torch.tensor([len(p) for p in prompt_encodings])

    # Pad
    max_len = max(len(enc) for enc in full_encodings)
    input_ids = torch.tensor([
        enc + [tokenizer.eos_token_id] * (max_len - len(enc))
        for enc in full_encodings
    ])
    attention_mask = torch.tensor([
        [1] * len(enc) + [0] * (max_len - len(enc))
        for enc in full_encodings
    ])

    # Compute
    ppls_manual = compute_conditional_perplexity(
        model, input_ids, prompt_lengths, attention_mask, per_sample=True
    )

    losses_manual, mask, num_tokens = compute_conditional_loss(
        model, input_ids, prompt_lengths, attention_mask, reduction='none'
    )
    avg_losses = (losses_manual * mask).sum(dim=1) / num_tokens

    print("Results:")
    for i, (ppl, loss, n_tok) in enumerate(zip(ppls_manual, avg_losses, num_tokens)):
        print(f"Sample {i+1}:")
        print(f"  Perplexity: {ppl:.3f}")
        print(f"  Avg Loss: {loss:.4f}")
        print(f"  Num generated tokens: {n_tok.item()}")
        print()

    # Method 3: Overall perplexity
    print("-" * 80)
    print("Method 3: Overall Corpus Perplexity")
    print("-" * 80)
    overall_ppl = compute_conditional_perplexity(
        model, input_ids, prompt_lengths, attention_mask, per_sample=False
    )
    print(f"Overall perplexity across all samples: {overall_ppl:.3f}")

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)
