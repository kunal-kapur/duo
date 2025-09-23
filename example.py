from pyexpat import model
from algo import MDLM
import dataloader
import hydra
from collections import defaultdict


def _load_from_checkpoint(diffusion_model, config, tokenizer):
  if 'hf' in config.algo.backbone:
    return diffusion_model(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion_model.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  tokenizer = dataloader.get_tokenizer(config)
  diffusion_model = MDLM
  model = _load_from_checkpoint(
  diffusion_model=diffusion_model,
  config=config,
  tokenizer=tokenizer)
  print("Model loaded from checkpoint")

  input_texts = ["A minute later, I started crying like an oversized",
                  "A Japanese man has become the first person to receive"]
  
  config.loader.eval_batch_size = len(input_texts) # Scrapping this together

  MAX_SEGMENT_LENGTH = 16

  samples, boundaries = model.restore_model_and_sample(
      num_steps=config.sampling.steps, input_texts=input_texts)

  sample_boundaries = defaultdict(list)
  for idx, pos_idx in boundaries:
      sample_boundaries[idx].append(pos_idx)

  for idx in sample_boundaries:
      sample_boundaries[idx] = sorted(sample_boundaries[idx])

  text_samples = model.tokenizer.batch_decode(samples)  # decode all
  for idx, sample in enumerate(samples):
      boundary_indices = sample_boundaries.get(idx, [])
      if not boundary_indices:
          # No boundaries—single segment, maybe chunk it
          print(f"\nSample {idx+1} Segments:\n")
          for seg_start in range(0, len(sample), MAX_SEGMENT_LENGTH):
              segment = sample[seg_start:seg_start+MAX_SEGMENT_LENGTH]
              segment_text = model.tokenizer.decode(segment)
              print(segment_text)
              print("---SEGMENT BREAK---")
          continue

      prev = 0
      print(f"\nSample {idx+1} Segments:\n")
      for boundary in boundary_indices:
          segment = sample[prev:boundary]
          # Split if segment too long
          for seg_start in range(0, len(segment), MAX_SEGMENT_LENGTH):
              sub_segment = segment[seg_start:seg_start+MAX_SEGMENT_LENGTH]
              segment_text = model.tokenizer.decode(sub_segment)
              print(segment_text)
              print("---SEGMENT BREAK---")
          prev = boundary
      # Final segment
      segment = sample[prev:]
      for seg_start in range(0, len(segment), MAX_SEGMENT_LENGTH):
          sub_segment = segment[seg_start:seg_start+MAX_SEGMENT_LENGTH]
          segment_text = model.tokenizer.decode(sub_segment)
          print(segment_text)
          print("---SEGMENT BREAK---")


  for sample in text_samples:
      print(sample)
      print("\n\n")


if __name__ == "__main__":
    main()