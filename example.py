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
  print("Configuration:", config)
  tokenizer = dataloader.get_tokenizer(config)
  diffusion_model = MDLM
  model = _load_from_checkpoint(
  diffusion_model=diffusion_model,
  config=config,
  tokenizer=tokenizer)
  print("Model loaded from checkpoint")

  input_texts = ["A minute later, I started crying like an oversized"]
#                  "Some people, we'll call them “Lucky Motherfuckers",
# "A Japanese man has become the first person to receive",
# "They tell of hearing anti-Semitic epithets and"]
  samples, boundaries = model.restore_model_and_sample(
      num_steps=config.sampling.steps, input_texts=input_texts)
  
  sample_boundaries = defaultdict(list)
  for idx, pos_idx in boundaries:
      sample_boundaries[idx].append(pos_idx)

  # Sort boundaries for each sample
  for idx in sample_boundaries:
      sample_boundaries[idx] = sorted(sample_boundaries[idx])

  text_samples = model.tokenizer.batch_decode(samples)  # decode all
  for idx, sample in enumerate(samples):
      boundary_indices = sample_boundaries.get(idx, [])
      if not boundary_indices:
          # No boundaries—single segment
          print(f"\nSample {idx+1}:")
          print(model.tokenizer.decode(sample))
          continue
      prev = 0
      print(f"\nSample {idx+1} Segments:\n")
      for boundary in boundary_indices:
          segment = sample[prev:boundary]
          segment_text = model.tokenizer.decode(segment)
          print(segment_text)
          print("---SEGMENT BREAK---")
          prev = boundary
      # Final segment
      segment = sample[prev:]
      segment_text = model.tokenizer.decode(segment)
      print(segment_text)

  print(text_samples)





if __name__ == "__main__":
    main()