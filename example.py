from algo import MDLM
import dataloader
import hydra


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

  input_texts = ["A minute later, I started crying like an oversized", 
                 "Some people, we'll call them “Lucky Motherfuckers",
"A Japanese man has become the first person to receive",
"They tell of hearing anti-Semitic epithets and"]
  samples = model.restore_model_and_sample(
      num_steps=config.sampling.steps, input_texts=input_texts)
  for sample in samples:
     print("Sample length:", len(sample))
  text_samples = model.tokenizer.batch_decode(samples)
  for idx,text_sample in enumerate(text_samples):
    print(f"\n\n\n\nSample {idx+1}:\n\n")
    print(text_sample)



if __name__ == "__main__":
    main()