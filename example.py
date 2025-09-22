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

  input_texts = ["The quick brown fox "]
  samples = model.restore_model_and_sample(
      num_steps=config.sampling.steps, input_texts=input_texts)
  text_samples = model.tokenizer.batch_decode(samples)
  print(text_samples)



if __name__ == "__main__":
    main()