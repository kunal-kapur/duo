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
    tokenizer = dataloader.get_tokenizer(config)
    diffusion_model = MDLM(config=config, tokenizer=tokenizer)
    model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)

    samples = model.restore_model_and_sample(
        num_steps=config.sampling.steps)
    
    print(samples)



if __name__ == "__main__":
    main()