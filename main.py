import json
import os

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
from metrics import Toxicity

from datasets import load_dataset


import algo
import dataloader
import utils

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(diffusion_model, config, tokenizer):
  if 'hf' in config.algo.backbone:
    return diffusion_model(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion_model.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)



# TODO Fix, this config is a mess
def _generate_samples(diffusion_model, config, logger,
                      tokenizer, prepend_data=None, model=None):
  logger.info('Starting Sample Eval.')
  if model is None:
    model = _load_from_checkpoint(
      diffusion_model=diffusion_model,
      config=config,
      tokenizer=tokenizer)
  
  hyperparameters = {
    "steps": config.sampling.steps,
  }
  model.metrics.gen_ppl.reset()
  model.metrics.sample_entropy.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  stride_length = config.sampling.stride_length
  num_strides = config.sampling.num_strides
  all_samples = []
  token_samples = []
  prepend_token_batches = []
  if prepend_data is not None:
      prepend_iter = iter(prepend_data)   # <--- create iterator ONCE
  else:
      prepend_iter = None
  for _ in range(config.sampling.num_sample_batches):
    if config.sampling.semi_ar:
      _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
        stride_length=stride_length,
        num_strides=num_strides,
        dt=1 / config.sampling.steps)
      text_samples = intermediate_samples[-1]
      # Note: Samples generated using semi-ar method
      # need to to be processed before computing generative perplexity
      # since these samples contain numerous <|endoftext|> tokens
      # and diffusion.compute_generative_perplexity() discards
      # any text after the first EOS token.
    else:
      prepended_text = None
      prepended_text = None
      if prepend_iter is not None:
        batch = next(prepend_iter)
        prepend_tokens = batch['input_ids'].to(model.device)
        prepended_text = prepend_tokens
        prepend_token_batches.extend(list(prepend_tokens.cpu()))  # <--- record

      samples = model.restore_model_and_sample(
        num_steps=config.sampling.steps,
        prepended_text=prepended_text
      )
      model.metrics.record_entropy(samples)
      token_samples.extend(list(samples.cpu()))
      text_samples = model.tokenizer.batch_decode(samples)
      model.metrics.record_generative_perplexity(
        text_samples, config.model.length, model.device)
      all_samples.extend(list(text_samples))
  generative_ppl = 0.
  entropy = 0.
  if not config.sampling.semi_ar:
    generative_ppl = model.metrics.gen_ppl.compute().item()
    entropy = model.metrics.sample_entropy.compute().item()
    metrics = {
      'generative_ppl': generative_ppl,
      'entropy': entropy
    }
    print('Generative perplexity:', generative_ppl)
    print('Sample entropy:', entropy)

  samples_path = config.eval.generated_samples_path
  with fsspec.open(samples_path, 'w') as f:
    json.dump({'generative_ppl': generative_ppl,
               'entropy': entropy,
               'generated_seqs': all_samples}, f, indent=4)
  print('Samples saved at:', samples_path)
  return metrics, token_samples, prepend_token_batches


def _gen_eval(diffusion_model, config, logger, tokenizer):
    temps_to_use = torch.linspace(.5, 1.0, steps=10).tolist()
    steps_to_use = [8, 16, 32]

    model = _load_from_checkpoint(
        diffusion_model=diffusion_model, config=config, tokenizer=tokenizer
    )
    hyperparameters = {
      "steps": config.sampling.steps,
    }


    if config.algo.name == 'mdlm_loo':
      hyperparameters['num_loo'] = config.algo.num_loo
      hyperparameters['num_segments'] = config.algo.num_segments
      hyperparameters['guidance_factor'] = config.algo.guidance_factor

    wandb_logger = None
    if config.get('wandb', None) is not None:
      wandb_logger = L.pytorch.loggers.WandbLogger(
        config=omegaconf.OmegaConf.to_object(config),
        ** config.wandb)
    # 
      wandb_logger.log_hyperparams(hyperparameters)


    # model.backbone = torch.compile(model.backbone)
    total_metadata = {}
    for steps in steps_to_use:
        cur_step_info = {}
        config.sampling.steps = steps
        for temp in temps_to_use:
            print("Itearting on ", temp)
            config.sampling.temperature = temp
            model.temperature = temp
            model.metrics.reset()
            res = _generate_samples(diffusion_model, config, logger, tokenizer, model=model)
            res['steps'] = steps
            res['temperature'] = temp
            cur_step_info[temp] = {
                "perplexity": res["generative_ppl"],
                "entropy": res["entropy"],
            }
        total_metadata[steps] = cur_step_info

    if wandb_logger is not None:
        flat_metrics = {
            f"ppl/steps_{steps}/temp_{temp}": vals["perplexity"]
            for steps, temps in total_metadata.items()
            for temp, vals in temps.items()
        }
        flat_metrics.update({
            f"entropy/steps_{steps}/temp_{temp}": vals["entropy"]
            for steps, temps in total_metadata.items()
            for temp, vals in temps.items()
        })
        wandb_logger.log_metrics(flat_metrics)

    return

def toxic_eval(diffusion_model, config, logger, tokenizer):
    logger.info('Starting Toxicity Eval.')

    temps_to_use = torch.linspace(0.5, 1.0, steps=10).tolist()
    steps_to_use = [8, 16, 32]

    model = _load_from_checkpoint(
        diffusion_model=diffusion_model, config=config, tokenizer=tokenizer
    )

    hyperparameters = {
        "steps": config.sampling.steps,
    }

    if config.algo.name == 'mdlm_constrain':
        hyperparameters['time_decay'] = config.algo.time_decay
        hyperparameters['signal_strength'] = config.algo.signal_strength

    wandb_logger = None
    if config.get('wandb', None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            **config.wandb
        )
        wandb_logger.log_hyperparams(hyperparameters)

    assert config.data.valid == 'toxicity'  # until I get more constraints

    # Load the validation dataset once
    print("GETTING DATALOADERS")
    _, valid_ds = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=config.seed
    )

    pad_id = tokenizer.pad_token_id
    constraint_model = model.constraint_function

    for steps in steps_to_use:
        config.sampling.steps = steps

        for temp in temps_to_use:
            print(f"Evaluating steps={steps}, temp={temp}")
            config.sampling.temperature = temp
            model.temperature = temp
            model.metrics.reset()

            # Generate samples
            res, generated_tokens, prepend_tokens = _generate_samples(
                diffusion_model, config, logger, tokenizer,
                prepend_data=valid_ds, model=model
            )

            # Process tokens to extract continuations
            prepend_tokens = torch.stack(prepend_tokens, dim=0)
            generated_tokens = torch.stack(generated_tokens, dim=0)
            pad_mask = (prepend_tokens == pad_id)

            batch_generated = []
            decoded_samples = []

            for i in range(generated_tokens.size(0)):
                continuation = generated_tokens[i][pad_mask[i]]
                batch_generated.append(continuation)
                decoded = tokenizer.decode(
                    continuation.tolist(),
                    skip_special_tokens=True
                )
                decoded_samples.append(decoded)

            # Evaluate toxicity
            toxicity = constraint_model.evaluate_constraint_text(
                decoded_samples, device=model.device
            )

            # Calculate metrics
            pct_not_toxic = toxicity[:, 0].mean(dim=0).item()
            pct_toxic = toxicity[:, 1].mean(dim=0).item()

            print(f"  Perplexity: {res['generative_ppl']:.4f}")
            print(f"  Entropy: {res['entropy']:.4f}")
            print(f"  % Not Toxic: {pct_not_toxic:.4f}")
            print(f"  % Toxic: {pct_toxic:.4f}")

            # Log metrics for current hyperparameter set
            if wandb_logger is not None:
                wandb_logger.log_metrics({
                    "steps": steps,
                    "temperature": temp,
                    "perplexity": res["generative_ppl"],
                    "entropy": res["entropy"],
                    "pct_not_toxic": pct_not_toxic,
                    "pct_toxic": pct_toxic,
                })

    print("Metrics logged to wandb (per hyperparameters set)")
    return



def _eval_ppl(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Perplexity Eval.')

  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  trainer.validate(model, valid_ds)


def _train(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      **config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  if config.training.finetune_path != '':
    assert utils.fsspec_exists(config.training.finetune_path)
    model = diffusion_model.load_from_checkpoint(
      config.training.finetune_path,
      tokenizer=tokenizer,
      config=config)
  else:
    model = diffusion_model(config, tokenizer=valid_ds.tokenizer)

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)
  if config.algo.name == 'ar':
    diffusion_model = algo.AR
  elif config.algo.name == 'mdlm':
    diffusion_model = algo.MDLM
  elif config.algo.name == 'mdlm_loo':
    diffusion_model = algo.MDLMLOO
  elif config.algo.name == 'mdlm_constrain':
    diffusion_model = algo.MDLMConstrain
  elif config.algo.name == 'duo_base':
    diffusion_model = algo.DUO_BASE
  elif config.algo.name == 'd3pm':
    diffusion_model = algo.D3PMAbsorb
  elif config.algo.name == 'sedd':
    diffusion_model = algo.SEDDAbsorb
  elif config.algo.name == 'duo':
    diffusion_model = algo.DUO
  elif config.algo.name == 'distillation':
    diffusion_model = algo.Distillation
  elif config.algo.name == 'ot-finetune':
    diffusion_model = algo.OptimalTransportFinetune
  else:
    raise ValueError(
      f'Invalid algorithm name: {config.algo.name}')
  kwargs = {'diffusion_model': diffusion_model,
            'config': config,
            'tokenizer': tokenizer,
            'logger': logger}
  if config.mode == 'sample_eval':
    _generate_samples(**kwargs)
  elif config.mode == 'ppl_eval':
    _eval_ppl(**kwargs)
  elif config.mode == 'gen_eval':
    _gen_eval(**kwargs)

  elif config.mode == 'toxic_eval':
    toxic_eval(**kwargs)

  else:
    _train(**kwargs)


if __name__ == '__main__':
  main()