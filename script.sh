python main.py \
  mode=toxic_eval \
  loader.eval_batch_size=32 \
  data=toxicity \
  algo=mdlm \
  sampling.temperature=0.7 \
  algo.constraint_function=toxicity \
  eval.checkpoint_path=/home/kkapur/models/mdlm.ckpt  \
  model.length=128 \
  sampling.num_sample_batches=1 \
  sampling.steps=64 \
  +wandb.offline=true



python main.py \
  mode=toxic_eval \
  loader.eval_batch_size=32 \
  data=toxicity \
  algo=mdlm_loo \
  algo.num_loo=128  \
  algo.guidance_factor=0.2\
  sampling.temperature=0.7 \
  algo.constraint_function=toxicity \
  eval.checkpoint_path=/home/kkapur/models/mdlm.ckpt  \
  model.length=128 \
  sampling.num_sample_batches=1 \
  sampling.steps=64 \
  +wandb.offline=true
