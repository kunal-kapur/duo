python main.py \
  mode=toxic_eval \
  loader.eval_batch_size=8 \
  data=toxicity \
  algo=mdlm \
  algo.constraint_function=toxicity \
  eval.checkpoint_path=/home/ubuntu/kkapur-v2/models/mdlm.ckpt  \
  model.length=128 \
  sampling.num_sample_batches=1 \
  sampling.steps=32 \
  +wandb.offline=true



python main.py \
  mode=toxic_eval \
  loader.eval_batch_size=8 \
  data=toxicity \
  algo=mdlm_loo \
  algo.num_loo=10  \
  algo.guidance_factor=0.1 \
  algo.constraint_function=toxicity \
  eval.checkpoint_path=/home/ubuntu/kkapur-v2/models/mdlm.ckpt  \
  model.length=128 \
  sampling.num_sample_batches=1 \
  sampling.steps=32 \
  +wandb.offline=true
