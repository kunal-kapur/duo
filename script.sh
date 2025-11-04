python main.py \
  mode=toxic_eval \
  loader.eval_batch_size=32 \
  data=toxicity \
  algo=mdlm \
  algo.constraint_function=toxicity \
  eval.checkpoint_path=/home/kkapur/models/mdlm.ckpt  \
  model.length=256 \
  sampling.num_sample_batches=1 \
  sampling.steps=32 \
  +wandb.offline=true



python main.py \
  mode=toxic_eval \
  loader.eval_batch_size=32 \
  data=toxicity \
  algo=mdlm_loo \
  algo.num_loo=40  \
  algo.guidance_factor=0.2\
  algo.constraint_function=toxicity \
  eval.checkpoint_path=/home/kkapur/models/mdlm.ckpt  \
  model.length=256 \
  sampling.num_sample_batches=1 \
  sampling.steps=32 \
  +wandb.offline=true
