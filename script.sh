python main.py \
  mode=toxic_eval \
  loader.eval_batch_size=32 \
  data=toxicity \
  algo=mdlm \
  sampling.temperature=0.7 \
  algo.constraint_function=toxicity \
  eval.checkpoint_path=/home/ubuntu/kkapur-v2/models/mdlm.ckpt  \
  model.length=128 \
  sampling.num_sample_batches=1 \
  sampling.steps=128 \
  +wandb.offline=true



python main.py \
  mode=toxic_eval \
  loader.eval_batch_size=32 \
  data=toxicity \
  algo=mdlm_constrain \
  algo.time_decay=1.0 \
  algo.signal_strength=50.0 \
  sampling.temperature=0.7 \
  algo.constraint_function=toxicity \
  eval.checkpoint_path=/home/ubuntu/kkapur-v2/models/mdlm.ckpt  \
  model.length=128 \
  sampling.num_sample_batches=1 \
  sampling.steps=64 \
  +wandb.offline=true

