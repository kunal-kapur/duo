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
  mode=ppl_eval  \
  loader.batch_size=1 \
  loader.eval_batch_size=16 \
  data=openwebtext-split \
  algo=mdlm \
  algo.constraint_function=toxicity \
  eval.checkpoint_path=/home/ubuntu/kkapur-v1/models/mdlm.ckpt  \
  sampling.steps=500 \
  sampling.num_sample_batches=10 \
  +wandb.offline=false