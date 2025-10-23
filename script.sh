python main.py \
  mode=sample_eval \
  loader.batch_size=1 \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  algo=mdlm_loo \
  eval.checkpoint_path=/home/ubuntu/kkapur-v1/models/mdlm.ckpt  \
  sampling.steps=500 \
  sampling.num_sample_batches=20 \
  +wandb.offline=false


  python main.py \
  mode=ppl_eval  \
  loader.batch_size=1 \
  loader.eval_batch_size=16 \
  data=openwebtext-split \
  algo=mdlm \
  eval.checkpoint_path=/home/ubuntu/kkapur-v1/models/mdlm.ckpt  \
  sampling.steps=500 \
  sampling.num_sample_batches=10 \
  +wandb.offline=false