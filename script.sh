python main.py \
  mode=sample_eval \
  loader.batch_size=2 \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  algo=mdlm \
  eval.checkpoint_path=/home/ubuntu/kkapur-v2/models/mdlm.ckpt  \
  sampling.steps=10 \
  sampling.num_sample_batches=2 \
  +wandb.offline=false


  # python main.py \
  # mode=ppl_eval  \
  # loader.batch_size=1 \
  # loader.eval_batch_size=16 \
  # data=openwebtext-split \
  # algo=mdlm \
  # eval.checkpoint_path=/home/ubuntu/kkapur-v1/models/mdlm.ckpt  \
  # sampling.steps=500 \
  # sampling.num_sample_batches=10 \
  # +wandb.offline=false