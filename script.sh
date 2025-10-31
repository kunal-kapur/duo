python main.py \
  mode=gen_eval \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  algo=mdlm_loo \
  algo.num_segments=8 \
  algo.guidance_factor=1.0 \
  algo.num_loo=3 \
  sampling.num_sample_batches=4 \
  eval.checkpoint_path=/home/ubuntu/kkapur-v2/models/mdlm.ckpt  \
  +wandb.offline=false

  python main.py \
  mode=gen_eval \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  algo=mdlm \
  sampling.num_sample_batches=4 \
  eval.checkpoint_path=/home/ubuntu/kkapur-v2/models/mdlm.ckpt  \
  +wandb.offline=false


# python main.py \
#   mode=sample_eval \
#   loader.eval_batch_size=8 \
#   data=openwebtext-split \
#   algo=mdlm_loo \
#   algo.num_segments=8 \
#   algo.guidance_factor=1.0 \
#   sampling.steps=10 \
#   sampling.num_sample_batches=1 \
#   eval.checkpoint_path=/home/ubuntu/kkapur-v1/models/mdlm.ckpt  \
#   +wandb.offline=false

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