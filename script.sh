python main.py \
  mode=sample_eval \
  loader.batch_size=1 \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  algo=mdlm_loo \
  eval.checkpoint_path=/home/kkapur/models_files/mdlm.ckpt  \
  sampling.steps=500 \
  sampling.num_sample_batches=2 \
  +wandb.offline=true 


  python main.py \
  mode=sample_eval \
  loader.batch_size=1 \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  algo=mdlm \
  eval.checkpoint_path=/home/kkapur/models_files/mdlm.ckpt  \
  sampling.steps=500 \
  sampling.num_sample_batches=4 \
  +wandb.offline=true 