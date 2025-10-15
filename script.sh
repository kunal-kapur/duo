python main.py \
  mode=sample_eval \
  loader.batch_size=2 \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  algo=mdlm \
  eval.checkpoint_path=/home/kkapur/models_files/mdlm.ckpt  \
  sampling.steps=8 \
  sampling.num_sample_batches=1 \
  +wandb.offline=true 