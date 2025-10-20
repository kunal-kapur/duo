python main.py \
  mode=sample_eval \
  loader.batch_size=2 \
  loader.eval_batch_size=16 \
  data=openwebtext-split \
  algo=mdlm_segmentation \
  eval.checkpoint_path=/home/kkapur/models_files/mdlm.ckpt  \
  sampling.steps=20 \
  sampling.num_sample_batches=2 \
  +wandb.offline=true 


  python main.py \
  mode=sample_eval \
  loader.batch_size=2 \
  loader.eval_batch_size=16 \
  data=openwebtext-split \
  algo=mdlm \
  eval.checkpoint_path=/home/kkapur/models_files/mdlm.ckpt  \
  sampling.steps=20 \
  sampling.num_sample_batches=2 \
  +wandb.offline=true 