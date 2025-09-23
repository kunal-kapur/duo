python example.py \
  mode=sample_eval \
  loader.eval_batch_size=1 \
  data=openwebtext-split \
  algo=mdlm \
  algo.backbone=hf_dit \
  eval.checkpoint_path="kuleshov-group/mdlm-owt" \
  sampling.steps=1000 \
  model.length=128 \
  sampling.predictor=ancestral_cache \
  sampling.num_sample_batches=1 \
  sampling.noise_removal=greedy \
  +wandb.offline=true
