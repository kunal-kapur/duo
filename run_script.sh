python example.py \
  mode=sample_eval \
  loader.eval_batch_size=4 \
  data=openwebtext-split \
  algo=mdlm \
  algo.backbone=hf_dit \
  eval.checkpoint_path="kuleshov-group/mdlm-owt" \
  sampling.steps=2500 \
  model.length=256 \
  sampling.predictor=ancestral_cache \
  sampling.num_sample_batches=1 \
  sampling.noise_removal=greedy \
  +wandb.offline=true
