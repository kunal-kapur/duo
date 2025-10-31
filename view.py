from wandb.apis.public import Api

api = Api()

# Get a single run by entity/project/run_id
run = api.run("kunalkapur888-purdue-university/MDLM_LOO/mdlm_openwebtext-train_log-linear_1_20251031_024723")

# Print run details
print(f"Run name: {run.name}")
print(f"Run ID: {run.id}")
print(f"Run URL: {run.url}")
print(f"Run state: {run.state}")
print(f"Run config: {run.config}")
print(f"Run summary: {run.summary}")
print(f"Run history (samples=5):")
print(run.history(samples=5))
print("----------")

# Get full history as DataFrame (you can limit keys if you like)
histories_df = run.history(samples=100, keys=["loss", "accuracy"])
print(histories_df.head())
