from huggingface_hub import HfApi, login

login()   # asks for your token in the terminal; nothing shows while you paste it

repo_id = "Majellan/arabic-dialect-classifier"
api = HfApi()
api.create_repo(repo_id, repo_type="model", exist_ok=True)   # public, so the app can load it
api.upload_folder(
    folder_path="final_model",
    repo_id=repo_id,
    repo_type="model",
    ignore_patterns=["training_args.bin"],   # skips a file the app doesn't need
)
print("done:", f"https://huggingface.co/{repo_id}")