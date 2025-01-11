# src/components/repository_setup.py

import os
import subprocess

def setup_finetune_repo():
    repo_url = "https://github.com/thisisAranya/lmms-finetune.git"
    repo_path = "./external_repos/lmms-finetune"

    if not os.path.exists(repo_path):
        print("Cloning lmms-finetune repository...")
        subprocess.run(["git", "clone", repo_url, repo_path], check=True)
    else:
        print("Repository already exists.")

if __name__ == "__main__":
    setup_finetune_repo()
