import os
from huggingface_hub import HfApi, hf_hub_download
import joblib
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

api = HfApi(token=hf_token)

def upload_file():
    api.upload_file(
        path_or_fileobj="/home/peter/projects/ngai/sd-scripts/finesse-model/finesse_prodstudio-step00002000.safetensors",
        path_in_repo="finesse_prodstudio-step00002000.safetensors",
        repo_id="pmlakner/flux-fin-fl1",
        repo_type="model",
    )

def download_file():

    REPO_ID = "pmlakner/flux-fin-fl1"
    FILENAME = "finesse_prodstudio-step00002000.safetensors"

    model = joblib.load(hf_hub_download(repo_id=REPO_ID, filename=FILENAME, token=hf_token))

#  HF_HUB_ENABLE_HF_TRANSFER=1 python hf_upload.py
