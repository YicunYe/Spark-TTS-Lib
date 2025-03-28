# Copyright (c) 2025 YowFung (yowfung@outlook.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from huggingface_hub import HfApi, snapshot_download
from retrying import retry

# Available mirror sites
MIRROR_SITES = [
    "https://hf-mirror.com",
    "https://huggingface.co",
    "https://huggingface.co/api",
]


def get_working_mirror():
    """Test and return a working mirror site"""
    api = HfApi()
    for mirror in MIRROR_SITES:
        try:
            os.environ["HF_ENDPOINT"] = mirror
            # Try accessing a public model to test connection
            api.model_info("gpt2")
            return mirror
        except Exception as e:
            continue
    raise RuntimeError("All mirror sites are unavailable.")


@retry(stop_max_attempt_number=3, wait_fixed=2000)
def download_pretrained_model(
    model_name: str = "SparkAudio/Spark-TTS-0.5B",
    local_dir: str = "pretrained_models/Spark-TTS-0.5B",
    token: str = None,
):
    """Download the pretrained model.

    Args:
        model_name (str): The name of the model to download.
        local_dir (str): The local directory to save the model.
        token (str, optional): Hugging Face token for authentication.
    """
    # Get a working mirror
    mirror = get_working_mirror()
    os.environ["HF_ENDPOINT"] = mirror
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Enable acceleration

    # If no token is provided, try to get it from the environment variable
    if token is None:
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # Ensure local_dir is an absolute path
    local_dir = os.path.abspath(local_dir)

    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        token=token,
    )


if __name__ == "__main__":
    download_pretrained_model()
