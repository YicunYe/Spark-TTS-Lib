import os
import time

import numpy as np
import pytest
import torch

from spark_tts_lib.SparkTTS import SparkTTS
from spark_tts_lib.utils.download import download_pretrained_model


def run_inference_test(device: torch.device):
    """General inference test function

    Args:
        device (torch.device): Running device (CPU/GPU)
    """
    model_dir = "pretrained_models/Spark-TTS-0.5B"
    if not os.path.exists(model_dir):
        print("⌛️ Downloading pretrained model...")
        download_pretrained_model(local_dir=model_dir)
        print("☑️ Downloaded pretrained model.")

    print("⌛️ Loading model...")
    start_at = time.time()
    model = SparkTTS(model_dir, device)
    elapsed_time = time.time() - start_at
    print(f"☑️ Model loaded in {elapsed_time:.2f} seconds.")

    text = "生活就像海洋，只有意志坚强的人才能到达彼岸。"
    prompt_speech_path = "tests/prompt_audio.wav"
    prompt_text = "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。"

    print("⌛️ Inferring...")
    start_at = time.time()
    wav = model.inference(text, prompt_speech_path, prompt_text)

    elapsed_time = time.time() - start_at
    print(f"☑️ Inference completed in {elapsed_time:.2f} seconds.")

    assert isinstance(wav, np.ndarray)
    assert wav.ndim == 1
    assert len(wav) > 1000

    print("✅ Inference test passed.")


@pytest.mark.order(1)
def test_inference_use_cpu():
    """Test CPU inference"""
    run_inference_test(torch.device("cpu"))


@pytest.mark.order(2)
def test_inference_use_cuda():
    """Test CUDA inference"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    run_inference_test(torch.device("cuda"))
