# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#               2025 YowFung (yowfung@outlook.com)
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
import time
import torch
import soundfile as sf
import logging
from datetime import datetime
import platform

from spark_tts_lib.SparkTTS import SparkTTS
from spark_tts_lib.download import download_pretrained_model


# Get the root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Args:
    """Arguments for the TTS inference."""

    model_dir = os.path.join(root_dir, "pretrained_models/Spark-TTS-0.5B")
    save_dir = os.path.join(root_dir, "examples/results")
    device = 0
    text = "生活就像海洋，只有意志坚强的人才能到达彼岸。某一天当你成功的时候，你是否会对着大海说： Thank you and fuck you!"
    prompt_speech_path = os.path.join(root_dir, "examples/prompt_audio.wav")
    prompt_text = "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。"
    gender = None        # female | male
    pitch = None         # very_low | low | moderate | high | very_high
    speed = 0.8          # very_low | low | moderate | high | very_high
    temperature = 0.8
    top_k = 50
    top_p = 0.95


def run_tts(args):
    """Perform TTS inference and save the generated audio."""
    logging.info(f"Using model from: {args.model_dir}")
    logging.info(f"Saving audio to: {args.save_dir}")

    # Ensure the save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Convert device argument to torch.device
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        # macOS with MPS support (Apple Silicon)
        device = torch.device(f"mps:{args.device}")
        logging.info(f"Using MPS device: {device}")
    elif torch.cuda.is_available():
        # System with CUDA support
        device = torch.device(f"cuda:{args.device}")
        logging.info(f"Using CUDA device: {device}")
    else:
        # Fall back to CPU
        device = torch.device("cpu")
        logging.info("GPU acceleration not available, using CPU")

    # Initialize the model
    logging.info(f"⌛️ Initializing model...")
    model = SparkTTS(args.model_dir, device)

    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(args.save_dir, f"{timestamp}.wav")

    logging.info("🚀 Starting inference...")

    # Perform inference and save the output audio
    start_at = time.time()
    with torch.no_grad():
        wav = model.inference(
            args.text,
            args.prompt_speech_path,
            prompt_text=args.prompt_text,
            gender=args.gender,
            pitch=args.pitch,
            speed=args.speed,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        sf.write(save_path, wav, samplerate=16000)

    logging.info(f"✅ Audio saved at: {save_path}")

    # Calculate the elapsed time
    elapsed = time.time() - start_at
    logging.info(f"🕒 Inference elapsed time: {elapsed:.2f} seconds.")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Initialize the arguments
    args = Args()

    # Download the pretrained model if it doesn't exist
    if not os.path.exists(args.model_dir):
        logging.info(f"⌛️ Downloading pretrained model to: {args.model_dir}")
        download_pretrained_model(local_dir=args.model_dir)
        logging.info(f"✅ Pretrained model downloaded.")

    # Perform TTS inference
    run_tts(args)
