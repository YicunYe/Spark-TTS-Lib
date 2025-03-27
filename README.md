<div align="center">
    <h1>
    Spark-TTS-Lib
    </h1>
    <p>
    A Python library for <b><em><a href="https://github.com/SparkAudio/Spark-TTS">Spark-TTS</a></em></b>
    </p>
    <p>
    </p>
    <a href="https://huggingface.co/SparkAudio/Spark-TTS-0.5B"><img src="https://img.shields.io/badge/Hugging%20Face-Model%20Page-yellow" alt="Hugging Face"></a>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/Platform-linux-lightgrey" alt="version"></a>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/Python-3.12+-orange" alt="version"></a>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/PyTorch-2.5+-brightgreen" alt="python"></a>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="mit"></a>
</div>


## Install 

```bash
pip install spark-tts-lib
```


## Usage

Model download:

```python
from spark_tts_lib.download import download_pretrained_model

download_pretrained_model()
```

> The model will be downloaded to the `pretrained_models/Spark-TTS-0.5B` directory in the current directory. You can also specify a directory to save it.
> 
> ```python
> download_pretrained_model(local_dir="/path/to/save/model")
> ```

Inference:

```python
from spark_tts_lib.SparkTTS import SparkTTS

# Initialize the model
model_dir = "pretrained_models/Spark-TTS-0.5B"
model = SparkTTS(model_dir)

# Perform inference and get the generated audio data
wav_data = model.inference(
    text="This is the text you want to synthesize into speech.",
    prompt_speech_path="prompt_audio.wav",
    prompt_text="This is the text corresponding to your reference audio.",
)

# Save or use the generated audio data
# ...
```

Set the `temperature`, `top_k`, `top_p` to control the generated audio:

```python
model.inference(
    text="...",
    prompt_speech_path="...",
    prompt_text="...",
    temperature=0.8,
    top_k=50,
    top_p=0.95,
)
```

> You can use more parameters and more components, please refer to the [Spark-TTS](https://github.com/SparkAudio/Spark-TTS) for more details.


## ⚠️ Usage Disclaimer

This project provides a zero-shot voice cloning TTS model intended for academic research, educational purposes, and legitimate applications, such as personalized speech synthesis, assistive technologies, and linguistic research.

Please note:

- Do not use this model for unauthorized voice cloning, impersonation, fraud, scams, deepfakes, or any illegal activities.

- Ensure compliance with local laws and regulations when using this model and uphold ethical standards.

- The developers assume no liability for any misuse of this model.

We advocate for the responsible development and use of AI and encourage the community to uphold safety and ethical principles in AI research and applications. If you have any concerns regarding ethics or misuse, please contact us.