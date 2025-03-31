<div align="center">
    <h1>
    Spark-TTS-Lib
    </h1>
    <p>
    A Python package for <b><em><a href="https://github.com/SparkAudio/Spark-TTS">Spark-TTS</a></em></b>
    </p>
    <p>
    </p>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/Python-3.10+-orange" alt="version"></a>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/PyTorch-2.5+-brightgreen" alt="python"></a>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="mit"></a>
</div>


## 📦 Install 

```bash
pip install spark-tts-lib
```

## 🌟 Sample Example

```python
from spark_tts_lib import SparkTTS

model = SparkTTS()
wav_data = model.inference(text="Hello, world!")
```

## 🚀 Usage

**Download the pretrained model:**

```python
from spark_tts_lib import download_pretrained_model

download_pretrained_model()
```

> The model will be downloaded to the `pretrained_models/Spark-TTS-0.5B` directory in the current directory. You can also specify a directory to save it.
> 
> ```python
> download_pretrained_model(local_dir="/path/to/save/model")
> ```
> You can also download the model from the [Hugging Face](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) page.

**Import SparkTTS:**

```python
from spark_tts_lib import SparkTTS
```

**Initialize the model:**

```python
model = SparkTTS()
```

> If you want to specify the model directory, you can do it like this:
> 
> ```python
> model_dir = "pretrained_models/Spark-TTS-0.5B"
> model = SparkTTS(model_dir)
> ```

**Perform voice creation inference:**

```python
text = "This is the text you want to synthesize into speech."
gender = "female" # "male"
pitch = "high" # "very_low" | "low" | "moderate" | "high" | "very_high"
speed = "high" # "very_low" | "low" | "moderate" | "high" | "very_high"

wav_data = model.inference(text=text, gender=gender, pitch=pitch, speed=speed)
```

**Perform voice cloning inference:**

```python
text = "This is the text you want to synthesize into speech."
prompt_speech_path = "prompt_audio.wav"
prompt_text = "This is the text corresponding to your reference audio."

wav_data = model.inference(
    text=text,
    prompt_speech_path=prompt_speech_path,
    prompt_text=prompt_text,
)
```

**Inference with more parameters:**

```python
model.inference(
    ...
    temperature=0.8,
    top_k=50,
    top_p=0.95,
)
```

## 📚 More information

Please refer to the [Spark-TTS](https://github.com/SparkAudio/Spark-TTS) for more details.


## ⚠️ Usage Disclaimer

This project provides a zero-shot voice cloning TTS model intended for academic research, educational purposes, and legitimate applications, such as personalized speech synthesis, assistive technologies, and linguistic research.

Please note:

- Do not use this model for unauthorized voice cloning, impersonation, fraud, scams, deepfakes, or any illegal activities.

- Ensure compliance with local laws and regulations when using this model and uphold ethical standards.

- The developers assume no liability for any misuse of this model.

We advocate for the responsible development and use of AI and encourage the community to uphold safety and ethical principles in AI research and applications. If you have any concerns regarding ethics or misuse, please contact us.