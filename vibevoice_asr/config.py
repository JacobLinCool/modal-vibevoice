"""Runtime constants and `RunnerConfig` shared across providers."""

from __future__ import annotations

from dataclasses import dataclass

MODEL_NAME = "microsoft/VibeVoice-ASR"
QWEN_TOKENIZER = "Qwen/Qwen2.5-7B"

# WeSpeaker CAM++ ONNX (VoxCeleb-trained, 192-dim x-vector). Language-
# agnostic in practice; for pure Mandarin workloads swap in the 3D-Speaker
# zh-cn model:
#   https://github.com/k2-fsa/sherpa-onnx/releases/download/
#     speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
# Exported by sherpa-onnx; runs purely through onnxruntime and avoids the
# torchaudio dependency that blocks SpeechBrain in NVIDIA's NGC container.
SPEAKER_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/"
    "wespeaker_en_voxceleb_CAM++.onnx"
)
SPEAKER_MODEL_PATH = "/opt/spk_wespeaker_en_campplus.onnx"

# silero-vad ONNX (run via onnxruntime; avoids the torchaudio wheel
# that ABI-conflicts with NGC's PyTorch nightly).
SILERO_VAD_URL = (
    "https://github.com/snakers4/silero-vad/raw/v5.1.2/"
    "src/silero_vad/data/silero_vad.onnx"
)
SILERO_VAD_PATH = "/opt/silero_vad.onnx"

HF_CACHE_DIR = "/root/.cache/huggingface"
TARGET_SR = 24_000


@dataclass
class RunnerConfig:
    """Paths and names consumed across the three runner stages."""

    model_name: str = MODEL_NAME
    tokenizer_name: str = QWEN_TOKENIZER
    hf_cache_dir: str = HF_CACHE_DIR
    silero_vad_path: str = SILERO_VAD_PATH
    speaker_model_path: str = SPEAKER_MODEL_PATH
    target_sr: int = TARGET_SR
