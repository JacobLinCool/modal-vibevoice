"""Provider-agnostic core for Microsoft VibeVoice-ASR.

Wrap `VibeVoiceASRRunner` in any deploy target (Modal, RunPod, Docker, local)
and reuse its three-stage lifecycle: prefetch_weights → load → transcribe.
"""

from .config import RunnerConfig
from .runner import VibeVoiceASRRunner

__all__ = ["RunnerConfig", "VibeVoiceASRRunner"]
