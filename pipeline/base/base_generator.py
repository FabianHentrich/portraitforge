from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from PIL import Image


@dataclass
class GeneratorConfig:
    num_steps: int = 30
    guidance_scale: float = 5.0
    seed: int | None = None
    height: int = 1024
    width: int = 1024
    gen_scale: float = 1.0    # Generation resolution = width/height × gen_scale (1.0 = full resolution, no LANCZOS quality loss)
    scheduler: str = "euler"   # euler | dpm++2m | dpm++2m_karras | euler_a | ddim
    extra: dict = field(default_factory=dict)
    # Provider-spezifische Parameter kommen in extra{}


class BaseGeneratorProvider(ABC):
    name: str               # Anzeigename in Gradio
    model_id: str           # HuggingFace-ID oder lokaler Pfad
    vram_gb: float          # Estimated VRAM requirement
    requires_reference: bool
    prompt_hint: str        # Kurzbeschreibung unter dem Prompt-Feld
    max_prompt_tokens: int = 77         # CLIP-Limit; FLUX/T5 = 512
    negative_prompt_hint: str = ""      # Empfohlene Negative-Prompt-Bausteine
    prompt_template: str = ""           # Beispiel-Prompt-Struktur
    cpu_offload: bool = False           # Text-Encoder etc. in RAM auslagern (spart VRAM, kostet Geschwindigkeit)
    vram_gb_offloaded: float = 0.0     # Estimated VRAM with active CPU offload (0 = same as vram_gb)
    ram_gb: float = 0.0                # Estimated additional RAM requirement (CPU offload weights, caches)
    max_gen_scale: float = 1.0          # Provider-specific maximum for gen_scale (VRAM-dependent)

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        config: GeneratorConfig,
        reference_images: list[Image.Image] | None = None,
    ) -> Image.Image: ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...
