from pipeline.registry import registry

# Registrierte Provider (nicht implementierte Provider hier NICHT eintragen)
from pipeline.providers.generator.photomaker import PhotoMakerProvider
from pipeline.providers.generator.sdxl_base import SDXLBaseProvider
from pipeline.providers.generator.sd35_medium import SD35MediumProvider
from pipeline.providers.generator.flux import FluxProvider
from pipeline.providers.enhancer.codeformer import CodeFormerProvider
from pipeline.providers.enhancer.realesrgan import RealESRGANProvider
from pipeline.providers.background.u2net import U2NetProvider

registry.register_generator("photomaker", PhotoMakerProvider)
registry.register_generator("sdxl_base", SDXLBaseProvider)
registry.register_generator("sd35_medium", SD35MediumProvider)
registry.register_generator("flux_schnell", FluxProvider)
registry.register_face_restore("codeformer", CodeFormerProvider)
registry.register_upscale("realesrgan", RealESRGANProvider)
registry.register_background("u2net", U2NetProvider)
