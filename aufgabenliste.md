# TODO List: PortraitForge

## 1. Metadata & Provenance
- [ ] **EXIF/PNG metadata integration**: Store prompt, model, resolution, and seed in the generated image.
- [ ] **AI labeling**: Visible watermark or robust invisible signature to clearly identify images as AI-generated.

## 2. Legal & Safety
- [ ] **License update**: Add usage terms (e.g., regarding deepfakes, copyright of generated training data/images).

## 3. UI / UX
- [ ] **Interface optimization**: Improve navigation, workflow, and result presentation for better usability.
- [ ] **Central parameter dashboard**: Consolidate all image generation settings in a central location (e.g., sidebar).

## 4. Pipeline & Model Integration
- [ ] **Implement missing providers (resolve stubs)**: Implement the current placeholders for `BiRefNet`, `SAM` (background), as well as `GFPGAN` and `SwinIR` (enhancer).
- [ ] **VRAM & stability tuning**: Refine VRAM management (offloading strategies) and user feedback (clear Gradio warnings) when `max_gen_scale` limits are applied on memory-intensive models.

## 5. Workflow & Features
- [ ] **Batch processing (batch generation)**: Generate multiple application photo variants (different outfits/backgrounds) in a single run.
- [ ] **Extended file logging**: Implement persistent logging for performance and error tracking.
