# Third-party notices

The root MIT license covers this repository's original software and accompanying documentation. It does not replace third-party terms.

- **Diffusion Policy source** in `vendor/diffusion_policy/`: copyright 2023 Columbia Artificial Intelligence and Robotics Lab, MIT. The original license is in [vendor/LICENSE](vendor/LICENSE). [vendor/provenance.json](vendor/provenance.json) records the upstream repository, immutable revision, and source hashes.
- **DDPM update** in `gpu_extension/batched.py`: follows Hugging Face Diffusers 0.27.2 `DDPMScheduler.step`, copyright 2024 UC Berkeley Team and The HuggingFace Team, Apache-2.0. See [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt). The local specialization uses precomputed coefficients and independent per-candidate noise; it is tested against the pinned upstream scheduler.
- **External policy checkpoint and Push-T demonstrations**: downloaded from the official Diffusion Policy host through the URLs in `assets.json`, not redistributed here. These external assets retain the rights and terms of their providers. The repository's MIT license does not grant new rights over them.
- **Fonts**: no third-party font files are bundled. Committed figures contain their existing embedded font subsets. Regeneration uses locally installed fonts.

Reference research outputs are provided to document and reproduce the study. The code's MIT grant does not assign ownership of third-party content.
