# Local LLM Resource Estimator

A professional VRAM estimation tool for Large Language Models, built with [Hugo](https://gohugo.io/).

## Features

- **Hugging Face Integration** — Import model metadata directly from HF Hub with autocomplete search
- **Architecture-Aware Calculations** — Supports Dense and MoE models with GQA/MQA KV cache sizing
- **Quantization Support** — GGUF, GPTQ, AWQ, EXL2, BitsAndBytes/NF4 formats with auto-detection
- **Multi-GPU Parallelism** — Tensor Parallelism (NVLink/NVSwitch) and Pipeline Parallelism modeling
- **RAM Offloading** — Bus Wall analysis with PCIe/RAM bandwidth bottleneck detection
- **Roofline Performance Model** — Prefill speed, decode speed, TTFT, latency, and throughput estimates
- **Power & Cost Estimation** — Electricity costs, CO₂ emissions, and cost-per-token calculations

## Quick Start

```bash
# Run the development server
hugo server -D

# Build for production
hugo --minify
```

The site will be available at [http://localhost:1313/vram-calculator/](http://localhost:1313/vram-calculator/)

## Project Structure

```
├── hugo.toml                       # Hugo configuration
├── assets/
│   ├── css/
│   │   └── calculator.css          # All styles
│   └── js/
│       ├── data/
│       │   ├── models.js           # Model & GPU presets
│       │   ├── constants.js        # Hardware constants (PCIe, RAM, GGUF)
│       │   └── popovers.js         # Tooltip definitions
│       ├── hf-api.js               # HF API integration & config parsing
│       ├── variants.js             # Quantized variant discovery
│       ├── popover.js              # Tooltip popover system
│       ├── calculator.js           # Main VRAM calculation engine
│       └── main.js                 # Initialization & UI wiring
├── content/
│   └── _index.md                   # Homepage content
├── layouts/
│   ├── index.html                  # Base template (asset pipeline)
│   └── partials/
│       ├── hf-import.html          # HF import panel
│       ├── calculator-inputs.html  # Model/hardware/quantization controls
│       ├── memory-use.html         # Memory use bar & breakdown
│       ├── performances-estimation.html  # Performances estimation
│       └── power-cost-estimation.html     # Power & cost estimation
└── static/                         # Static assets (future use)
```

## GPU Database

Includes presets for: H200, H100 (SXM/PCIe), A100 (40/80GB), A6000 Ada, RTX 4090, RTX 3090, L40S, MI300X, MI250X, plus custom GPU support.

## Model Presets

Built-in presets for: Llama 3.1 (8B/70B/405B), Mistral 7B, Mixtral 8x7B/8x22B, Qwen 2.5 72B, DeepSeek R1 671B, Qwen 3.6 35B-A3B, Gemma 4 (26B-A4B/31B), Phi-3 Mini 3.8B.

## Documentation

Technical documentation (LaTeX sources and PDFs) is available in the [`docs/`](./docs/) directory:
- [LaTeX Sources](./docs/latex/documentation.tex)
- [PDF Documentation](./docs/pdf/)

## License

GNU GPL (General Public License)

