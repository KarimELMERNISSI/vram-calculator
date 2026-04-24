// ═══════════════════════════════════════════════════════════
// MODEL PRESETS
// ═══════════════════════════════════════════════════════════
var MODELS = [
  { name: "Llama 3.1 8B", p: 8, L: 32, h: 4096, nh: 32, nkv: 8, arch: "Dense" },
  { name: "Llama 3.1 70B", p: 70, L: 80, h: 8192, nh: 64, nkv: 8, arch: "Dense" },
  { name: "Llama 3.1 405B", p: 405, L: 126, h: 16384, nh: 128, nkv: 8, arch: "Dense" },
  { name: "Mistral 7B", p: 7, L: 32, h: 4096, nh: 32, nkv: 8, arch: "Dense" },
  { name: "Mixtral 8x7B", p: 47, L: 32, h: 4096, nh: 32, nkv: 8, arch: "MoE", active: 13 },
  { name: "Mixtral 8x22B", p: 141, L: 56, h: 6144, nh: 48, nkv: 8, arch: "MoE", active: 39 },
  { name: "Qwen2.5 72B", p: 72, L: 80, h: 8192, nh: 64, nkv: 8, arch: "Dense" },
  { name: "DeepSeek R1 671B", p: 671, L: 61, h: 7168, nh: 128, nkv: 128, arch: "MoE", active: 37 },
  { name: "Qwen 3.6 35B A3B", p: 35, L: 40, h: 2048, nh: 16, nkv: 2, arch: "MoE", active: 3 },
  { name: "Gemma 4 26B A4B", p: 25.2, L: 30, h: 2560, nh: 32, nkv: 16, arch: "MoE", active: 3.8 },
  { name: "Gemma 4 31B", p: 30.7, L: 62, h: 3584, nh: 28, nkv: 16, arch: "Dense" },
  { name: "Phi-3 Mini 3.8B", p: 3.8, L: 32, h: 3072, nh: 32, nkv: 32, arch: "Dense" },
];

// ═══════════════════════════════════════════════════════════
// GPU PRESETS
// ═══════════════════════════════════════════════════════════
var GPUS = [
  { name: "NVIDIA H200 SXM", vram: 141, bw: 4800, tdp: 700, tflops: 990, pcieGen: 5, pcieLanes: 16, nvlink: 900, nvswitch: true },
  { name: "NVIDIA H100 SXM", vram: 80, bw: 3350, tdp: 700, tflops: 990, pcieGen: 5, pcieLanes: 16, nvlink: 900, nvswitch: true },
  { name: "NVIDIA H100 PCIe", vram: 80, bw: 2000, tdp: 350, tflops: 756, pcieGen: 5, pcieLanes: 16, nvlink: 0, nvswitch: false },
  { name: "NVIDIA A100 80GB", vram: 80, bw: 2000, tdp: 300, tflops: 312, pcieGen: 4, pcieLanes: 16, nvlink: 600, nvswitch: true },
  { name: "NVIDIA A100 40GB", vram: 40, bw: 1555, tdp: 250, tflops: 312, pcieGen: 4, pcieLanes: 16, nvlink: 600, nvswitch: true },
  { name: "NVIDIA A6000 Ada", vram: 48, bw: 960, tdp: 300, tflops: 182, pcieGen: 4, pcieLanes: 16, nvlink: 0, nvswitch: false },
  { name: "NVIDIA RTX 4090", vram: 24, bw: 1008, tdp: 450, tflops: 165, pcieGen: 4, pcieLanes: 16, nvlink: 0, nvswitch: false },
  { name: "NVIDIA RTX 3090", vram: 24, bw: 936, tdp: 350, tflops: 71, pcieGen: 4, pcieLanes: 16, nvlink: 0, nvswitch: false },
  { name: "NVIDIA L40S", vram: 48, bw: 864, tdp: 350, tflops: 366, pcieGen: 4, pcieLanes: 16, nvlink: 0, nvswitch: false },
  { name: "AMD MI300X", vram: 192, bw: 5300, tdp: 750, tflops: 1307, pcieGen: 5, pcieLanes: 16, nvlink: 400, nvswitch: false },
  { name: "AMD MI250X", vram: 128, bw: 3276, tdp: 560, tflops: 383, pcieGen: 4, pcieLanes: 16, nvlink: 400, nvswitch: false },
  { name: "Custom GPU", vram: 0, bw: 0, tdp: 0, tflops: 0, pcieGen: 4, pcieLanes: 16, nvlink: 0, nvswitch: false },
];
