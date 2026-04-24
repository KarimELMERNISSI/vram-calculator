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
var GPUS = {{ site.Data.gpus | jsonify | safeJS }};
