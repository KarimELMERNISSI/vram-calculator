// ============================================================================
// HARDWARE & QUANTIZATION CONSTANTS
// ============================================================================

// PCIe bandwidth per lane (GB/s, theoretical peak with 128b/130b encoding)
var PCIE_BW_PER_LANE = { 3: 0.985, 4: 1.969, 5: 3.938 };
// PCIe x16 convenience lookup (GB/s, theoretical)
var PCIE_BW = { 3: 15.75, 4: 31.5, 5: 63.0 };
// PCIe practical efficiency for large DMA transfers (TLP overhead, protocol)
var PCIE_EFFICIENCY = 0.9;
// RAM bandwidth lookup (theoretical peak, GB/s)
var RAM_BW = {};
{{ range site.Data.ram }}
RAM_BW["{{ .id }}"] = {{ .bw }};
{{ end }}
// RAM practical efficiency for sequential reads
var RAM_EFFICIENCY = 0.85;
// NVLink / Interconnect latency for all-reduce (seconds)
var INTERCONN_LATENCY = {
  nvswitch: 5e-6,   // NVSwitch: ~5 us (shortest path, all-to-all switch)
  nvlink: 10e-6,    // NVLink P2P: ~10 us (direct link, ring all-reduce)
  pcie: 40e-6,      // PCIe P2P: ~40 us (higher latency, through root complex)
  none: 100e-6,     // Through CPU: ~100 us (GPU -> CPU -> GPU)
};

// ============================================================================
// GGUF QUANT LEVEL MAP
// ============================================================================
var GGUF_QUANTS = {
  q2_k:   { label: "Q2_K",   bpb: 0.32,  desc: "2-bit quant, K-quants" },
  q3_k_s: { label: "Q3_K_S", bpb: 0.34,  desc: "3-bit small" },
  q3_k_m: { label: "Q3_K_M", bpb: 0.43,  desc: "3-bit medium" },
  q3_k_l: { label: "Q3_K_L", bpb: 0.45,  desc: "3-bit large" },
  q4_0:   { label: "Q4_0",   bpb: 0.56,  desc: "4-bit base" },
  q4_k_s: { label: "Q4_K_S", bpb: 0.575, desc: "4-bit small K-quant" },
  q4_k_m: { label: "Q4_K_M", bpb: 0.6,   desc: "4-bit medium K-quant" },
  q5_0:   { label: "Q5_0",   bpb: 0.675, desc: "5-bit base" },
  q5_k_s: { label: "Q5_K_S", bpb: 0.69,  desc: "5-bit small K-quant" },
  q5_k_m: { label: "Q5_K_M", bpb: 0.71,  desc: "5-bit medium K-quant" },
  q6_k:   { label: "Q6_K",   bpb: 0.825, desc: "6-bit K-quant" },
  q8_0:   { label: "Q8_0",   bpb: 1.06,  desc: "8-bit quant (near FP16)" },
  f16:    { label: "F16",    bpb: 2.0,   desc: "Half precision" },
  f32:    { label: "F32",    bpb: 4.0,   desc: "Full precision" },
  bf16:   { label: "BF16",   bpb: 2.0,   desc: "BFloat16" },
};

var GGUF_ORDER = [
  "q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_k_s", "q4_k_m",
  "q5_0", "q5_k_s", "q5_k_m", "q6_k", "q8_0", "bf16", "f16", "f32",
];
