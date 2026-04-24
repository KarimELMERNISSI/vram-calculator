// ═══════════════════════════════════════════════════════════
// FORMULA DEFINITIONS (TeX)
// ═══════════════════════════════════════════════════════════

/**
 * STATIC_FORMULAS: Used in tooltips (popovers)
 */
var STATIC_FORMULAS = {
  total_params: "P_{total} = \\sum_{i} |\\theta_i|",
  active_params: "P_{active} = P_{embed} + L \\cdot (P_{attn} + k \\cdot P_{FFN})",
  architecture: "\\text{Dense}: P_{active} = P_{total} \\quad \\text{MoE}: P_{active} \\ll P_{total}",
  layers: "L = \\text{num\\_hidden\\_layers}",
  kv_heads: "\\text{KV heads} = n_{kv} \\leq n_{heads}",
  weight_precision: "\\text{Mem}_{weights} = P \\times b_{per\\_param}",
  kv_cache_precision: "\\text{KV mem} = 2 \\cdot L \\cdot n_{kv} \\cdot d_{head} \\cdot C \\cdot b_{kv}",
  context_length: "C = \\text{max\\_position\\_embeddings}",
  batch_size: "\\text{KV mem} \\propto B",
  concurrent_users: "\\text{Total KV} = \\text{KV per user} \\times U",
  weights_memory: "\\text{Mem}_{w} = P_{total} \\times b_{per\\_param}",
  kv_cache_memory: "\\text{KV} = 2 \\cdot L \\cdot n_{kv} \\cdot d_h \\cdot C \\cdot B \\cdot U \\cdot b_{kv}",
  overhead: "\\text{Overhead} \\approx 0.12 \\times \\text{Weights} \\;(\\text{inference})",
  ram_overflow: "\\text{RAM offload} = \\max(0, \\; \\text{Total} - \\text{VRAM})",
  prefill_speed: "\\text{Prefill} \\approx \\frac{\\text{Compute (FLOPS)}}{P_{active} \\times C \\times 2 \\times b}",
  decode_speed: "\\text{Decode} \\approx \\frac{BW}{P_{active} \\times 2 \\times b}",
  latency_per_tok: "\\text{Latency} = \\frac{1}{\\text{Decode speed}} \\times 1000 \\;\\text{ms}",
  ttft: "\\text{TTFT} = \\frac{P_{active} \\times C \\times 2 \\times b}{BW} \\times 1000 \\;\\text{ms}",
  time_100: "T_{100} = \\text{TTFT} + \\frac{100}{\\text{Decode speed}}",
  time_1000: "T_{1000} = \\text{TTFT} + \\frac{1000}{\\text{Decode speed}}",
  throughput: "\\text{Throughput} = \\text{Decode speed} \\times U",
  power_draw: "P = \\text{TDP} \\times \\frac{\\text{Utilization}}{100}",
  cost_per_kwh: "\\text{Cost/hr} = \\frac{P}{1000} \\times \\$\\text{/kWh}",
  gpu_util: "P_{actual} = \\text{TDP} \\times \\text{Util}\\%",
  co2: "\\text{CO}_2 = \\text{kWh} \\times \\text{Carbon intensity (kg CO}_2\\text{/kWh)}",
  pcie_bus: "\\text{BW}_{PCIe} = \\text{Gen} \\times \\text{lanes} \\times \\text{encoding rate}",
  bus_wall: "\\text{Bus Wall Ratio} = \\frac{\\text{BW}_{HBM}}{\\text{BW}_{transfer}}",
  gpu_compute: "\\text{Prefill tok/s} = \\frac{\\text{TFLOPS} \\times 10^{12}}{2 \\times P_{active} \\times b_{per\\_param}}",
  ram_bandwidth: "\\text{BW}_{RAM} = \\text{MT/s} \\times \\text{channels} \\times 8 \\;\\text{bytes}",
  nvlink_interconnect: "\\text{TP efficiency} = \\frac{T_{compute}}{T_{compute} + T_{communication}}",
  tensor_parallelism: "T_{decode} = \\frac{P_{active} \\times b}{N \\times BW_{HBM}} + \\frac{L \\times 2 \\times \\frac{N-1}{N} \\times h \\times 2}{BW_{interconnect}}",
  pcie_lanes: "\\text{BW}_{PCIe} = \\text{BW}_{per\\_lane} \\times \\text{lanes} \\times \\eta_{PCIe}",
  pcie_efficiency: "\\text{BW}_{effective} = \\text{BW}_{theoretical} \\times \\eta \\approx 0.90 \\times \\text{BW}_{theoretical}",
  ram_efficiency: "\\text{BW}_{RAM,effective} = \\text{BW}_{theoretical} \\times \\eta_{RAM} \\approx 0.85 \\times \\text{BW}_{theoretical}",
  nvswitch: "\\text{All-reduce}_{NVSwitch} = 2 \\times \\frac{h \\times 2}{BW_{NVLink}}",
  gpu_hbm: "\\text{Decode TPS} = \\frac{BW_{HBM}}{P_{active} \\times b_{param}}",
  roofline_model: "\\text{Arithmetic Intensity} = \\frac{\\text{FLOPs}}{\\text{Bytes accessed}} = \\frac{2}{b}",
  numa_effects: "\\text{BW}_{remote} \\approx 0.5 \\times \\text{BW}_{local}",
  pipeline_parallelism: "T_{PP} = \\sum_{i=1}^{N} T_{compute,i} + (N-1) \\times T_{bubble}",
};

/**
 * RESULT_TEMPLATES: Used in the "Calculated Formulas" section of the UI.
 * Placeholders like {p}, {qW} are replaced dynamically.
 */
var RESULT_TEMPLATES = {
  weights: "W = P_{total} \\times b = {p} \\text{B} \\times {qW} \\text{ B/p} = {wGB}",
  kv_cache_full: "KV = 2 \\times {L} \\times {nkv} \\times {hd} \\times {ctx} \\times {bs} \\times {uS} \\times {qKV} = {kvGB}",
  kv_cache_approx: "KV \\approx P \\times 0.12 \\times \\frac{C}{4096} \\times B \\times U = {kvGB}",
  overhead: "\\text{Overhead} = {ovF} \\times W = {ovGB}",
  total: "\\text{Total} = {wGB} + {kvGB} + {ovGB} {ramNote} / {totV} \\text{ available}",
  pcie_eff: "\\text{PCIe}_{eff} = {pcieTheoretical} \\times {pcieLanes}/16 \\times {PCIE_EFFICIENCY} = {pcieEffective} \\text{ GB/s}",
  ram_eff: "\\text{RAM}_{eff} = {ramTheoretical} \\times {RAM_EFFICIENCY} \\times {numaFactor} = {ramEffective} \\text{ GB/s}",
  bus_wall: "\\text{Bus Wall} = \\frac{{gpuBw}}{\\min({pcieEffective}, {ramEffective})} = \\frac{{gpuBw}}{{transferBW}} = {busWallRatio}\\times",
};
