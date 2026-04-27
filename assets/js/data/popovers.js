// ═══════════════════════════════════════════════════════════
// POPOVER DEFINITIONS
// ═══════════════════════════════════════════════════════════

var POPDEFS = {
  total_params: {
    title: "Total Parameters",
    explain: "The total number of learnable parameters in the model, including all MoE experts. This determines the total memory needed to store model weights.",
    refs: "Typical: 7B (7B), Llama 3.1 70B (70B), GPT-4 (~1.8T estimated). For MoE, total \u2260 active.",
  },
  active_params: {
    title: "Active Parameters",
    explain: "Parameters actually used per forward pass. For dense models, active = total. For MoE, only the routed experts (k of n) are active, so active &lt; total.",
    refs: "Mixtral 8x7B: 47B total, 13B active. DeepSeek R1: 671B total, 37B active.",
  },
  architecture: {
    title: "Architecture (Dense vs MoE)",
    explain: "Dense models activate all parameters every token. Mixture-of-Experts (MoE) models activate only a subset (k experts) per token, giving better quality per compute.",
    refs: "Dense: GPT-3, Llama, Mistral. MoE: Mixtral, DeepSeek, Qwen MoE variants.",
  },
  layers: {
    title: "Transformer Layers",
    explain: "The number of transformer blocks stacked in the model. More layers generally means deeper representations but more memory and compute.",
    refs: "Typical: 7B ~32L, 70B ~80L, 405B ~126L. Affects both KV cache size and compute.",
  },
  kv_heads: {
    title: "KV Heads (GQA)",
    explain: "Number of key-value heads in Grouped-Query Attention. When n_kv < n_heads, multiple query heads share the same KV head (GQA), reducing KV cache size.",
    refs: "Llama 3.1 70B: 64 query heads, 8 KV heads (8:1 GQA). Full MHA: n_kv = n_heads. MQA: n_kv = 1.",
  },
  weight_precision: {
    title: "Weight Precision",
    explain: "Number of bytes used to store each weight parameter. Lower precision reduces memory but may degrade quality. Quantization maps FP16/BF16 weights to fewer bits.",
    refs: "FP32: 4B, BF16/FP16: 2B, INT8: 1B, NF4/Q4: 0.5B per param. GGUF formats vary from 0.32 (Q2_K) to 1.06 (Q8_0).",
  },
  kv_cache_precision: {
    title: "KV Cache Precision",
    explain: "Precision of cached key-value pairs. FP16 is default; FP8/INT8 halves KV memory; Q4 quarters it with some quality loss.",
    refs: "FP16 (2B) is standard. FP8/INT8 (1B) offers good trade-off. Q4 (0.5B) is aggressive.",
  },
  context_length: {
    title: "Context Length",
    explain: "Maximum number of input tokens the model can process in a single forward pass. KV cache memory scales linearly with context length.",
    refs: "Llama 3.1: 128K. GPT-4: 128K. Gemini: 1M+. Longer context = more KV cache memory.",
  },
  batch_size: {
    title: "Batch Size",
    explain: "Number of sequences processed simultaneously. Each sequence in a batch requires its own KV cache. Larger batches improve throughput but use more memory.",
    refs: "Inference: typically 1-8. Training: 16-256+. Directly multiplies KV cache memory.",
  },
  concurrent_users: {
    title: "Concurrent Users",
    explain: "Number of simultaneous users generating tokens. Each user needs a separate KV cache. Affects both memory and total throughput.",
    refs: "1 user = minimal KV. 32 users = 32\u00d7 KV memory but better hardware utilization via batching.",
  },
  weights_memory: {
    title: "Weights Memory",
    explain: "Memory to store all model weights. This is the dominant cost for large models. Quantization directly reduces this.",
    refs: "Llama 70B in FP16: 140 GB. In Q4: 35 GB. In NF4: 35 GB.",
  },
  kv_cache_memory: {
    title: "KV Cache Memory",
    explain: "Memory for cached key-value pairs during generation. Scales with layers, KV heads, head dimension, context, batch, users, and precision. GQA reduces this significantly.",
    refs: "Llama 70B, 4K ctx, 1 user, FP16: ~1.6 GB. Same at 128K ctx: ~52 GB.",
  },
  overhead: {
    title: "Overhead Memory",
    explain: "Additional GPU memory for CUDA kernels, activation buffers, temporary tensors, and framework overhead. Typically 10-15% of weights memory for inference.",
    refs: "vLLM/llama.cpp typically 10-15%. Fine-tuning with LoRA: ~40% of weights. Full fine-tune: ~2\u00d7 weights.",
  },
  ram_overflow: {
    title: "RAM Overflow / Offloading",
    explain: "When total required memory exceeds GPU VRAM, layers can be offloaded to system RAM. This allows running models that don't fit, but RAM-offloaded layers are 10-50\u00d7 slower due to PCIe bandwidth limits.",
    refs: "Typical RAM bandwidth: 50-100 GB/s vs GPU: 900-4800 GB/s. Use only when no GPU alternative exists.",
  },
  prefill_speed: {
    title: "Prefill Speed",
    explain: "Speed of processing the initial prompt (prefill phase). Typically compute-bound, limited by GPU FLOPS. Faster with fewer active params.",
    refs: "H100: ~990 TFLOPS FP16. A100: ~312 TFLOPS. RTX 4090: ~165 TFLOPS.",
  },
  decode_speed: {
    title: "Decode Speed (per user)",
    explain: "Token generation speed per user during autoregressive decode. Bandwidth-bound: each step loads all weights but generates 1 token. The primary throughput metric.",
    refs: "H200 at 4800 GB/s with 8B Q4: ~750 tok/s. Same with 70B Q4: ~86 tok/s.",
  },
  latency_per_tok: {
    title: "Latency per Token",
    explain: "Time to generate a single token during decode. Inverse of decode speed. Lower is better for interactive use.",
    refs: "Interactive needs: <50ms/tok (20+ tok/s). Reading speed: ~5 tok/s is sufficient.",
  },
  ttft: {
    title: "Time to First Token (TTFT)",
    explain: "Time to process the full prompt and generate the first token. Scales linearly with context length. Bandwidth-bound estimate.",
    refs: "Short prompts (128 tokens): 10-100ms. Long prompts (32K tokens): 1-30s. Critical for chat UX.",
  },
  time_100: {
    title: "Time to Generate 100 Tokens",
    explain: "Total wall time for TTFT plus generating 100 tokens. Good benchmark for typical chat responses.",
    refs: "Short chat responses: 50-200 tokens. This metric represents a concise answer.",
  },
  time_1000: {
    title: "Time to Generate 1000 Tokens",
    explain: "Total wall time for TTFT plus generating 1000 tokens. Represents a long generation task.",
    refs: "Useful for estimating batch job times, long-form generation, or RAG with detailed answers.",
  },
  throughput: {
    title: "Multi-user Throughput",
    explain: "Total tokens per second across all concurrent users. Scales linearly with users until memory or compute saturation.",
    refs: "1 user at 50 tok/s = 50 tok/s total. 8 users at 50 tok/s each = 400 tok/s (if memory permits).",
  },
  power_draw: {
    title: "Power Draw",
    explain: "Actual power consumption based on GPU Thermal Design Power and utilization rate. Not all workloads use 100% of the GPU.",
    refs: "H100 TDP: 700W. RTX 4090 TDP: 450W. A100 TDP: 300W. Idle: ~50-100W.",
  },
  cost_per_kwh: {
    title: "Electricity Cost",
    explain: "Hourly electricity cost. Power (W) converted to kW, multiplied by local electricity rate. Varies dramatically by region.",
    refs: "US avg: $0.12/kWh. EU avg: $0.25/kWh. China: $0.08/kWh. Hawaii: $0.40/kWh.",
  },
  gpu_util: {
    title: "GPU Utilization",
    explain: "Percentage of GPU compute actually being used. LLM inference is often memory-bandwidth-bound, not compute-bound, so utilization may be 60-90% during decode.",
    refs: "Inference decode: 60-90%. Inference prefill: 80-100%. Training: 90-100%. Idle: 5-15%.",
  },
  co2: {
    title: "CO\u2082 Emissions",
    explain: "Carbon dioxide emissions from electricity generation. Depends on the energy mix of your power grid. Coal-heavy grids produce much more CO\u2082 than nuclear/renewable.",
    refs: "World avg: 0.417 kg CO\u2082/kWh. France (nuclear): 0.056. Poland (coal): 0.769. Sweden (hydro): 0.045.",
  },
  pcie_bus: {
    title: "PCIe Bus & Bandwidth",
    explain: "PCIe (Peripheral Component Interconnect Express) is the primary data highway between GPU and CPU/RAM. Each generation roughly doubles bandwidth per lane. The bandwidth listed is theoretical peak for x16 configuration; practical throughput is ~80-90% due to protocol overhead and TLP headers.",
    refs: "Gen3 x16: 15.75 GB/s. Gen4 x16: 31.5 GB/s. Gen5 x16: 63.0 GB/s. SXM GPUs bypass PCIe for GPU-GPU via NVLink.",
  },
  bus_wall: {
    title: "Le Mur du Bus (Bus Wall)",
    explain: "The \"Bus Wall\" is the fundamental bottleneck when data must traverse the PCIe bus instead of staying in GPU HBM. The ratio of HBM bandwidth to transfer bandwidth tells you how many times slower RAM-offloaded layers are compared to VRAM-resident layers. A ratio of 100 means offloaded layers run 100x slower.",
    refs: "H100 SXM + Gen4: 3350/31.5 = 106x. RTX 4090 + Gen4: 1008/31.5 = 32x. H200 + Gen5: 4800/63 = 76x.",
  },
  gpu_compute: {
    title: "GPU Compute (TFLOPS)",
    explain: "GPU compute performance in FP16/BF16 TFLOPS determines prefill speed (compute-bound phase). Each token requires 2 FLOPs per active parameter (multiply + accumulate). Modern GPUs achieve 70-312 TFLOPS (A100) up to 990+ TFLOPS (H100) in FP16 with tensor cores.",
    refs: "H100: 990 TFLOPS. A100: 312. RTX 4090: 165. MI300X: 1307. These are peak tensor core FP16/BF16 numbers.",
  },
  ram_bandwidth: {
    title: "System RAM Bandwidth",
    explain: "CPU memory bandwidth determines how fast data can be read from RAM. For RAM-offloaded layers, the effective transfer bandwidth is min(BW_RAM, BW_PCIe) since data must traverse both. DDR5 roughly doubles DDR4 bandwidth per channel.",
    refs: "DDR4-3200 2-ch: 51.2 GB/s. DDR5-5600 2-ch: 89.6 GB/s. DDR5-5600 8-ch (server): 358.4 GB/s. PCIe Gen4 x16 is usually the bottleneck.",
  },
  nvlink_interconnect: {
    title: "NVLink / Inter-GPU Interconnect",
    explain: "NVLink is NVIDIA's high-speed inter-GPU interconnect, far faster than PCIe. It enables efficient Tensor Parallelism (TP) by reducing all-reduce communication overhead. Without NVLink, multi-GPU TP uses PCIe which severely limits scaling efficiency.",
    refs: "NVLink 4 (H100): 900 GB/s. NVLink 3 (A100): 600 GB/s. PCIe Gen4 x16: 31.5 GB/s. AMD uses Infinity Fabric (~400 GB/s on MI300X).",
  },
  tensor_parallelism: {
    title: "Tensor Parallelism",
    explain: "Tensor Parallelism splits model weights across N GPUs. Each GPU holds 1/N of weights and reads from its local HBM. After each layer, an all-reduce synchronizes partial results. The communication cost depends on hidden size, layers, and interconnect bandwidth. NVLink makes TP highly efficient; PCIe limits scaling.",
    refs: "2x H100 NVLink, 70B Q4: ~92% efficiency. 2x A100 NVLink: ~88%. 2x RTX 4090 PCIe: ~55-65% (not recommended for TP).",
  },
  pcie_lanes: {
    title: "PCIe Lane Width (x4 / x8 / x16)",
    explain: "PCIe bandwidth scales linearly with the number of lanes. A x8 connection provides exactly half the bandwidth of x16. Most desktop GPUs use x16, but some motherboards reduce to x8 when multiple slots are populated. SXM form factors bypass PCIe entirely for GPU-GPU communication via NVLink/NVSwitch.",
    refs: "Gen4 x16: 31.5 GB/s (theoretical). Gen4 x8: 15.75 GB/s. Gen4 x4: 7.88 GB/s. Practical efficiency: ~90% for large DMA transfers.",
  },
  pcie_efficiency: {
    title: "PCIe Practical Efficiency",
    explain: "Theoretical PCIe bandwidth is never fully achieved. Protocol overhead (TLP headers, DLLP, ACK/NAK), transaction layer framing, and root complex latency reduce practical throughput. For large sequential DMA transfers (typical of weight loading), efficiency reaches ~90%. For small random transfers, it can drop below 50%.",
    refs: "Large DMA (weight loading): 88-95% efficiency. Small packets: 40-70%. The 0.90 factor used here is conservative for realistic estimates.",
  },
  ram_efficiency: {
    title: "RAM Practical Efficiency",
    explain: "Theoretical RAM bandwidth assumes perfect sequential access with zero refresh overhead. In practice, DRAM refresh cycles, row misses, and memory controller scheduling reduce achievable bandwidth to about 80-88% of theoretical peak for large sequential reads. NUMA effects on multi-socket systems can further reduce this by 30-50% for cross-socket access.",
    refs: "Sequential read: 82-88% of theoretical. Random access: 40-60%. Cross-NUMA: 50-70% of local bandwidth. The 0.85 factor is a conservative average.",
  },
  nvswitch: {
    title: "NVSwitch (All-to-All Topology)",
    explain: "NVSwitch is NVIDIA's switching fabric that provides full all-to-all connectivity between all GPUs in a node. Unlike point-to-point NVLink (ring topology), NVSwitch allows any GPU to communicate with any other GPU at full NVLink speed simultaneously. This dramatically improves Tensor Parallelism efficiency for 4+ GPUs by reducing all-reduce hops.",
    refs: "DGX H100: 4 NVSwitches connecting 8 H100s at 900 GB/s each. DGX A100: 6 NVSwitches at 600 GB/s. Without NVSwitch, 8-GPU ring all-reduce requires 7 hops.",
  },
  gpu_hbm: {
    title: "GPU HBM (High Bandwidth Memory)",
    explain: "HBM is the GPU's on-board memory, connected via an ultra-wide bus (3072-6144 bits) to the GPU die. It provides 900-4800 GB/s bandwidth, orders of magnitude faster than any external interconnect. LLM decode is almost always HBM-bandwidth-bound: each token generation requires reading ALL weights from HBM.",
    refs: "H200: 4800 GB/s (HBM3e). H100: 3350 GB/s (HBM3). A100: 2000 GB/s (HBM2e). RTX 4090: 1008 GB/s (GDDR6X). This is THE bottleneck for decode speed.",
  },
  roofline_model: {
    title: "Roofline Performance Model",
    explain: "The roofline model determines whether inference is compute-bound or bandwidth-bound. For LLM decode, arithmetic intensity is 2/b (FLOPs per byte), which at Q4 is only 4 FLOP/byte \u2014 far below the GPU's ridge point (~100 FLOP/byte for H100). This confirms decode is deeply bandwidth-bound. Prefill has much higher arithmetic intensity due to batched matrix operations.",
    refs: "H100 ridge point: ~120 FLOP/byte (312 TFLOPS / 2600 GB/s effective). Q4 decode at 4 FLOP/byte is 30x below ridge. Q16 prefill at ~80 FLOP/byte approaches ridge.",
  },
  numa_effects: {
    title: "NUMA & Multi-Socket Effects",
    explain: "On multi-socket servers (common with AMD EPYC, Intel Xeon), each CPU socket has its own memory controller. Accessing RAM attached to the local socket is fast; accessing remote socket memory (via inter-socket link like AMD Infinity Fabric or Intel UPI) adds latency and reduces bandwidth by 30-50%. For RAM offloading, ensure weights are allocated on the NUMA node closest to the GPU.",
    refs: "AMD EPYC 9654: 460 GB/s local, ~200 GB/s cross-socket. Intel Xeon w9-3495X: ~410 GB/s local, ~180 GB/s cross-socket. Always pin GPU and RAM to the same NUMA node.",
  },
  pipeline_parallelism: {
    title: "Pipeline Parallelism",
    explain: "Pipeline Parallelism splits model layers across GPUs sequentially. Each GPU processes a contiguous group of layers and passes activations to the next GPU. Unlike Tensor Parallelism, PP requires minimal communication (only activation tensors, not all-reduce). However, it introduces pipeline bubbles \u2014 idle time where GPUs wait for preceding stages. PP is preferred when NVLink is unavailable or for cross-node deployment.",
    refs: "2-GPU PP, 70B Q4: ~90% efficiency (small bubble). 4-GPU PP: ~75% efficiency. Cross-node PP: ~70% efficiency. Use PP over PCIe, TP over NVLink.",
  },
};

