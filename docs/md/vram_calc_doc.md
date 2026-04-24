# LLM VRAM Calculator — Technical Documentation & Formula Reference

**Author:** Z.ai  
**Date:** April 2026

---

## Table of Contents

1. [Introduction](#introduction)  
   1.1 [Design Philosophy](#design-philosophy)  
   1.2 [Scope](#scope)  
2. [VRAM Estimation Model](#vram-estimation-model)  
   2.1 [Model Weight Memory](#model-weight-memory)  
   2.2 [KV Cache Memory](#kv-cache-memory)  
   2.3 [Framework Overhead](#framework-overhead)  
3. [RAM Overflow / Offloading](#ram-overflow--offloading)  
   3.1 [The Bus Wall ("Le Mur du Bus")](#the-bus-wall-le-mur-du-bus)  
   3.2 [Improved RAM Offload Performance Model](#improved-ram-offload-performance-model)  
4. [Connectivity & Bandwidth Architecture](#connectivity--bandwidth-architecture)  
   4.1 [The Data Path Hierarchy](#the-data-path-hierarchy)  
   4.2 [PCIe Bus Architecture](#pcie-bus-architecture)  
   4.3 [System RAM Bandwidth](#system-ram-bandwidth)  
   4.4 [The Transfer Chain: RAM → CPU → PCIe → GPU](#the-transfer-chain-ram--cpu--pcie--gpu)  
   4.5 [GPU Internal Architecture & HBM](#gpu-internal-architecture--hbm)  
   4.6 [Multi-GPU Interconnectivity](#multi-gpu-interconnectivity)  
   4.7 [Tensor Parallelism vs. Pipeline Parallelism](#tensor-parallelism-vs-pipeline-parallelism)  
   4.8 [Combined Performance Model](#combined-performance-model)  
5. [Performance Estimation](#performance-estimation)  
   5.1 [Decode Speed (Tokens per Second)](#decode-speed-tokens-per-second)  
   5.2 [Time to First Token (TTFT)](#time-to-first-token-ttft)  
   5.3 [Extended Performance Metrics](#extended-performance-metrics)  
6. [Power & Cost Estimation](#power--cost-estimation)  
   6.1 [Power Model](#power-model)  
   6.2 [Energy and Cost Calculations](#energy-and-cost-calculations)  
   6.3 [Carbon Emissions](#carbon-emissions)  
7. [Fine-Tuning Memory Model](#fine-tuning-memory-model)  
   7.1 [LoRA / QLoRA](#lora--qlora)  
   7.2 [Full Fine-Tuning](#full-fine-tuning)  
8. [Quantization Reference](#quantization-reference)  
   8.1 [GGUF Quantization Levels](#gguf-quantization-levels)  
   8.2 [Weighted Quantization Methods](#weighted-quantization-methods)  
9. [HuggingFace Integration](#huggingface-integration)  
   9.1 [Model Metadata Retrieval](#model-metadata-retrieval)  
   9.2 [Tensor Type Detection](#tensor-type-detection)  
   9.3 [Quantized Variant Discovery](#quantized-variant-discovery)  
10. [Hardware Reference](#hardware-reference)  
11. [Limitations & Assumptions](#limitations--assumptions)  
12. [Glossary](#glossary)  

---

## 1. Introduction

The LLM VRAM Calculator is a self-contained, browser-based tool designed to estimate the memory requirements, inference performance, and operational costs of deploying Large Language Models (LLMs) on GPU hardware. It integrates directly with the HuggingFace model hub to retrieve real model metadata, including parameter counts, tensor types, quantization configurations, and available quantized variants.

This document provides the complete mathematical foundation, algorithmic descriptions, and pedagogical references for every calculation performed by the tool. Each formula is derived from first principles and annotated with its assumptions, limitations, and practical implications.

### 1.1 Design Philosophy

The calculator follows three core principles: **rigor** (every estimate is grounded in a well-defined mathematical model), **transparency** (all formulas are displayed to the user with live substitution of their chosen parameters), and **pedagogy** (every input and output is accompanied by an on-demand explanation accessible via the "?" buttons throughout the interface).

### 1.2 Scope

The calculator addresses two primary deployment scenarios:

- **Inference**: Estimating VRAM requirements for model serving, including weight storage, KV cache, framework overhead, and performance metrics such as tokens-per-second and time-to-first-token.
- **Fine-tuning**: Estimating additional memory for gradient storage, optimizer states (AdamW), and activation memory for both LoRA and full fine-tuning regimes.

---

## 2. VRAM Estimation Model

Total GPU memory consumption is modeled as the sum of three primary components:

> **Equation 1**  
> `V_total = V_weights + V_kv + V_overhead`

where `V_weights` is the memory for model parameters, `V_kv` is the KV cache memory, and `V_overhead` accounts for framework and runtime overhead.

### 2.1 Model Weight Memory

The dominant memory component is the storage of model weights. Its size is determined by the total parameter count and the precision (bytes per parameter):

> **Equation 2**  
> `V_weights = P_total × b_param`

**Definition — Total Parameters:**  
`P_total` is the total number of learnable parameters in the model, including *all* experts for Mixture-of-Experts (MoE) architectures. For dense models, `P_total = P_active`. For MoE models, `P_total ≫ P_active` since only a subset of experts is activated per token.

**Definition — Bytes per Parameter:**  
`b_param` denotes the number of bytes used to store each weight parameter. This depends on the chosen precision or quantization level. The table below lists common values.

| Precision | Bits | B/param | Notes |
|-----------|------|---------|-------|
| FP32 | 32 | 4.00 | Full precision, rarely used for inference |
| BF16 / FP16 | 16 | 2.00 | Standard training & inference precision |
| FP8 / INT8 | 8 | 1.00 | 8-bit quantization |
| Q8_0 | 8 | 1.06 | GGUF 8-bit with overhead |
| Q6_K | 6 | 0.83 | 6-bit K-quant |
| Q5_K_M | 5 | 0.71 | 5-bit medium K-quant |
| Q4_K_M | 4 | 0.60 | 4-bit medium K-quant |
| Q4 / NF4 | 4 | 0.50 | 4-bit quantization, QLoRA default |
| Q3_K_M | 3 | 0.43 | 3-bit medium K-quant |
| Q2_K | 2 | 0.32 | 2-bit K-quant (aggressive) |

### 2.2 KV Cache Memory

During autoregressive generation, the model caches past Key and Value states from attention layers to avoid recomputation. The KV cache size is:

> **Equation 3**  
> `V_kv = 2 × L × n_kv × d_h × C × B × U × b_kv`

where the factor of 2 accounts for separate Key and Value tensors.

**KV Cache Parameters:**

- **L**: Number of transformer layers (blocks)
- **n_kv**: Number of key-value heads (depends on attention architecture)
- **d_h**: Head dimension, typically `d_h = h / n_heads` or explicitly defined in config
- **C**: Context length (maximum sequence length in tokens)
- **B**: Batch size (sequences processed simultaneously)
- **U**: Number of concurrent users (each requires separate KV cache)
- **b_kv**: Bytes per cached element (2 for FP16, 1 for FP8/INT8, 0.5 for Q4)

#### Attention Architecture Variants

The number of KV heads `n_kv` varies significantly across attention architectures, directly impacting KV cache memory:

| Architecture | KV Heads | Description |
|-------------|----------|-------------|
| Multi-Head (MHA) | `n_kv = n_q` | Each query head has its own KV pair. Largest KV cache. |
| Grouped-Query (GQA) | `1 < n_kv < n_q` | Query heads share KV heads in groups. Good balance. |
| Multi-Query (MQA) | `n_kv = 1` | All query heads share a single KV head. Smallest cache. |

For example, Llama 3.1 70B uses GQA with `n_q = 64` query heads but only `n_kv = 8` KV heads, reducing KV cache memory by 8× compared to MHA.

#### Fallback Estimation

When architecture details (`L`, `n_kv`, `d_h`) are unavailable, the calculator falls back to a rule-of-thumb estimate:

> **Equation 4**  
> `V_kv ≈ P_total × 0.12 × (C / 4096) × B × U`

This assumes KV cache is approximately 12% of weight memory at 4096-token context for a typical GQA model.

### 2.3 Framework Overhead

Framework overhead includes CUDA kernels, temporary buffers, activation memory, and communication buffers. It is estimated as a fraction of weight memory:

> **Equation 5**  
> `V_overhead = V_weights × f_overhead`

where `f_overhead` varies by operation mode:

| Mode | f_overhead |
|------|-----------|
| Inference | 0.12 (12%) |
| LoRA / QLoRA fine-tuning | 0.40 (40%) |
| Full fine-tuning | 2.00 (200%) |

For full fine-tuning, the overhead factor of 2.0 accounts for gradients (1× weights in training precision) and AdamW optimizer states (8 bytes per parameter in FP32), plus activation memory.

---

## 3. RAM Overflow / Offloading

> ⚠️ **Performance Warning**  
> RAM offloading allows models that exceed VRAM to run by spilling layers to system RAM, but at a severe performance cost: 10–50× slower inference for the offloaded portion.

When the total required memory exceeds available VRAM, the calculator supports an optional RAM offloading mode:

> **Equation 6**  
> `V_overflow = max(0, V_total − V_VRAM)`

> **Equation 7**  
> `V_RAM,usable = min(V_overflow, V_RAM,available)`

The effective available memory becomes:

> **Equation 8**  
> `V_effective = V_VRAM + V_RAM,usable`

### 3.1 The Bus Wall ("Le Mur du Bus")

The fundamental performance limitation of RAM offloading is captured by the **Bus Wall** concept — the ratio of GPU HBM bandwidth to the effective transfer bandwidth between CPU and GPU:

> **Equation 9**  
> `Bus Wall Ratio = BW_HBM / BW_transfer`

where `BW_transfer` is the effective bandwidth of the data path from system RAM to GPU, determined by the bottleneck in the transfer chain:

> **Equation 10**  
> `BW_transfer = min(BW_PCIe,effective, BW_RAM,effective)`

The Bus Wall ratio tells you how many times slower RAM-offloaded layers are compared to VRAM-resident layers. Typical values range from 30× (RTX 4090 with Gen4 x16) to over 100× (H100 SXM with Gen5 x16).

### 3.2 Improved RAM Offload Performance Model

The performance degradation from RAM offloading is modeled by decomposing the decode time into two independent data paths:

> **Equation 11**  
> `T_decode = W_VRAM / BW_HBM + W_RAM / BW_transfer`

where `W_VRAM` is the fraction of weights in VRAM and `W_RAM` is the fraction offloaded to RAM. This two-path model is more accurate than a simple degradation factor because it correctly accounts for the fact that VRAM-resident layers still run at full HBM speed, while only the offloaded portion is slowed by the Bus Wall.

The fraction of weights in RAM is:

> **Equation 12**  
> `f_RAM = V_RAM,usable / V_total`

Substituting, the effective decode speed becomes:

> **Equation 13**  
> `TPS_offload = 1 / ((1−f_RAM) × W / BW_HBM + f_RAM × W / BW_transfer)`

where `W = P_active × b_param` is the total weight memory in GB.

---

## 4. Connectivity & Bandwidth Architecture

This section provides the complete theoretical foundation for the data transfer paths that determine inference performance. Understanding these paths is essential for accurate performance estimation, especially when RAM offloading or multi-GPU configurations are involved.

### 4.1 The Data Path Hierarchy

Data involved in LLM inference traverses a strict hierarchy of interconnects, each with vastly different bandwidth characteristics:

| Path | Interconnect | BW Range | Use Case |
|------|-------------|----------|----------|
| GPU HBM | On-die bus | 900–4800 GB/s | VRAM-resident weight reading |
| NVLink/NVSwitch | GPU-GPU link | 400–900 GB/s | Tensor Parallelism all-reduce |
| PCIe | CPU-GPU bus | 7–57 GB/s eff. | RAM offload data transfer |
| System RAM | Memory bus | 43–304 GB/s eff. | Offloaded weight storage |

The key insight is that each level in this hierarchy is roughly 10–100× slower than the one above it. This creates a "bandwidth cliff" when inference must access slower paths.

### 4.2 PCIe Bus Architecture

PCI Express (PCIe) is the primary data highway between the CPU and GPU. Its bandwidth is determined by three factors: the generation (signaling rate), the number of lanes (bus width), and practical protocol efficiency.

#### PCIe Bandwidth Calculation

Theoretical PCIe bandwidth is calculated as:

> **Equation 14**  
> `BW_PCIe,theoretical = R_GT/s × N_lanes × (128/130) ÷ 8`

where `R_GT/s` is the transfer rate per lane, `N_lanes` is the number of lanes (typically 16, 8, or 4), and the 128/130 factor accounts for the encoding overhead introduced in PCIe 3.0+.

| Generation | GT/s/lane | x16 (GB/s) | x8 (GB/s) |
|-----------|-----------|------------|-----------|
| PCIe 3.0 | 8 | 15.75 | 7.88 |
| PCIe 4.0 | 16 | 31.50 | 15.75 |
| PCIe 5.0 | 32 | 63.00 | 31.50 |

#### Practical PCIe Efficiency

Theoretical bandwidth is never fully achieved. Protocol overhead reduces practical throughput:

> **Equation 15**  
> `BW_PCIe,effective = BW_PCIe,theoretical × η_PCIe`

where `η_PCIe ≈ 0.90` for large sequential DMA transfers. The overhead comes from:

- **TLP headers**: Each Transaction Layer Packet has a 12–16 byte header for a 256–4096 byte payload
- **DLLP and ACK/NAK**: Data Link Layer packets add protocol overhead
- **Root complex latency**: CPU-side DMA controller adds a few microseconds of setup latency
- **Memory mapping**: IOMMU address translation for DMA adds overhead

For small random transfers (e.g., individual parameter updates), efficiency can drop to 40–70%. The calculator uses `η_PCIe = 0.90` as a conservative estimate for the large sequential DMA transfers characteristic of weight loading during inference.

#### PCIe Lane Width Impact

Most desktop and server GPUs connect via x16 (16 lanes). However, some configurations reduce the effective lane width:

- Motherboards that share PCIe lanes between slots may run the GPU at x8 when both slots are populated
- Some budget GPUs are physically x8 or x4
- SXM form factors bypass PCIe entirely for GPU-GPU communication, but still use PCIe for CPU-GPU data transfer

Reducing from x16 to x8 exactly halves the available bandwidth, which significantly impacts RAM offload performance.

### 4.3 System RAM Bandwidth

#### Theoretical RAM Bandwidth

System RAM bandwidth is determined by the memory type, transfer rate, and number of channels:

> **Equation 16**  
> `BW_RAM,theoretical = MT/s × N_channels × 8 bytes/transfer`

| Configuration | MT/s | Channels | BW (GB/s) |
|--------------|------|----------|-----------|
| DDR4-3200 2-ch | 3200 | 2 | 51.2 |
| DDR4-3200 4-ch | 3200 | 4 | 102.4 |
| DDR4-3200 8-ch | 3200 | 8 | 204.8 |
| DDR5-5600 2-ch | 5600 | 2 | 89.6 |
| DDR5-5600 4-ch | 5600 | 4 | 179.2 |
| DDR5-5600 8-ch | 5600 | 8 | 358.4 |

#### Practical RAM Efficiency

As with PCIe, theoretical RAM bandwidth is not fully achievable:

> **Equation 17**  
> `BW_RAM,effective = BW_RAM,theoretical × η_RAM × f_NUMA`

where `η_RAM ≈ 0.85` accounts for DRAM refresh cycles, row misses, and memory controller scheduling, and `f_NUMA` is the NUMA efficiency factor.

#### NUMA Effects on Multi-Socket Systems

On multi-socket servers (AMD EPYC, Intel Xeon), each CPU socket has its own memory controller and attached RAM. Accessing RAM on the local socket is fast, but accessing RAM attached to a remote socket traverses an inter-socket link (AMD Infinity Fabric or Intel UPI) that adds latency and reduces bandwidth by 30–50%:

> **Equation 18**  
> `f_NUMA = 1.0` if NUMA-aware (weights on local socket)  
> `f_NUMA = 0.65` if no NUMA awareness (potential cross-socket access)

The calculator uses `f_NUMA = 0.65` when NUMA awareness is disabled, reflecting the worst case where weight data may be allocated on a remote socket. For production deployments, always enable NUMA-aware allocation and pin the GPU to the same NUMA node as the RAM holding the offloaded weights.

### 4.4 The Transfer Chain: RAM → CPU → PCIe → GPU

When RAM-offloaded layers are accessed during inference, data must traverse the entire transfer chain:

1. **RAM read**: Data is read from DRAM through the CPU memory controller
2. **CPU processing**: The CPU's IOMMU maps the DMA buffer address
3. **PCIe DMA transfer**: Data is transferred via DMA from the pinned CPU buffer to GPU VRAM
4. **GPU reception**: The GPU's PCIe controller receives the data into VRAM

The bottleneck in this chain is the slower of PCIe effective bandwidth and RAM effective bandwidth:

> **Equation 19**  
> `BW_transfer = min(BW_PCIe,effective, BW_RAM,effective)`

In most configurations, PCIe is the bottleneck. Even DDR4 2-channel at 51.2 GB/s theoretical (≈43 GB/s effective) exceeds PCIe Gen4 x8 at ≈14 GB/s effective. However, with fast PCIe Gen5 x16 (≈57 GB/s effective), slower RAM configurations (DDR4 2-channel) can become the bottleneck instead.

### 4.5 GPU Internal Architecture & HBM

#### High Bandwidth Memory (HBM)

GPU HBM is the fastest memory in the inference data path, connected to the GPU die via an ultra-wide bus (3072–6144 bits) on an organic or silicon interposer. This provides bandwidth of 900–4800 GB/s, orders of magnitude faster than any external interconnect.

| Memory Type | Example GPU | Bandwidth | Bus Width |
|------------|------------|-----------|-----------|
| GDDR6X | RTX 4090 | 1008 GB/s | 384-bit |
| HBM2e | A100 | 2000 GB/s | 5120-bit |
| HBM3 | H100 SXM | 3350 GB/s | 5120-bit |
| HBM3e | H200 SXM | 4800 GB/s | 6144-bit |

LLM decode is almost always HBM-bandwidth-bound: each token generation requires reading ALL active weights from HBM, but only performs `2/b` FLOPs per byte of weight data (where `b` is bytes per parameter). This arithmetic intensity is far below the GPU's ridge point, confirming the bandwidth-bound nature of decode.

#### GPU Clock Speed & Compute Throughput

GPU clock speed (typically 1.5–2.5 GHz) affects compute throughput (TFLOPS) but has limited impact on bandwidth-bound inference. The relationship between clock speed and compute throughput is:

> **Equation 20**  
> `TFLOPS = N_cores × f_clock × 2 FLOP/clock/core (tensor cores)`

For bandwidth-bound decode, increasing clock speed provides no benefit — the bottleneck is HBM read speed, not compute. Clock speed only matters for compute-bound prefill, where higher TFLOPS directly translates to faster prompt processing.

#### The Roofline Model

The roofline model provides a unified framework for understanding whether a workload is compute-bound or bandwidth-bound:

**Definition — Arithmetic Intensity:**  
Arithmetic intensity is the ratio of FLOPs performed to bytes of data accessed:

> **Equation 21**  
> `AI = FLOPs / Bytes accessed`

For LLM decode with active parameters `P_active` stored at `b` bytes per parameter:

> **Equation 22**  
> `AI_decode = (2 × P_active) / (P_active × b) = 2/b`

At Q4 (`b = 0.5`), the arithmetic intensity is only 4 FLOP/byte. At FP16 (`b = 2`), it drops to 1 FLOP/byte. These values are far below the GPU's ridge point (the arithmetic intensity at which compute and bandwidth are equally limiting):

> **Equation 23**  
> `AI_ridge = TFLOPS / BW_HBM`

For H100: `AI_ridge = 990/3350 ≈ 0.30 TFLOPS/(GB/s) = 295 FLOP/byte`. This confirms that decode is deeply bandwidth-bound regardless of quantization level.

For prefill, the arithmetic intensity is much higher because the same weights are reused across all prompt tokens in a single batched matrix multiplication. Prefill is typically compute-bound.

### 4.6 Multi-GPU Interconnectivity

#### NVLink

NVLink is NVIDIA's proprietary high-speed GPU-to-GPU interconnect. It provides dramatically higher bandwidth than PCIe, enabling efficient Tensor Parallelism (TP) where model weights are split across multiple GPUs.

| Generation | Links | BW per GPU | Example GPU |
|-----------|-------|------------|-------------|
| NVLink 2 | 6 | 300 GB/s | V100 |
| NVLink 3 | 12 | 600 GB/s | A100 |
| NVLink 4 | 18 | 900 GB/s | H100/H200 |

NVLink uses a point-to-point topology: each GPU has direct links to specific other GPUs. In a 4-GPU system, this creates a mesh where each GPU connects to every other GPU. In an 8-GPU system, each GPU typically connects to 4 neighbors, and multi-hop routing is required for non-adjacent communication.

#### NVSwitch

NVSwitch is NVIDIA's switching fabric that provides full all-to-all connectivity between all GPUs in a node. Unlike point-to-point NVLink, NVSwitch allows any GPU to communicate with any other GPU at full NVLink speed simultaneously.

> **Equation 24** — All-reduce with NVSwitch:  
> `All-reduce_NVSwitch = 2 × (h × 2) / BW_NVLink`

vs. ring all-reduce on point-to-point NVLink:

> **Equation 25**  
> `All-reduce_ring = 2 × ((N−1)/N) × (h × 2) / BW_NVLink`

where `h` is the hidden size and `N` is the number of GPUs. NVSwitch eliminates the `(N−1)/N` penalty and reduces the number of hops, providing significantly better TP efficiency for 4+ GPUs.

#### AMD Infinity Fabric

AMD's Infinity Fabric serves a similar role to NVLink on MI-series GPUs. The MI300X uses Infinity Fabric to connect its 8 chiplets (8 compute + 4 I/O) within a single package, providing up to 400 GB/s of inter-chiplet bandwidth.

#### PCIe Peer-to-Peer (P2P)

Without NVLink or Infinity Fabric, GPUs can communicate via PCIe peer-to-peer transfers. This uses the PCIe bus for direct GPU-to-GPU communication without CPU involvement, but the bandwidth is limited to PCIe speeds (typically 15–57 GB/s effective), making TP inefficient for all but the smallest models.

### 4.7 Tensor Parallelism vs. Pipeline Parallelism

#### Tensor Parallelism (TP)

TP splits model weights across `N` GPUs. Each GPU holds `1/N` of the weights and performs the corresponding portion of each matrix multiplication. After each transformer layer, an all-reduce synchronizes the partial results across all GPUs.

The decode time per token with TP is:

> **Equation 26**  
> `T_TP = (P_active × b) / (N × BW_HBM) + L × ((2 × (N−1)/N × h × 2) / BW_interconnect + λ_AR)`

where `λ_AR` is the all-reduce latency per layer (5–100 μs depending on interconnect).

TP efficiency is:

> **Equation 27**  
> `η_TP = T_compute / (T_compute + T_communication)`

| Interconnect | 2-GPU | 4-GPU | 8-GPU |
|-------------|-------|-------|-------|
| NVSwitch (H100) | 96% | 93% | 88% |
| NVLink P2P (H100) | 94% | 87% | 75% |
| PCIe Gen4 x16 | 65% | 40% | 20% |

TP is the preferred strategy when NVLink or NVSwitch is available, as it provides near-linear scaling with minimal latency overhead.

#### Pipeline Parallelism (PP)

PP splits model layers across GPUs sequentially. Each GPU processes a contiguous group of layers and passes the activation tensor to the next GPU. PP requires much less communication than TP — only the activation tensor (hidden size × batch × precision) is sent between stages, not an all-reduce.

However, PP introduces pipeline bubbles — idle time where GPUs wait for preceding stages to complete:

> **Equation 28**  
> `Bubble fraction = (N−1) / (N + M − 1)`

where `M` is the number of micro-batches. For single-user inference with batch size 1, only 1 micro-batch is possible, giving bubble fraction `(N−1)/N`.

PP is preferred when NVLink is unavailable (e.g., consumer GPUs connected via PCIe) or for cross-node deployment where network latency makes TP impractical.

#### Combined TP + PP

For very large models on many GPUs, both strategies can be combined: TP within a node (using NVLink) and PP across nodes (using network). For example, a 405B model on 8 H100s might use TP=4 within two nodes and PP=2 across nodes.

### 4.8 Combined Performance Model

The complete decode time model accounts for all connectivity factors:

> **Equation 29**  
> `T_decode = (1−f_RAM) × W / (η_TP × N × BW_HBM)  +  f_RAM × W / min(η_PCIe × BW_PCIe, η_RAM × f_NUMA × BW_RAM)`

This formula captures the essential physics of LLM inference:

- VRAM-resident layers benefit from both HBM bandwidth and TP parallelism
- RAM-offloaded layers are limited by the slowest link in the transfer chain
- The Bus Wall ratio quantifies the performance gap between these two regimes
- NUMA effects can further degrade RAM-offloaded performance on multi-socket systems
- PCIe lane width and generation directly affect the transfer bottleneck

---

## 5. Performance Estimation

### 5.1 Decode Speed (Tokens per Second)

During autoregressive decoding, each token generation requires loading all model weights from GPU memory. This makes inference **bandwidth-bound**:

> **Equation 30**  
> `TPS = BW / (P_active × b_param × 1000)`

where `BW` is the GPU memory bandwidth in GB/s. For MoE models, `P_active` (the number of parameters active per forward pass) is used instead of `P_total`, since only the routed experts are loaded during decode.

**Definition — Active Parameters:**  
For dense models, `P_active = P_total`. For MoE models, `P_active` represents only the parameters used per forward pass, including the embedding layer, shared experts, and the `k` routed experts per token. For example, Mixtral 8x7B has `P_total = 47B` but `P_active ≈ 13B`.

### 5.2 Time to First Token (TTFT)

The prefill phase processes the entire prompt in parallel. The time to first token is estimated as:

> **Equation 31**  
> `TTFT = (P_active × C × 2 × b_param) / BW × 1000 ms`

where `C` is the prompt length in tokens. The factor of 2 accounts for the read and write of activations during the forward pass.

### 5.3 Extended Performance Metrics

The calculator provides additional derived metrics for practical deployment planning:

> `Latency per token = 1000 / TPS` ms &nbsp;&nbsp; *(Equation 32)*

> `Time_100 = TTFT/1000 + 100/TPS` seconds &nbsp;&nbsp; *(Equation 33)*

> `Time_1000 = TTFT/1000 + 1000/TPS` seconds &nbsp;&nbsp; *(Equation 34)*

> `Throughput = TPS × U` tok/s total &nbsp;&nbsp; *(Equation 35)*

---

## 6. Power & Cost Estimation

### 6.1 Power Model

GPU power consumption is estimated from the Thermal Design Power (TDP) and utilization:

> **Equation 36**  
> `P_draw = TDP × (U_GPU / 100)`

where `U_GPU` is the GPU utilization percentage. LLM inference is typically memory-bandwidth-bound rather than compute-bound, resulting in 60–90% utilization during decode.

### 6.2 Energy and Cost Calculations

> `E_hour = P_draw / 1000` kWh &nbsp;&nbsp; *(Equation 37)*

> `C_hour = E_hour × R_elec` &nbsp;&nbsp; *(Equation 38)*

> `C_day = C_hour × H_day` &nbsp;&nbsp; *(Equation 39)*

> `C_month = C_day × 30` &nbsp;&nbsp; *(Equation 40)*

> `C_1M tok = (C_hour / (TPS × 3600)) × 10⁶` &nbsp;&nbsp; *(Equation 41)*

where `R_elec` is the electricity rate ($/kWh), `H_day` is operating hours per day, and TPS is the decode speed.

### 6.3 Carbon Emissions

> `CO₂,hour = E_hour × I_carbon` &nbsp;&nbsp; *(Equation 42)*

> `CO₂,annual = CO₂,hour × H_day × 365 / 1000` tonnes &nbsp;&nbsp; *(Equation 43)*

where `I_carbon` is the grid carbon intensity in kg CO₂/kWh. Default values and regional references:

| Region | kg CO₂/kWh |
|--------|-----------|
| World average | 0.417 |
| European Union | 0.255 |
| United States | 0.387 |
| France (nuclear) | 0.056 |
| Sweden (hydro) | 0.045 |
| Poland (coal) | 0.769 |
| China | 0.555 |

---

## 7. Fine-Tuning Memory Model

For fine-tuning, the memory model extends to include gradients and optimizer states:

> **Equation 44**  
> `V_FT = V_weights + V_gradients + V_optimizer + V_activations`

### 7.1 LoRA / QLoRA

With LoRA, only low-rank adaptation matrices are trained. The additional memory is:

> **Equation 45**  
> `V_LoRA ≈ 0.40 × V_weights`

This accounts for LoRA weights (typically <1% of base weights), their gradients (FP32), and 8-bit AdamW states.

### 7.2 Full Fine-Tuning

Full fine-tuning requires gradients for all parameters and optimizer states:

> `V_gradients = P_total × b_train` &nbsp;&nbsp; *(Equation 46)*

> `V_optimizer = P_total × 8` (AdamW FP32: momentum + variance) &nbsp;&nbsp; *(Equation 47)*

Combined with the overhead factor of 2.0, this yields approximately 3× the weight memory for FP16 training with AdamW.

---

## 8. Quantization Reference

### 8.1 GGUF Quantization Levels

GGUF (GPT-Generated Unified Format) is the standard format for llama.cpp and compatible engines. The following table lists supported quantization levels:

| Level | Label | B/param | Description |
|-------|-------|---------|-------------|
| Q2_K | 2-bit K-quant | 0.32 | Aggressive 2-bit quantization, K-quants method |
| Q3_K_S | 3-bit small | 0.34 | 3-bit quantization, small variant |
| Q3_K_M | 3-bit medium | 0.43 | 3-bit quantization, medium variant |
| Q3_K_L | 3-bit large | 0.45 | 3-bit quantization, large variant |
| Q4_0 | 4-bit base | 0.56 | Basic 4-bit quantization |
| Q4_K_S | 4-bit small K | 0.58 | 4-bit K-quant, small variant |
| Q4_K_M | 4-bit medium K | 0.60 | 4-bit K-quant, medium variant |
| Q5_0 | 5-bit base | 0.68 | Basic 5-bit quantization |
| Q5_K_S | 5-bit small K | 0.69 | 5-bit K-quant, small variant |
| Q5_K_M | 5-bit medium K | 0.71 | 5-bit K-quant, medium variant |
| Q6_K | 6-bit K-quant | 0.83 | 6-bit K-quant |
| Q8_0 | 8-bit quant | 1.06 | Near-FP16 quality, 8-bit quantization |

### 8.2 Weighted Quantization Methods

Beyond GGUF, the calculator recognizes several weighted quantization formats commonly found on HuggingFace:

| Method | Typical Bits | Characteristics |
|--------|-------------|-----------------|
| GPTQ | 2–8 bits | Post-training quantization with calibration dataset; group-wise quantization with optional desc_act; commonly 4-bit with group_size=128 |
| AWQ | 4–8 bits | Activation-aware weight quantization; preserves salient weights; group-wise with zero-point |
| EXL2 | 2–8 bits | ExLlamaV2 format; mixed-precision per-layer quantization; optimized for ExLlamaV2 inference engine |
| BNB/NF4 | 4 bits | BitsAndBytes NF4 quantization; default for QLoRA; double quantization support |

---

## 9. HuggingFace Integration

### 9.1 Model Metadata Retrieval

The calculator fetches model metadata from the HuggingFace API using the endpoint:

```
GET https://huggingface.co/api/models/{model_id}
```

This returns the model card data including parameter counts (from `safetensors.total`), tensor types (from `safetensors.parameters`), siblings (file list), and default branch information.

### 9.2 Tensor Type Detection

Tensor type detection follows a 4-level priority chain:

1. **quantization_config** in `config.json`: Highest priority for explicitly quantized models (GPTQ, AWQ, EXL2, BNB).
2. **safetensors.parameters** from the API: Shows actual dtype names of stored tensors (e.g., `F16`, `Q8_0`).
3. **Safetensors index metadata**: dtype fields in `model.safetensors.index.json`.
4. **torch_dtype** in `config.json`: Fallback for unquantized models (typically `bfloat16` or `float16`).

### 9.3 Quantized Variant Discovery

The calculator automatically discovers quantized variants by:

1. Scanning current model's siblings for GGUF files (parsing filenames like `model-Q4_K_M.gguf`).
2. Searching HuggingFace for related repos: `[base-name] GGUF`, `[base-name] GPTQ`, `[base-name] AWQ`.
3. Deduplicating and sorting variants by quantization level.

---

## 10. Hardware Reference

### Supported GPU Hardware

| GPU | VRAM (GB) | HBM BW (GB/s) | TFLOPS (FP16) | PCIe Gen | NVLink (GB/s) | NVS |
|-----|-----------|---------------|---------------|----------|---------------|-----|
| H200 SXM | 141 | 4800 | 990 | 5 | 900 | Yes |
| H100 SXM | 80 | 3350 | 990 | 5 | 900 | Yes |
| H100 PCIe | 80 | 2000 | 756 | 5 | 0 | No |
| A100 80GB | 80 | 2000 | 312 | 4 | 600 | Yes |
| A100 40GB | 40 | 1555 | 312 | 4 | 600 | Yes |
| A6000 Ada | 48 | 960 | 182 | 4 | 0 | No |
| RTX 4090 | 24 | 1008 | 165 | 4 | 0 | No |
| RTX 3090 | 24 | 936 | 71 | 4 | 0 | No |
| L40S | 48 | 864 | 366 | 4 | 0 | No |
| MI300X | 192 | 5300 | 1307 | 5 | 400 | No |
| MI250X | 128 | 3276 | 383 | 4 | 400 | No |

### PCIe Bandwidth Reference

| Config | Theoretical | Effective (η=0.90) | Bus Wall (H200) | Bus Wall (4090) |
|--------|------------|-------------------|-----------------|----------------|
| Gen3 x16 | 15.75 GB/s | 14.2 GB/s | 338× | 71× |
| Gen3 x8 | 7.88 GB/s | 7.1 GB/s | 676× | 142× |
| Gen4 x16 | 31.5 GB/s | 28.4 GB/s | 169× | 36× |
| Gen4 x8 | 15.75 GB/s | 14.2 GB/s | 338× | 71× |
| Gen5 x16 | 63.0 GB/s | 56.7 GB/s | 85× | 18× |
| Gen5 x8 | 31.5 GB/s | 28.4 GB/s | 169× | 36× |

### Interconnect Latency Reference

| Interconnect | Latency | Topology | Notes |
|-------------|---------|----------|-------|
| NVSwitch | ~5 μs | All-to-all | Best for 4+ GPUs |
| NVLink P2P | ~10 μs | Ring/mesh | Direct GPU-GPU link |
| PCIe P2P | ~40 μs | Ring | Through root complex |
| Through CPU | ~100 μs | Multi-hop | GPU → CPU → GPU |

---

## 11. Limitations & Assumptions

1. **Roofline model**: Performance estimates assume bandwidth-bound decode (confirmed by arithmetic intensity analysis). Actual performance may be compute-bound for very small models or very short sequences with high batch sizes.

2. **PCIe efficiency**: The 0.90 efficiency factor is a conservative estimate for large DMA transfers. Real efficiency varies by motherboard, IOMMU configuration, and CPU architecture. Measured values range from 0.85 to 0.95.

3. **RAM efficiency**: The 0.85 factor for sequential RAM reads is an average. Actual values depend on DRAM type, frequency, and access pattern. Mixed read/write workloads achieve lower efficiency.

4. **NUMA model**: The 0.65 factor for non-NUMA-aware allocation is an approximation. Actual cross-socket bandwidth depends on the inter-socket link (AMD Infinity Fabric, Intel UPI), memory topology, and system firmware configuration.

5. **KV cache quantization**: Quality impact of aggressive KV cache quantization (Q4) is not modeled; it may degrade output quality for sensitive tasks.

6. **RAM offloading**: The two-path decode model (`W_VRAM/BW_HBM + W_RAM/BW_transfer`) is a simplification. In practice, offloading engines (llama.cpp, vLLM) may use pipelined or overlapped transfers that partially hide latency.

7. **Multi-GPU TP**: Estimates use the analytical all-reduce model with fixed latency. Real implementations may use custom all-reduce algorithms (e.g., NVLink SHARP, NCCL topology-aware) that differ from the ring model.

8. **Pipeline Parallelism**: The bubble fraction model assumes single micro-batch inference. With continuous batching or multiple concurrent requests, the bubble can be hidden more effectively.

9. **Framework overhead**: The 12% inference overhead is a conservative average. vLLM and llama.cpp may use less; PyTorch native may use more.

10. **MoE active parameters**: The calculator uses the active parameter count reported by the model card or estimated from config. Actual activation patterns may differ, and expert routing is token-dependent.

11. **Power model**: TDP-based power estimation is approximate. Real power varies with clock speed, temperature throttling, and workload characteristics.

12. **GPU clock speed**: Clock speed is not explicitly modeled in the calculator since decode is bandwidth-bound. For compute-bound prefill, the TFLOPS value implicitly captures clock speed effects.

---

## 12. Glossary

- **VRAM** — Video Random Access Memory: the dedicated high-bandwidth memory on a GPU, used to store model weights, activations, and KV cache during inference.
- **HBM** — High Bandwidth Memory: a type of VRAM connected to the GPU die via an ultra-wide bus (3072–6144 bits) on an interposer, providing 900–4800 GB/s bandwidth.
- **KV Cache** — Key-Value Cache: stored attention key and value tensors from previous positions, enabling efficient autoregressive generation without recomputing the entire context at each step.
- **GQA** — Grouped-Query Attention: an attention mechanism where query heads are divided into groups, each sharing a single KV head. Balances MHA quality with MQA efficiency.
- **MQA** — Multi-Query Attention: all query heads share a single KV head, minimizing KV cache size at the cost of some expressiveness.
- **MoE** — Mixture of Experts: a model architecture where only a subset (typically 2–8) of expert modules is activated per token, reducing compute while maintaining model capacity.
- **TPS** — Tokens Per Second: the generation speed during autoregressive decoding, the primary throughput metric for LLM inference.
- **TTFT** — Time to First Token: the latency from receiving a prompt to generating the first output token, dominated by the prefill (prompt processing) phase.
- **TDP** — Thermal Design Power: the maximum sustained power dissipation a cooling system must handle, used as a proxy for peak GPU power consumption.
- **QLoRA** — Quantized Low-Rank Adaptation: a parameter-efficient fine-tuning method that quantizes the base model to 4-bit (NF4) and trains small low-rank adapter matrices.
- **GGUF** — GPT-Generated Unified Format: a file format for storing quantized LLM weights, designed for efficient loading and inference with llama.cpp and compatible engines.
- **Bus Wall** — Le Mur du Bus: the ratio of GPU HBM bandwidth to the effective transfer bandwidth (PCIe + RAM) for RAM-offloaded layers. Quantifies how many times slower offloaded layers are compared to VRAM-resident layers.
- **PCIe** — Peripheral Component Interconnect Express: the primary data bus between CPU and GPU, providing 7–57 GB/s effective bandwidth depending on generation and lane count.
- **NVLink** — NVIDIA's proprietary high-speed GPU-to-GPU interconnect, providing 300–900 GB/s bandwidth for Tensor Parallelism communication.
- **NVSwitch** — NVIDIA's switching fabric providing all-to-all connectivity between GPUs in a node, improving TP efficiency for 4+ GPUs compared to point-to-point NVLink.
- **NUMA** — Non-Uniform Memory Access: a memory architecture where each CPU socket has local memory; accessing remote socket memory is slower, affecting RAM offload performance on multi-socket servers.
- **Arithmetic Intensity** — FLOPs performed per byte of data accessed, determining whether a workload is compute-bound or bandwidth-bound according to the roofline model.
- **Ridge Point** — The arithmetic intensity at which a GPU transitions from bandwidth-bound to compute-bound execution, equal to `TFLOPS / BW_HBM`.
- **TP** — Tensor Parallelism: a multi-GPU strategy that splits model weights across GPUs, requiring all-reduce communication after each layer.
- **PP** — Pipeline Parallelism: a multi-GPU strategy that splits model layers across GPUs sequentially, requiring only activation tensor communication between stages.
