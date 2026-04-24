# GPU Architecture Reference â€” A Deep Dive for LLM Inference

**Author:** Karim El Mernissi | **Date:** April 2026

This guide provides a concise, pedagogical overview of GPU architecture, specifically tailored to explain the performance characteristics of **Large Language Model (LLM) inference**.

---

## 1. The Core Bottleneck: Compute vs. Memory

To understand LLM inference, one must grasp the fundamental difference between **Compute-Bound** and **Memory-Bandwidth-Bound** workloads.

- **Compute-Bound:** The GPU processes data slower than it can fetch it from memory. Increasing TFLOPS improves performance.
- **Bandwidth-Bound:** The GPU fetches data slower than it can process it. The compute cores sit idle waiting for data. Increasing Memory Bandwidth (GB/s) improves performance.

**For LLM Inference:**

- **Prefill Phase (Prompt Processing):** Highly parallelized matrix multiplications. Typically **Compute-Bound**.
- **Decode Phase (Token Generation):** Autoregressive and sequential. The entire model must be read from memory to generate _one_ token. Highly **Bandwidth-Bound**.

---

## 2. Core GPU Components

### 2.1 CUDA Cores & Tensor Cores

- **CUDA Cores:** General-purpose scalar processing units for parallel computation.
- **Tensor Cores:** Specialized AI units designed exclusively to accelerate matrix operations. They perform mixed-precision Multiply-Accumulate (MAC) operations in a single clock cycle:

  $$ D = A \times B + C $$

  Where:
  - $A, B$: Input matrices containing activations and weights (often in lower precision, e.g., FP16, INT8).
  - $C$: Accumulator matrix (often in higher precision, e.g., FP32).
  - $D$: Output matrix (result of the MAC operation).

### 2.2 VRAM (Video RAM)

VRAM stores model weights, the KV Cache, and runtime activations. If the model size exceeds VRAM, it must be offloaded to system RAM, causing a severe performance cliff due to the slow PCIe bus interface.

### 2.3 Memory Bus & Bandwidth

Bandwidth defines how fast data flows from VRAM to the compute cores. It is the single most critical specification governing LLM token generation speed.

$$ \text{Bandwidth (GB/s)} = \frac{\text{Data Rate (MT/s)} \times \text{Bus Width (bits)}}{8 \times 1000} $$

Where:
- $\text{Bandwidth (GB/s)}$: Maximum theoretical memory throughput in Gigabytes per second.
- $\text{Data Rate (MT/s)}$: Memory transfer rate in Megatransfers per second.
- $\text{Bus Width (bits)}$: Width of the physical memory interface.
- $8$: Conversion factor from bits to bytes.
- $1000$: Conversion factor from Megabytes to Gigabytes.

---

## 3. GPU Memory Technologies

| Technology                      | Characteristics                                                                                 | Example GPU  | Max Bandwidth      |
| :------------------------------ | :---------------------------------------------------------------------------------------------- | :----------- | :----------------- |
| **GDDR6 / GDDR6X**              | Standard VRAM. Narrower bus (256-384 bit). High speed but limited capacity per GPU.             | RTX 4090     | ~1,000 GB/s        |
| **HBM (High Bandwidth Memory)** | Premium data center tech. 3D-stacked dies on a silicon interposer. Ultra-wide bus (5000+ bits). | H100, MI300X | 3,300 - 5,300 GB/s |

> **Insight:** HBM is what allows data center GPUs to generate tokens 3-5x faster than consumer GPUs, despite having similar or sometimes lower raw TFLOPS.

---

## 4. The Roofline Model for LLMs

The Roofline Model predicts performance limits based on **Arithmetic Intensity (AI)**:

$$ AI_{\text{ridge}} = \frac{\text{TFLOPS}}{\text{Memory Bandwidth (TB/s)}} $$

Where:
- $AI$: Arithmetic Intensity, representing the ratio of computation to memory access.
- $\text{FLOPs}$: Floating Point Operations performed.
- $\text{Bytes Accessed}$: Total memory data transferred to/from VRAM.
- $AI_{\text{ridge}}$: The ridge point where the hardware transitions from bandwidth-bound to compute-bound.
- $\text{TFLOPS}$: Peak hardware compute throughput in Tera Floating Point Operations Per Second.
- $\text{Memory Bandwidth (TB/s)}$: Peak hardware memory bandwidth in Terabytes per second.

### Why LLM Decode is Bandwidth-Bound:

During token generation (at batch size 1), every weight parameter is read exactly once from memory to perform one multiply-accumulate operation (which counts as 2 FLOPs).

$$ AI_{\text{decode}} = \frac{2 \times P_{\text{active}}}{P_{\text{active}} \times b_{\text{param}}} = \frac{2}{b_{\text{param}}} $$

Where:
- $AI_{\text{decode}}$: Arithmetic Intensity during the decode phase (batch size 1).
- $P_{\text{active}}$: Number of active model parameters used during one forward pass.
- $b_{\text{param}}$: Number of bytes required to store a single parameter in VRAM.
- $2$: The number of FLOPs per parameter (one multiply, one accumulate operation).

- For **FP16** ($b = 2$), $AI = 1 \text{ FLOP/byte}$.
- For **INT4** ($b = 0.5$), $AI = 4 \text{ FLOP/byte}$.

Because a typical modern GPU's ridge point is $>100 \text{ FLOP/byte}$, LLM decode operates _far below_ the compute ceiling. **The compute units spend the majority of their time starved for data.**

---

## 5. Impact of Quantization

Since decode is bottlenecked by memory bandwidth, reducing the physical byte-size of the model weights directly and proportionally increases token generation speed.

$$ \text{Theoretical Speedup} \approx \frac{b_{\text{FP16}}}{b_{\text{quantized}}} $$

Where:
- $\text{Theoretical Speedup}$: The expected multiplier in token generation speed.
- $b_{\text{FP16}}$: Bytes per parameter in FP16 precision (constant: 2 bytes).
- $b_{\text{quantized}}$: Bytes per parameter in the target quantized precision (e.g., 0.5 for INT4).

| Precision     | Bytes ($b$) | Relative Decode Speed  | Quality Impact |
| :------------ | :---------- | :--------------------- | :------------- |
| **FP16**      | 2.0         | $1.0\times$ (Baseline) | None           |
| **INT8**      | 1.0         | $\approx 2.0\times$    | Minimal        |
| **Q4 (INT4)** | 0.5         | $\approx 4.0\times$    | Moderate       |

---

## 6. Multi-GPU Interconnects

When a model is too large for a single GPU, **Tensor Parallelism** splits the matrix multiplications across multiple cards. This requires massive inter-GPU bandwidth to synchronize layers.

- **PCIe:** 15â€“57 GB/s. Too slow for Tensor Parallelism; causes severe communication bottlenecks.
- **NVLink (NVIDIA):** 600â€“900 GB/s. Enables seamless multi-GPU pooling and low-latency tensor parallelism.
- **Infinity Fabric (AMD):** High-speed interconnect functioning similarly for AMD's ecosystem.

---

## 7. Hardware Selection Priority

When configuring hardware for LLM inference, follow this strict hierarchy of constraints:

1.  **VRAM Capacity:** Does the model (weights + KV cache + overhead) mathematically fit in memory?
2.  **Memory Bandwidth:** Defines your maximum token generation speed (tokens/sec).
3.  **Interconnects:** NVLink is strictly required if splitting a model across multiple GPUs for latency-sensitive applications.
4.  **TFLOPS:** Primarily affects time-to-first-token (Prefill) and intensive training workloads, but rarely improves decoding.

