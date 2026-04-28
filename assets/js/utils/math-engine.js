// ═══════════════════════════════════════════════════════════
// MATH ENGINE
// ═══════════════════════════════════════════════════════════

/**
 * Core mathematical functions for VRAM and Performance estimation.
 * Decoupled from the DOM to allow for reuse and testing.
 *
 * Includes the Offloading Regime Model (A/B/C/D):
 *   Regime A: All in VRAM → full HBM speed
 *   Regime B: Weights in VRAM, KV Cache offloaded to RAM → full decode speed, TTFT + KV swap
 *   Regime C: Weights offloaded to RAM → Bus Wall on every token
 *   Regime D: Both offloaded → Bus Wall + KV swap (worst case)
 */
var MathEngine = {
  /**
   * Weights Memory: W = P_total * b
   */
  calcWeights: function (p, qW) {
    return p * qW;
  },

  /**
   * KV Cache Memory: Accurate model (if parameters provided) or approximation.
   */
  calcKVCache: function (L, nkv, hd, ctx, bs, uS, qKV, p) {
    if (L > 0 && nkv > 0 && hd > 0) {
      return (2 * L * nkv * hd * ctx * bs * uS * qKV) / 1e9;
    }
    // Fallback approximation
    return p * 0.12 * (ctx / 4096) * bs * uS;
  },

  /**
   * Overhead: CUDA kernels, activation buffers, etc.
   */
  calcOverhead: function (wGB, factor) {
    return wGB * factor;
  },

  /**
   * PCIe Effective Bandwidth
   */
  calcPcieEffective: function (pcieGen, pcieLanes, pcieEfficiency) {
    var pcieBwPerLane = { 1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.939, 6: 7.877 };
    var theoretical = (pcieBwPerLane[pcieGen] || 1.969) * pcieLanes;
    return theoretical * pcieEfficiency;
  },

  /**
   * System RAM Effective Bandwidth
   */
  calcRamEffective: function (ramTheoretical, efficiency, numaFactor) {
    return ramTheoretical * efficiency * numaFactor;
  },

  /**
   * Decode Speed (tokens/s) - Bandwidth Bound
   */
  calcDecodeSpeed: function (bw, wBytes) {
    if (wBytes === 0) return 0;
    return bw / wBytes;
  },

  /**
   * Prefill Speed (tokens/s) - Compute Bound
   */
  calcPrefillSpeed: function (tflops, act, tpEfficiency) {
    if (act === 0) return 0;
    var speed = (tflops * 1e12) / (2 * act * 1e9);
    return speed * tpEfficiency;
  },

  /**
   * Power Draw (Watts)
   */
  calcPowerDraw: function (tdp, utilPct, nG) {
    return tdp * utilPct * nG;
  },

  // ═══════════════════════════════════════════════════════════
  // OFFLOADING REGIME MODEL
  // ═══════════════════════════════════════════════════════════

  /**
   * N1: VRAM Priority Allocation
   *
   * Weights are allocated to VRAM first (highest priority since they are
   * read on every decode token). KV cache gets whatever VRAM remains.
   *
   * Returns an allocation object:
   *   { weightsVRAM, weightsRAM, kvVRAM, kvRAM, overheadVRAM, overheadRAM }
   */
  calcVRAMAllocation: function (weightsGB, kvGB, overheadGB, vramGB, ramAvailableGB) {
    var allocation = {
      weightsVRAM: 0,
      weightsRAM: 0,
      kvVRAM: 0,
      kvRAM: 0,
      overheadVRAM: 0,
      overheadRAM: 0
    };

    // Overhead is always allocated first (CUDA kernels must be in VRAM)
    allocation.overheadVRAM = Math.min(overheadGB, vramGB);
    var vramAfterOverhead = Math.max(0, vramGB - allocation.overheadVRAM);

    // Priority 1: Weights → VRAM
    allocation.weightsVRAM = Math.min(weightsGB, vramAfterOverhead);
    allocation.weightsRAM = Math.max(0, weightsGB - allocation.weightsVRAM);
    var vramAfterWeights = Math.max(0, vramAfterOverhead - allocation.weightsVRAM);

    // Priority 2: KV Cache → remaining VRAM
    allocation.kvVRAM = Math.min(kvGB, vramAfterWeights);
    allocation.kvRAM = Math.max(0, kvGB - allocation.kvVRAM);

    // Check if RAM can absorb overflow
    var totalRAMNeeded = allocation.weightsRAM + allocation.kvRAM + allocation.overheadRAM;
    var ramUsable = Math.min(totalRAMNeeded, ramAvailableGB || 0);

    // If RAM is insufficient, proportionally reduce what goes to RAM
    if (totalRAMNeeded > 0 && ramUsable < totalRAMNeeded) {
      var ramScale = ramUsable / totalRAMNeeded;
      allocation.weightsRAM *= ramScale;
      allocation.kvRAM *= ramScale;
      allocation.overheadRAM *= ramScale;
    }

    return allocation;
  },

  /**
   * N2: Regime Classification
   *
   * Classifies the deployment into one of four performance regimes:
   *   A: All in VRAM → full performance
   *   B: Weights in VRAM, KV partially in RAM → decode at full speed, TTFT + swap
   *   C: Weights partially in RAM → Bus Wall on every token
   *   D: Both weights and KV in RAM → Bus Wall + KV swap
   */
  classifyRegime: function (allocation) {
    var wRAM = allocation.weightsRAM;
    var kvRAM = allocation.kvRAM;

    if (wRAM <= 0 && kvRAM <= 0) {
      return "A";  // All in VRAM
    } else if (wRAM <= 0 && kvRAM > 0) {
      return "B";  // KV offloaded, weights in VRAM
    } else if (wRAM > 0 && kvRAM <= 0) {
      return "C";  // Weights offloaded, KV in VRAM (rare)
    } else {
      return "D";  // Both offloaded
    }
  },

  /**
   * Get regime description
   */
  getRegimeDescription: function (regime) {
    var descriptions = {
      "A": "All in VRAM — full HBM performance",
      "B": "Weights in VRAM, KV Cache offloaded — full decode speed, TTFT includes KV swap",
      "C": "Weights offloaded to RAM — Bus Wall penalty on every token",
      "D": "Both weights & KV offloaded — Bus Wall + KV swap (worst case)"
    };
    return descriptions[regime] || "Unknown regime";
  },

  /**
   * N3 & N4: KV Cache Swap-In Latency
   *
   * When KV cache is stored in RAM, it must be swapped into VRAM before
   * generation can begin for a given user context. This is a one-time
   * cost per context switch.
   *
   * T_kv_swap = V_kv_RAM / BW_transfer  (seconds)
   */
  calcKVSwapLatency: function (kvRAM_GB, transferBW_GBps) {
    if (kvRAM_GB <= 0 || transferBW_GBps <= 0) return 0;
    return kvRAM_GB / transferBW_GBps; // seconds
  },

  /**
   * N5-N7: Regime-Aware TTFT
   *
   * TTFT varies by regime:
   *   A: Standard prefill time
   *   B: TTFT_A + T_kv_swap (one-time KV swap for offloaded cache)
   *   C: max(compute_time, weight_load_time) — weight loading dominates
   *   D: TTFT_C + T_kv_swap
   */
  calcRegimeTTFT: function (regime, ttftBase_ms, kvSwapTime_ms, weightLoadTime_ms) {
    switch (regime) {
      case "A":
        return ttftBase_ms;
      case "B":
        return ttftBase_ms + kvSwapTime_ms;
      case "C":
        return Math.max(ttftBase_ms, weightLoadTime_ms);
      case "D":
        return Math.max(ttftBase_ms, weightLoadTime_ms) + kvSwapTime_ms;
      default:
        return ttftBase_ms;
    }
  },

  /**
   * N8: Concurrency Limits
   *
   * U_active = floor(V_kv_VRAM_max / V_kv_per_user)
   *   — max concurrent users whose KV cache fits entirely in VRAM
   *
   * U_swapped = floor(V_RAM_for_kv / V_kv_per_user)
   *   — additional users whose KV cache can be stored in RAM (swapped)
   *
   * U_total = U_active + U_swapped
   */
  calcConcurrencyLimits: function (kvVRAMMax_GB, kvPerUser_GB, ramForKV_GB) {
    var uActive = kvPerUser_GB > 0 ? Math.floor(kvVRAMMax_GB / kvPerUser_GB) : 0;
    var uSwapped = kvPerUser_GB > 0 ? Math.floor(ramForKV_GB / kvPerUser_GB) : 0;
    return {
      uActive: Math.max(0, uActive),
      uSwapped: Math.max(0, uSwapped),
      uTotal: Math.max(0, uActive) + Math.max(0, uSwapped)
    };
  },

  /**
   * N9: Effective Throughput with KV Swapping
   *
   * When some users have their KV cache in RAM, context switches incur
   * a swap penalty. The effective throughput models this:
   *
   * Throughput = U_active × TPS + U_swapped × TPS × swap_efficiency
   *
   * where swap_efficiency = 1 / (1 + T_kv_swap / T_generation_per_context)
   */
  calcEffectiveThroughput: function (tps, uActive, uSwapped, kvSwapTime_s, avgContextTokens) {
    var throughput_active = uActive * tps;

    if (uSwapped <= 0 || kvSwapTime_s <= 0) {
      return throughput_active;
    }

    // Time to generate one full context
    var tGenPerContext = avgContextTokens / tps; // seconds
    // Swap efficiency: fraction of time spent generating vs swapping
    var swapEfficiency = 1 / (1 + kvSwapTime_s / tGenPerContext);
    var throughput_swapped = uSwapped * tps * swapEfficiency;

    return throughput_active + throughput_swapped;
  },

  /**
   * N10: Quantization vs Offload Decision Threshold
   *
   * Compares two strategies:
   *   1. Quantize weights more aggressively to fit in VRAM
   *   2. Keep higher precision but offload to RAM
   *
   * Returns the quantization level at which weights fit in VRAM,
   * and whether that's better than offloading.
   *
   * quantBytesNeeded = V_VRAM_available / P_total  (bytes/param to fit in VRAM)
   * offloadPenalty = f_RAM × Bus_Wall_ratio (slowdown factor)
   */
  calcQuantVsOffload: function (vramAvailableGB, totalParams_B, ramFraction, busWallRatio) {
    var bytesNeeded = vramAvailableGB / totalParams_B; // bytes/param to fit in VRAM
    var offloadSlowdown = ramFraction * busWallRatio;  // relative slowdown from offloading

    return {
      bytesNeededForVRAM: bytesNeeded,
      offloadSlowdown: offloadSlowdown,
      quantizationBetter: offloadSlowdown > 2,  // if offloading is >2x slower, quantize instead
      recommendation: offloadSlowdown > 2
        ? "Quantize to " + (bytesNeeded * 8).toFixed(1) + "-bit to fit in VRAM — faster than offloading"
        : "Offloading may be acceptable if quantization quality loss is unacceptable"
    };
  }
};
