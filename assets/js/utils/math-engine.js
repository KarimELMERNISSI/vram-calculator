// ═══════════════════════════════════════════════════════════
// MATH ENGINE
// ═══════════════════════════════════════════════════════════

/**
 * Core mathematical functions for VRAM and Performance estimation.
 * Decoupled from the DOM to allow for reuse and testing.
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
  }
};
