// ============================================================================
// MAIN CALC
// ============================================================================
function calc() {
  var m = getModel();
  if (!m) return;
  var p = parseFloat(m.p || 0);
  var act = parseFloat(m.active || p);
  var L = parseInt(m.L || 0);
  var h = parseInt(m.h || 0);
  var nh = parseInt(m.nh || 0);
  var nkv = parseInt(m.nkv || nh || 0);
  var hd = parseInt(m.headDim || 0) || (nh > 0 ? Math.round(h / nh) : 64);
  var nG = parseInt(document.getElementById("nG").value);
  var qW = parseFloat(document.getElementById("qW").value);
  var qKV = parseFloat(document.getElementById("qKV").value);
  var ctx = parseInt(document.getElementById("ctx").value);
  var bs = parseInt(document.getElementById("bs").value);
  var uS = parseInt(document.getElementById("uS").value || 1);
  var ftSel = document.getElementById("ftSel");
  var isFT = ftSel ? ftSel.style.display !== "none" : false;
  var tMv = isFT ? document.getElementById("tM").value : "inf";
  var gName = document.getElementById("gSel").value;
  var isCustomGpu = gName === "Custom GPU";
  var gpu;
  if (isCustomGpu) {
    gpu = {
      name: "Custom GPU",
      vram: parseInt(document.getElementById("cVram").value) || 24,
      bw: parseInt(document.getElementById("cBw").value) || 1000,
      tdp: parseInt(document.getElementById("cTdp").value) || 300,
      tflops: parseInt(document.getElementById("cTflops").value) || 165,
      pcieGen: parseInt(document.getElementById("cPcieGen").value) || 4,
      pcieLanes:
        parseInt(document.getElementById("cPcieLanes").value) || 16,
      nvlink: parseInt(document.getElementById("cNvlink").value) || 0,
      nvswitch: document.getElementById("cNvswitch").value === "1",
    };
  } else {
    gpu =
      GPUS.find(function (g) {
        return g.name === gName;
      }) || GPUS[0];
  }
  var totV = gpu.vram * nG;
  document.getElementById("tV").textContent = totV + " GB";

  // ──── CONNECTIVITY PARAMETERS (v3: regime-aware offload model) ────
  //
  // Data path hierarchy (from fastest to slowest):
  //   1. GPU HBM  →  900-4800 GB/s  (weights resident in VRAM)
  //   2. NVLink   →  400-900 GB/s   (GPU↔GPU for Tensor Parallelism)
  //   3. PCIe     →  7-57 GB/s eff. (CPU↔GPU for RAM offload)
  //   4. System RAM →  43-304 GB/s eff. (offloaded layer weights / KV cache)
  //
  // Offloading Regime Model (N1-N10):
  //   Regime A: All in VRAM → full HBM speed
  //   Regime B: Weights in VRAM, KV Cache offloaded → full decode, TTFT + KV swap
  //   Regime C: Weights offloaded to RAM → Bus Wall on every token
  //   Regime D: Both offloaded → Bus Wall + KV swap

  var pcieGen = gpu.pcieGen || 4;
  var pcieLanes = gpu.pcieLanes || 16;
  var pcieTheoretical = (PCIE_BW_PER_LANE[pcieGen] || 1.969) * pcieLanes; // GB/s
  var pcieEffective = MathEngine.calcPcieEffective(pcieGen, pcieLanes, PCIE_EFFICIENCY); // GB/s practical

  // RAM bandwidth: theoretical peak and practical effective
  var ramTypeKey = document.getElementById("ramType")
    ? document.getElementById("ramType").value
    : "ddr5_2";
  var ramTheoretical = RAM_BW[ramTypeKey] || 89.6; // GB/s
  var numaAware = document.getElementById("numaAware")
    ? document.getElementById("numaAware").checked
    : false;
  var numaFactor = numaAware ? 1.0 : 0.65;
  var ramEffective = MathEngine.calcRamEffective(ramTheoretical, RAM_EFFICIENCY, numaFactor); // GB/s practical

  // Transfer bandwidth for RAM offload: bottleneck is the slower of PCIe or RAM
  var transferBW = Math.min(pcieEffective, ramEffective); // GB/s

  // Bus Wall Ratio: how many times slower RAM-offloaded layers are vs HBM
  var busWallRatio = gpu.bw / transferBW;

  // Identify the specific bottleneck in the transfer chain
  var transferBottleneck = pcieEffective <= ramEffective ? "PCIe" : "RAM";

  // Multi-GPU interconnect
  var interconnType = document.getElementById("interconnType")
    ? document.getElementById("interconnType").value
    : "nvswitch";
  var interconnBW = 0;
  var effectiveInterconnType = interconnType;

  if (nG > 1) {
    if (interconnType === "nvswitch") {
      if (gpu.nvswitch && gpu.nvlink > 0) {
        interconnBW = gpu.nvlink;
      } else if (gpu.nvlink > 0) {
        effectiveInterconnType = "nvlink";
        interconnBW = gpu.nvlink;
      } else {
        effectiveInterconnType = "pcie";
        interconnBW = pcieEffective;
      }
    } else if (interconnType === "nvlink") {
      if (gpu.nvlink > 0) {
        interconnBW = gpu.nvlink;
      } else {
        effectiveInterconnType = "pcie";
        interconnBW = pcieEffective;
      }
    } else if (interconnType === "pcie") {
      interconnBW = pcieEffective;
    } else if (interconnType === "pipeline") {
      interconnBW = pcieEffective;
    } else {
      interconnBW = 0;
    }
  }

  // Show/hide interconnect row
  var interconnRow = document.getElementById("interconnRow");
  if (interconnRow) {
    interconnRow.style.display = nG > 1 ? "block" : "none";
  }
  if (nG > 1) {
    var el = document.getElementById("dInterBW");
    if (el)
      el.textContent = interconnBW > 0 ? interconnBW.toFixed(0) : "0";
  }

  // Update RAM connectivity displays
  var pcieBwEl = document.getElementById("dPcieBW");
  var ramBwEl = document.getElementById("dRamBW");
  var busWallEl = document.getElementById("dBusWall");
  var bottleneckEl = document.getElementById("dBottleneck");
  var pcieLanesEl = document.getElementById("dPcieLanes");
  var hbmBwEl = document.getElementById("dHbmBW");
  if (pcieBwEl) pcieBwEl.textContent = pcieEffective.toFixed(1);
  if (ramBwEl) ramBwEl.textContent = ramEffective.toFixed(1);
  if (busWallEl)
    busWallEl.textContent =
      busWallRatio >= 10
        ? busWallRatio.toFixed(0)
        : busWallRatio.toFixed(1);
  if (bottleneckEl) bottleneckEl.textContent = transferBottleneck;
  if (pcieLanesEl) pcieLanesEl.textContent = "x" + pcieLanes;
  if (hbmBwEl) hbmBwEl.textContent = gpu.bw.toFixed(0);

  // ──── MEMORY CALCULATIONS ────────────────────────────────────────
  var wGB = MathEngine.calcWeights(p, qW);
  var kvGB = MathEngine.calcKVCache(L, nkv, hd, ctx, bs, uS, qKV, p);
  var ovF = isFT ? (tMv === "lora" ? 0.4 : 2.0) : 0.12;
  var ovGB = MathEngine.calcOverhead(wGB, ovF);
  var tot = wGB + kvGB + ovGB;

  // RAM offloading
  var ramOffEnabled = document.getElementById("ramOff").checked;
  var sysRam = ramOffEnabled
    ? parseFloat(document.getElementById("sysRam").value) || 0
    : 0;

  // ============================================================================
  // N1: VRAM PRIORITY ALLOCATION MODEL
  // ============================================================================
  // Weights get VRAM priority (read every token), KV gets remaining VRAM
  var allocation = MathEngine.calcVRAMAllocation(wGB, kvGB, ovGB, totV, sysRam);

  // N2: Classify the offloading regime
  var regime = MathEngine.classifyRegime(allocation);
  var regimeDesc = MathEngine.getRegimeDescription(regime);

  // Derived allocation values
  var vramPortion = allocation.weightsVRAM + allocation.kvVRAM + allocation.overheadVRAM;
  var ramUsable = allocation.weightsRAM + allocation.kvRAM;
  var stillOom = (wGB + kvGB + ovGB) > (totV + sysRam);

  // RAM fractions for performance model
  var weightRamFraction = wGB > 0 ? allocation.weightsRAM / wGB : 0;
  var kvRamFraction = kvGB > 0 ? allocation.kvRAM / kvGB : 0;

  // Update RAM met display
  var ramMet = document.getElementById("ramMet");
  var legRamWts = document.getElementById("legRamWts");
  var legRamKv = document.getElementById("legRamKv");
  if (ramUsable > 0) {
    if (ramMet) ramMet.style.display = "";
    var rRamEl = document.getElementById("rRam");
    if (rRamEl) rRamEl.textContent = fmt(ramUsable);
    if (legRamWts) legRamWts.style.display = allocation.weightsRAM > 0 ? "" : "none";
    if (legRamKv) legRamKv.style.display = allocation.kvRAM > 0 ? "" : "none";
  } else {
    if (ramMet) ramMet.style.display = "none";
    if (legRamWts) legRamWts.style.display = "none";
    if (legRamKv) legRamKv.style.display = "none";
  }

  // Display memory metrics
  document.getElementById("rW").textContent = fmt(wGB);
  document.getElementById("rKV").textContent = fmt(kvGB);
  document.getElementById("rOv").textContent = fmt(ovGB);
  document.getElementById("rKVL").textContent = isFT
    ? "Gradient mem"
    : "KV cache";
  document.getElementById("rOvL").textContent = isFT
    ? "Optimizer"
    : "Overhead";

  // Stacked bar
  var pctW = Math.min(100, (wGB / totV) * 100);
  var pctKV = Math.min(100, (kvGB / totV) * 100);
  var pctOv = Math.min(100, (ovGB / totV) * 100);
  var totalPct = pctW + pctKV + pctOv;

  var segWts = document.getElementById("segWts");
  var segKv = document.getElementById("segKv");
  var segOv = document.getElementById("segOv");
  var barInner = document.getElementById("barInner");
  if (segWts) segWts.style.width = pctW + "%";
  if (segKv) segKv.style.width = pctKV + "%";
  if (segOv) segOv.style.width = pctOv + "%";
  if (barInner) barInner.style.width = Math.min(100, totalPct) + "%";

  var barWrap = document.getElementById("barWrap");
  var barWrapRam = document.getElementById("barWrapRam");
  if (ramUsable > 0 && sysRam > 0) {
    if (barWrapRam) barWrapRam.style.display = "block";
    if (barWrap) barWrap.style.flex = totV;
    if (barWrapRam) barWrapRam.style.flex = sysRam;
    
    var pctRam = Math.min(100, (ramUsable / sysRam) * 100);
    var barInnerRam = document.getElementById("barInnerRam");
    if (barInnerRam) barInnerRam.style.width = pctRam + "%";
    
    var segRamWts = document.getElementById("segRamWts");
    var segRamKv = document.getElementById("segRamKv");
    if (segRamWts) segRamWts.style.width = (allocation.weightsRAM / ramUsable * 100) + "%";
    if (segRamKv) segRamKv.style.width = (allocation.kvRAM / ramUsable * 100) + "%";
  } else {
    if (barWrapRam) barWrapRam.style.display = "none";
    if (barWrap) barWrap.style.flex = "1";
  }

  // Labels
  if (ramUsable > 0) {
    var ramBreakdown = "";
    if (allocation.weightsRAM > 0) ramBreakdown += fmt(allocation.weightsRAM) + " weights";
    if (allocation.weightsRAM > 0 && allocation.kvRAM > 0) ramBreakdown += " + ";
    if (allocation.kvRAM > 0) ramBreakdown += fmt(allocation.kvRAM) + " KV cache";
    document.getElementById("usedL").textContent =
      fmt(tot) +
      " required (" +
      fmt(vramPortion) +
      " VRAM + " +
      ramBreakdown + " RAM)";
  } else {
    document.getElementById("usedL").textContent = fmt(tot) + " required";
  }
  document.getElementById("totL").textContent = fmt(totV) + " available";

  // Badge — show regime when offloading is active
  var bdg = document.getElementById("bdg");
  if (stillOom && !ramOffEnabled) {
    bdg.textContent = "OOM";
    bdg.className = "bdg er";
  } else if (stillOom && ramOffEnabled) {
    bdg.textContent = "OOM (even with RAM)";
    bdg.className = "bdg er";
  } else if (regime === "B") {
    bdg.textContent = "REGIME B — KV OFFLOAD";
    bdg.className = "bdg amber-bdg";
  } else if (regime === "C") {
    bdg.textContent = "REGIME C — WEIGHT OFFLOAD";
    bdg.className = "bdg amber-bdg";
  } else if (regime === "D") {
    bdg.textContent = "REGIME D — FULL OFFLOAD";
    bdg.className = "bdg amber-bdg";
  } else if (ramUsable > 0) {
    bdg.textContent = "RAM OFFLOAD";
    bdg.className = "bdg amber-bdg";
  } else if (tot <= totV * 0.75) {
    bdg.textContent = "FITS";
    bdg.className = "bdg ok";
  } else {
    bdg.textContent = "TIGHT";
    bdg.className = "bdg wa";
  }

  // Breakdown rows — show allocation breakdown
  var rows = [
    ["Model weights", wGB],
    [isFT ? "Gradient memory" : "KV cache", kvGB],
    [isFT ? "Optimizer state" : "Overhead", ovGB],
  ];
  if (allocation.weightsRAM > 0) {
    rows.push(["  Weights in RAM", allocation.weightsRAM]);
  }
  if (allocation.kvRAM > 0) {
    rows.push(["  KV Cache in RAM", allocation.kvRAM]);
  }
  rows.push(["Total", tot]);
  document.getElementById("bdDiv").innerHTML = rows
    .map(function (r) {
      var isRAM = r[0].indexOf("in RAM") >= 0;
      var isIndent = r[0].indexOf("  ") === 0;
      var cls =
        isRAM ? ' style="color:var(--amber)"' :
        isIndent ? ' style="padding-left:1em;color:var(--amber)"' : "";
      return (
        '<div class="bdr"' +
        cls +
        "><span>" +
        r[0] +
        "</span><span>" +
        fmt(r[1]) +
        "</span></div>"
      );
    })
    .join("");

  // Formula box with KaTeX
  var fmlEl = document.getElementById("fml");
  var fmlHtml = '<div id="fmlContent"></div>';
  fmlEl.innerHTML = fmlHtml;

  // Build formulas using templates
  var formulas = [];
  
  formulas.push({
    label: "Weights",
    tex: RESULT_TEMPLATES.weights
      .replace("{p}", p.toFixed(2))
      .replace("{qW}", qW)
      .replace("{wGB}", fmt(wGB))
  });
  
  if (L > 0 && nkv > 0 && hd > 0) {
    formulas.push({
      label: "KV Cache",
      tex: RESULT_TEMPLATES.kv_cache_full
        .replace("{L}", L)
        .replace("{nkv}", nkv)
        .replace("{hd}", hd)
        .replace("{ctx}", ctx)
        .replace("{bs}", bs)
        .replace("{uS}", uS)
        .replace("{qKV}", qKV)
        .replace("{kvGB}", fmt(kvGB))
    });
  } else {
    formulas.push({
      label: "KV Cache",
      tex: RESULT_TEMPLATES.kv_cache_approx
        .replace("{kvGB}", fmt(kvGB))
    });
  }
  
  formulas.push({
    label: "Overhead",
    tex: RESULT_TEMPLATES.overhead
      .replace("{ovF}", ovF)
      .replace("{ovGB}", fmt(ovGB))
  });
  
  formulas.push({
    label: "Total",
    tex: RESULT_TEMPLATES.total
      .replace("{wGB}", fmt(wGB))
      .replace("{kvGB}", fmt(kvGB))
      .replace("{ovGB}", fmt(ovGB))
      .replace("{ramNote}", ramUsable > 0 ? " = " + fmt(tot) : "")
      .replace("{totV}", fmt(totV))
  });
  
  if (ramUsable > 0 || nG > 1) {
    formulas.push({
      label: "PCIe (effective)",
      tex: RESULT_TEMPLATES.pcie_eff
        .replace("{pcieTheoretical}", pcieTheoretical.toFixed(1))
        .replace("{pcieLanes}", pcieLanes)
        .replace("{PCIE_EFFICIENCY}", PCIE_EFFICIENCY)
        .replace("{pcieEffective}", pcieEffective.toFixed(1))
    });
    formulas.push({
      label: "RAM (effective)",
      tex: RESULT_TEMPLATES.ram_eff
        .replace("{ramTheoretical}", ramTheoretical.toFixed(1))
        .replace("{RAM_EFFICIENCY}", RAM_EFFICIENCY)
        .replace("{numaFactor}", numaFactor.toFixed(2))
        .replace("{ramEffective}", ramEffective.toFixed(1))
    });
    formulas.push({
      label: "Bus Wall",
      tex: RESULT_TEMPLATES.bus_wall
        .replace(/{gpuBw}/g, gpu.bw)
        .replace("{pcieEffective}", pcieEffective.toFixed(1))
        .replace("{ramEffective}", ramEffective.toFixed(1))
        .replace("{transferBW}", transferBW.toFixed(1))
        .replace("{busWallRatio}", busWallRatio.toFixed(0))
    });
  }

  // N1: Show VRAM allocation formula when offloading
  if (regime !== "A") {
    formulas.push({
      label: "Regime " + regime,
      tex: RESULT_TEMPLATES.regime_class
        .replace("{regime}", regime)
        .replace("{wVRAM}", fmt(allocation.weightsVRAM))
        .replace("{wRAM}", fmt(allocation.weightsRAM))
        .replace("{kvVRAM}", fmt(allocation.kvVRAM))
        .replace("{kvRAM}", fmt(allocation.kvRAM))
    });

    // N4: Show KV swap latency for regimes B and D
    if (regime === "B" || regime === "D") {
      var kvSwapTimeS = MathEngine.calcKVSwapLatency(allocation.kvRAM, transferBW);
      formulas.push({
        label: "KV Swap",
        tex: RESULT_TEMPLATES.kv_swap
          .replace("{kvRAM}", fmt(allocation.kvRAM))
          .replace("{transferBW}", transferBW.toFixed(1))
          .replace("{kvSwapMs}", (kvSwapTimeS * 1000).toFixed(1))
      });
    }
  }

  var fmlContent = document.getElementById("fmlContent");
  if (fmlContent && typeof katex !== "undefined") {
    fmlContent.innerHTML = formulas
      .map(function (f) {
        return (
          '<div style="margin-bottom:4px"><strong>' +
          f.label +
          ':</strong> <span class="katex-inline" data-tex="' +
          escHtml(f.tex) +
          '"></span></div>'
        );
      })
      .join("");
    var inlines = fmlContent.querySelectorAll(".katex-inline");
    for (var i = 0; i < inlines.length; i++) {
      try {
        katex.render(inlines[i].getAttribute("data-tex"), inlines[i], {
          displayMode: false,
          throwOnError: false,
        });
      } catch (e) {
        inlines[i].textContent = inlines[i].getAttribute("data-tex");
      }
    }
  } else if (fmlContent) {
    fmlContent.textContent = formulas
      .map(function (f) {
        return f.label + ": " + f.tex;
      })
      .join("\n");
  }

  // ============================================================================
  // PERFORMANCE ESTIMATES (v3: Regime-Aware Offload Model)
  // ============================================================================
  //
  // The key distinction from the previous model:
  //   - Weight offloading → Bus Wall penalty on EVERY decode token
  //   - KV Cache offloading → Only impacts TTFT (swap-in latency)
  //
  // Regime A: TPS = BW_HBM / (P_active × b) [full speed]
  // Regime B: TPS = same as A [weights still in VRAM!]
  // ============================================================================TTFT_B = TTFT_A + T_kv_swap
  // Regime C: TPS = 1 / (W_VRAM/BW_HBM + W_RAM/BW_transfer) [Bus Wall]
  // ============================================================================TTFT_C = max(compute, weight_load)
  // Regime D: TPS = same as C [Bus Wall from weight offload]
  // ============================================================================TTFT_D = TTFT_C + T_kv_swap
  if (!isFT) {
    var bw = gpu.bw || 900;
    var tflops = gpu.tflops || 0;
    var tdp = gpu.tdp || 300;

    // ──── 1. DECODE SPEED — regime-aware ────────────────────────────
    var wBytes = act * qW; // Total weight bytes in GB
    var decodeTps = MathEngine.calcDecodeSpeed(bw, wBytes); // baseline single-GPU

    // ──── 2. MULTI-GPU PARALLELISM ──────────────────────────────────
    var tpEfficiency = 1.0;
    var tpNote = "";
    var isPipelineParallel = false;

    if (nG > 1 && interconnBW > 0) {
      if (
        effectiveInterconnType === "pipeline" ||
        interconnType === "pipeline"
      ) {
        // PIPELINE PARALLELISM
        isPipelineParallel = true;
        var bubbleFraction = (nG - 1) / nG;
        var effectiveH = h || hd * nh || 4096;
        var actSize = effectiveH * qW;
        var commTime = (actSize / (interconnBW * 1e9)) * 1e6;
        var commLatency = INTERCONN_LATENCY["pcie"] || 40e-6;
        var tCommPP = (nG - 1) * (commTime + commLatency * 1e6);
        var tStage = wBytes / nG / bw;
        var tBubble = tStage * bubbleFraction;
        tpEfficiency = tStage / (tStage + tBubble);
        decodeTps = (1 / tStage) * tpEfficiency;
        tpNote =
          nG +
          "\u00D7 GPU Pipeline Parallelism. Bubble overhead: " +
          (bubbleFraction * 100).toFixed(0) +
          "%. Efficiency: " +
          (tpEfficiency * 100).toFixed(0) +
          "%. Best for PCIe-only setups or cross-node.";
      } else {
        // TENSOR PARALLELISM
        var tCompute = (wBytes * 1e9) / (nG * bw * 1e9);
        var effectiveH = h || hd * nh || 4096;
        var msgSize = effectiveH * 2;

        var allReduceBandwidth;
        if (effectiveInterconnType === "nvswitch") {
          allReduceBandwidth = (2 * msgSize) / (interconnBW * 1e9);
        } else {
          allReduceBandwidth =
            (2 * ((nG - 1) / nG) * msgSize) / (interconnBW * 1e9);
        }

        var allReduceLatency =
          INTERCONN_LATENCY[effectiveInterconnType] || 10e-6;

        var tComm = L * (allReduceBandwidth + allReduceLatency);

        var arLatPerLayer = (allReduceBandwidth + allReduceLatency) * 1e6;

        tpEfficiency = tCompute / (tCompute + tComm);

        var tpEffEl = document.getElementById("dTPEff");
        if (tpEffEl)
          tpEffEl.textContent = (tpEfficiency * 100).toFixed(0) + "%";

        var arLatEl = document.getElementById("dARLat");
        if (arLatEl) arLatEl.textContent = arLatPerLayer.toFixed(1);

        decodeTps = ((bw * nG) / wBytes) * tpEfficiency;

        if (
          effectiveInterconnType === "none" ||
          interconnType === "none"
        ) {
          tpNote =
            "No direct GPU interconnect \u2014 Tensor Parallelism not viable. Use Pipeline Parallelism or separate instances instead.";
          tpEfficiency = 0;
          decodeTps = bw / wBytes;
        } else if (effectiveInterconnType === "pcie") {
          tpNote =
            nG +
            "\u00D7 GPU via PCIe P2P (" +
            interconnBW.toFixed(0) +
            " GB/s effective). TP efficiency: " +
            (tpEfficiency * 100).toFixed(0) +
            "%. NVLink would significantly improve scaling. Consider Pipeline Parallelism instead.";
        } else if (effectiveInterconnType === "nvswitch") {
          tpNote =
            nG +
            "\u00D7 GPU via NVSwitch (" +
            interconnBW +
            " GB/s, all-to-all). TP efficiency: " +
            (tpEfficiency * 100).toFixed(0) +
            "%. NVSwitch provides optimal scaling for " +
            nG +
            " GPUs.";
        } else {
          tpNote =
            nG +
            "\u00D7 GPU via NVLink P2P ring (" +
            interconnBW +
            " GB/s). TP efficiency: " +
            (tpEfficiency * 100).toFixed(0) +
            "%. NVSwitch would improve " +
            nG +
            "-GPU scaling.";
        }
      }
    } else if (nG > 1 && interconnBW === 0) {
      tpNote =
        nG +
        "\u00D7 GPU but no interconnect bandwidth. Cannot use parallelism effectively.";
      var tpEffEl = document.getElementById("dTPEff");
      if (tpEffEl) tpEffEl.textContent = "N/A";
      var arLatEl = document.getElementById("dARLat");
      if (arLatEl) arLatEl.textContent = "N/A";
    }

    // ──── 3. PREFILL SPEED (compute-bound) ──────────────────────────
    var prefillTps;
    if (tflops > 0) {
      prefillTps = MathEngine.calcPrefillSpeed(tflops, act, (nG > 1 && !isPipelineParallel) ? nG * tpEfficiency : tpEfficiency);
    } else {
      prefillTps = decodeTps * 4;
    }

    // ──── 4. TTFT baseline ──────────────────────────────────────────
    var ttftBase = (wBytes / bw) * 1000; // ms, single-GPU bandwidth-bound
    if (nG > 1 && interconnBW > 0 && !isPipelineParallel) {
      ttftBase = ((wBytes / (bw * nG)) * 1000) / tpEfficiency;
    } else if (nG > 1 && isPipelineParallel) {
      ttftBase = ((wBytes / bw) * 1000) / tpEfficiency;
    }

    // ============================================================================
    // 5. APPLY REGIME-AWARE PERFORMANCE MODEL (N3-N7)
    // ============================================================================
    var ramPerfNote = document.getElementById("ramPerfNote");
    var ttft = ttftBase;
    var kvSwapTimeS = 0;
    var kvSwapTimeMs = 0;

    // N4: Compute KV swap latency (for regimes B and D)
    if (regime === "B" || regime === "D") {
      kvSwapTimeS = MathEngine.calcKVSwapLatency(allocation.kvRAM, transferBW);
      kvSwapTimeMs = kvSwapTimeS * 1000;
    }

    if (regime === "A") {
      // ──── REGIME A: All in VRAM ────────────────────────────────────
      // Full performance, no penalties
      if (ramPerfNote) {
        ramPerfNote.style.display = "none";
        ramPerfNote.innerHTML = "";
      }
    } else if (regime === "B") {
      // ──── REGIME B: Weights in VRAM, KV offloaded ──────────────────
      // KEY INSIGHT: Decode speed is UNCHANGED because weights are in VRAM!
      // Only TTFT is impacted by KV swap-in latency
      // N3: TPS_B = TPS_A (full decode speed preserved)

      // TTFT gets the swap penalty
      ttft = ttftBase + kvSwapTimeMs;

      if (ramPerfNote) {
        ramPerfNote.style.display = "block";
        ramPerfNote.innerHTML =
          "<strong>Regime B — KV Cache Offload:</strong> " +
          "All weights in VRAM → decode at <strong>full HBM speed</strong> (" +
          decodeTps.toFixed(1) +
          " tok/s). KV Cache partially in RAM: " +
          fmt(allocation.kvVRAM) +
          " GB VRAM + " +
          fmt(allocation.kvRAM) +
          " GB RAM. " +
          "KV swap-in adds <strong>" +
          kvSwapTimeMs.toFixed(1) +
          " ms</strong> to TTFT (" +
          fmt(allocation.kvRAM) +
          " GB / " +
          transferBW.toFixed(1) +
          " GB/s). " +
          "Bottleneck: " +
          transferBottleneck +
          ". This is <strong>much better</strong> than weight offloading (Regime C/D).";
      }
    } else if (regime === "C") {
      // ──── REGIME C: Weights offloaded to RAM ───────────────────────
      // Bus Wall penalty on every decode token
      // N6: TTFT_C = max(compute_time, weight_load_time)

      var effectiveBwHBM = nG > 1 ? bw * nG * tpEfficiency : bw;
      var wVramGB = allocation.weightsVRAM;
      var wRamGB = allocation.weightsRAM;

      var tVramDecode = wVramGB / effectiveBwHBM;
      var tRamDecode = wRamGB / transferBW;
      var tDecodeTotal = tVramDecode + tRamDecode;

      decodeTps = 1 / tDecodeTotal;

      // Prefill also affected
      if (tflops > 0) {
        var prefillDegradation =
          1 / (1 - weightRamFraction + weightRamFraction * busWallRatio);
        prefillTps = prefillTps * (1 / prefillDegradation);
      }

      // TTFT: weight loading dominates
      var weightLoadTimeMs = (wRamGB / transferBW) * 1000;
      ttft = Math.max(ttftBase, weightLoadTimeMs);

      if (ramPerfNote) {
        ramPerfNote.style.display = "block";
        ramPerfNote.innerHTML =
          "<strong>Regime C — Weight Offload (Bus Wall):</strong> " +
          (weightRamFraction * 100).toFixed(0) +
          "% of weights in RAM, running " +
          busWallRatio.toFixed(0) +
          "\u00D7 slower than HBM on every token. " +
          "Bottleneck: " +
          transferBottleneck +
          " (" +
          transferBW.toFixed(1) +
          " GB/s effective vs " +
          bw +
          " GB/s HBM). " +
          "RAM layers: " +
          tRamDecode.toFixed(4) +
          "s/tok, VRAM layers: " +
          tVramDecode.toFixed(6) +
          "s/tok. " +
          "<strong>Recommendation:</strong> Quantize weights more aggressively to fit in VRAM — this avoids the Bus Wall entirely.";
      }
    } else if (regime === "D") {
      // ──── REGIME D: Both weights and KV in RAM ─────────────────────
      // Bus Wall + KV swap penalty (worst case)

      var effectiveBwHBM = nG > 1 ? bw * nG * tpEfficiency : bw;
      var wVramGB = allocation.weightsVRAM;
      var wRamGB = allocation.weightsRAM;

      var tVramDecode = wVramGB / effectiveBwHBM;
      var tRamDecode = wRamGB / transferBW;
      var tDecodeTotal = tVramDecode + tRamDecode;

      decodeTps = 1 / tDecodeTotal;

      // Prefill also affected
      if (tflops > 0) {
        var prefillDegradation =
          1 / (1 - weightRamFraction + weightRamFraction * busWallRatio);
        prefillTps = prefillTps * (1 / prefillDegradation);
      }

      // TTFT: weight loading + KV swap
      var weightLoadTimeMs = (wRamGB / transferBW) * 1000;
      ttft = Math.max(ttftBase, weightLoadTimeMs) + kvSwapTimeMs;

      if (ramPerfNote) {
        ramPerfNote.style.display = "block";
        ramPerfNote.innerHTML =
          "<strong>Regime D — Full Offload (Bus Wall + KV Swap):</strong> " +
          (weightRamFraction * 100).toFixed(0) +
          "% of weights in RAM (" +
          busWallRatio.toFixed(0) +
          "\u00D7 slower) + KV swap (" +
          kvSwapTimeMs.toFixed(1) +
          " ms). " +
          "This is the worst case. " +
          "<strong>Recommendation:</strong> Use more GPUs, quantize weights, or reduce context/users to get to Regime A or B.";
      }
    }

    // ──── 6. CONCURRENCY (N8-N9) ────────────────────────────────────
    // Calculate how many users can be served simultaneously
    var kvPerUser = uS > 0 ? kvGB / uS : kvGB;  // KV per single user
    var kvVRAMMax = allocation.kvVRAM;  // VRAM available for KV
    var ramForKV = allocation.kvRAM;     // RAM available for KV
    var concurrency = MathEngine.calcConcurrencyLimits(kvVRAMMax, kvPerUser, ramForKV);

    // ──── 7. DERIVED METRICS ────────────────────────────────────────
    var latPerTok = 1000 / decodeTps;
    var time100 = ttft / 1000 + 100 / decodeTps;
    var time1000 = ttft / 1000 + 1000 / decodeTps;

    // N9: Effective throughput accounts for KV swap overhead
    var throughput;
    if (regime === "B" || regime === "D") {
      throughput = MathEngine.calcEffectiveThroughput(
        decodeTps, concurrency.uActive, concurrency.uSwapped,
        kvSwapTimeS, 256  // assume ~256 avg tokens per context for swap calculation
      );
    } else {
      throughput = decodeTps * uS;
    }

    // HBM bandwidth actually used per decode step
    var bwUsed = wBytes * decodeTps;
    if (nG > 1 && !isPipelineParallel)
      bwUsed = bwUsed / (nG * tpEfficiency || 1);

    // Arithmetic intensity
    var arithIntensity = 2 / qW;

    // Bottleneck analysis with regime context
    var bottleneckLabel = "HBM bandwidth";
    if (regime === "C" || regime === "D") {
      bottleneckLabel =
        transferBottleneck +
        "+RAM (" +
        busWallRatio.toFixed(0) +
        "\u00D7 Bus Wall)";
    } else if (regime === "B") {
      bottleneckLabel = "HBM bandwidth (full speed)";
    } else if (nG > 1 && effectiveInterconnType === "pcie") {
      bottleneckLabel = "PCIe interconnect";
    }

    // ──── 8. DISPLAY RESULTS ────────────────────────────────────────
    document.getElementById("pPrefill").textContent =
      prefillTps >= 1000
        ? (prefillTps / 1000).toFixed(1) + "k"
        : prefillTps.toFixed(0);
    document.getElementById("pDecode").textContent =
      decodeTps >= 1000
        ? (decodeTps / 1000).toFixed(1) + "k"
        : decodeTps.toFixed(1);
    document.getElementById("pLat").textContent = latPerTok.toFixed(1);
    document.getElementById("pTTFT").textContent = ttft.toFixed(0);
    document.getElementById("p100").textContent = time100.toFixed(1);
    document.getElementById("p1000").textContent = time1000.toFixed(1);
    document.getElementById("pThr").textContent =
      throughput >= 1000
        ? (throughput / 1000).toFixed(1) + "k"
        : throughput.toFixed(0);
    document.getElementById("pBwUsed").textContent = bwUsed.toFixed(0);
    document.getElementById("pBottleneck").textContent = bottleneckLabel;
    document.getElementById("pArithInt").textContent =
      arithIntensity.toFixed(1);
    document.getElementById("perfCard").style.display = "block";

    // Regime badge in performance section
    var regimeEl = document.getElementById("pRegime");
    if (regimeEl) {
      regimeEl.textContent = "Regime " + regime;
      regimeEl.className = regime === "A" ? "bdg ok" :
                           regime === "B" ? "bdg amber-bdg" :
                           "bdg er";
      regimeEl.style.display = "inline-block";
    }

    // Concurrency display
    var concEl = document.getElementById("pConcurrency");
    if (concEl) {
      if (regime === "B" || regime === "D") {
        concEl.textContent = concurrency.uActive + " active + " + concurrency.uSwapped + " swapped";
        concEl.style.display = "";
      } else {
        concEl.textContent = concurrency.uActive + " active";
        concEl.style.display = "";
      }
    }

    // KV swap display
    var kvSwapEl = document.getElementById("pKVSwap");
    if (kvSwapEl) {
      if (regime === "B" || regime === "D") {
        kvSwapEl.textContent = kvSwapTimeMs.toFixed(1) + " ms";
        kvSwapEl.style.display = "";
      } else {
        kvSwapEl.style.display = "none";
      }
    }

    // TP note
    var tpPerfNote = document.getElementById("tpPerfNote");
    if (tpPerfNote) {
      if (nG > 1 && tpNote) {
        tpPerfNote.innerHTML = tpNote;
        tpPerfNote.style.display = "block";
      } else {
        tpPerfNote.style.display = "none";
      }
    }

    // ──── 9. POWER & COST CALCULATIONS ──────────────────────────────
    var utilPct =
      parseFloat(document.getElementById("pUtil").value) / 100;
    var elecCost = parseFloat(document.getElementById("pElec").value);
    var hrsPerDay = parseFloat(document.getElementById("pHrs").value);
    var carbonInt = parseFloat(document.getElementById("pCarbon").value);

    var powerDraw = MathEngine.calcPowerDraw(tdp, utilPct, nG);
    var energyPerHr = powerDraw / 1000; // kWh
    var costPerHr = energyPerHr * elecCost;
    var costPerDay = costPerHr * hrsPerDay;
    var costPerMonth = costPerDay * 30;
    var costPer1M =
      decodeTps > 0 ? (costPerHr / (decodeTps * 3600)) * 1e6 : 0;
    var co2PerHr = energyPerHr * carbonInt;
    var annualCo2 = (co2PerHr * hrsPerDay * 365) / 1000; // tonnes

    document.getElementById("cPower").textContent = powerDraw.toFixed(0);
    document.getElementById("cHr").textContent = fmtMoney(costPerHr);
    document.getElementById("cDay").textContent = fmtMoney(costPerDay);
    document.getElementById("cMonth").textContent =
      fmtMoney(costPerMonth);
    document.getElementById("c1M").textContent = fmtMoney(costPer1M);
    document.getElementById("cKwh").textContent = energyPerHr.toFixed(2);
    document.getElementById("cCo2Hr").textContent =
      co2PerHr >= 1 ? co2PerHr.toFixed(1) : co2PerHr.toFixed(3);
    document.getElementById("cCo2Yr").textContent =
      annualCo2 >= 1 ? annualCo2.toFixed(1) : annualCo2.toFixed(3);

    document.getElementById("powerCard").style.display = "block";
  } else {
    document.getElementById("perfCard").style.display = "none";
    document.getElementById("powerCard").style.display = "none";
  }
}
