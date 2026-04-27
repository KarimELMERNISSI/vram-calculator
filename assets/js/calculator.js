// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// MAIN CALC
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
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
  var isFT = document.getElementById("ftSel").style.display !== "none";
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

  // â”€â”€â”€ CONNECTIVITY PARAMETERS (v2: comprehensive model) â”€â”€
  //
  // Data path hierarchy (from fastest to slowest):
  //   1. GPU HBM  â†’  900-4800 GB/s  (weights resident in VRAM)
  //   2. NVLink   â†’  400-900 GB/s   (GPUâ†”GPU for Tensor Parallelism)
  //   3. PCIe     â†’  7-57 GB/s eff. (CPUâ†”GPU for RAM offload)
  //   4. System RAM â†’  43-304 GB/s eff. (offloaded layer weights)
  //
  // For RAM offload, the effective transfer BW is:
  //   BW_transfer = min(PCIe_effective, RAM_effective)
  // where PCIe_effective = theoretical Ã— (lanes/16) Ã— Î·_PCIe
  // and   RAM_effective  = theoretical Ã— Î·_RAM Ã— NUMA_factor

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
  var numaFactor = numaAware ? 1.0 : 0.65; // Without NUMA awareness, cross-socket access degrades BW
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
  var effectiveInterconnType = interconnType; // may be overridden

  // Auto-detect best available interconnect for non-custom GPUs
  if (nG > 1) {
    if (interconnType === "nvswitch") {
      if (gpu.nvswitch && gpu.nvlink > 0) {
        interconnBW = gpu.nvlink; // NVSwitch provides full NVLink BW all-to-all
      } else if (gpu.nvlink > 0) {
        // GPU has NVLink but no NVSwitch â€” fall back to P2P ring
        effectiveInterconnType = "nvlink";
        interconnBW = gpu.nvlink;
      } else {
        // No NVLink at all â€” fall back to PCIe P2P
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
      interconnBW = pcieEffective; // PCIe P2P bandwidth
    } else if (interconnType === "pipeline") {
      // Pipeline Parallelism: communication is only activation tensors between stages
      // Much less data than all-reduce: just hidden_size Ã— batch Ã— precision
      interconnBW = pcieEffective; // Use PCIe for activation passing
    } else {
      interconnBW = 0; // No direct interconnect
    }
  }

  // Show/hide interconnect row
  var interconnRow = document.getElementById("interconnRow");
  if (interconnRow) {
    interconnRow.style.display = nG > 1 ? "block" : "none";
  }
  // Update interconnect displays
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

  // â”€â”€â”€ MEMORY CALCULATIONS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
  var vramPortion = Math.min(tot, totV);
  var ramOverflow = Math.max(0, tot - totV);
  var ramUsable = ramOffEnabled ? Math.min(ramOverflow, sysRam) : 0;
  var stillOom = ramOverflow > ramUsable;

  // Fraction of total memory in RAM (for offload performance)
  var ramFraction = ramUsable > 0 ? ramUsable / tot : 0;

  // Update RAM met display
  var ramMet = document.getElementById("ramMet");
  var legRam = document.getElementById("legRam");
  if (ramUsable > 0) {
    ramMet.style.display = "";
    legRam.style.display = "";
    document.getElementById("rRam").textContent = fmt(ramUsable);
  } else {
    ramMet.style.display = "none";
    legRam.style.display = "none";
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

  document.getElementById("segWts").style.width = pctW + "%";
  document.getElementById("segKv").style.width = pctKV + "%";
  document.getElementById("segOv").style.width = pctOv + "%";
  document.getElementById("barInner").style.width =
    Math.min(100, totalPct) + "%";

  var barWrap = document.getElementById("barWrap");
  var barWrapRam = document.getElementById("barWrapRam");
  if (ramUsable > 0 && sysRam > 0) {
    if (barWrapRam) barWrapRam.style.display = "block";
    if (barWrap) barWrap.style.flex = totV;
    if (barWrapRam) barWrapRam.style.flex = sysRam;
    
    var pctRam = Math.min(100, (ramUsable / sysRam) * 100);
    var segRam = document.getElementById("segRam");
    if (segRam) segRam.style.width = "100%";
    var barInnerRam = document.getElementById("barInnerRam");
    if (barInnerRam) barInnerRam.style.width = pctRam + "%";
  } else {
    if (barWrapRam) barWrapRam.style.display = "none";
    if (barWrap) barWrap.style.flex = "1";
  }

  // Labels
  if (ramUsable > 0) {
    document.getElementById("usedL").textContent =
      fmt(tot) +
      " required (" +
      fmt(vramPortion) +
      " VRAM + " +
      fmt(ramUsable) +
      " RAM)";
  } else {
    document.getElementById("usedL").textContent = fmt(tot) + " required";
  }
  document.getElementById("totL").textContent = fmt(totV) + " available";

  // Badge
  var bdg = document.getElementById("bdg");
  if (stillOom && !ramOffEnabled) {
    bdg.textContent = "OOM";
    bdg.className = "bdg er";
  } else if (stillOom && ramOffEnabled) {
    bdg.textContent = "OOM (even with RAM)";
    bdg.className = "bdg er";
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

  // Breakdown rows
  var rows = [
    ["Model weights", wGB],
    [isFT ? "Gradient memory" : "KV cache", kvGB],
    [isFT ? "Optimizer state" : "Overhead", ovGB],
  ];
  if (ramUsable > 0) {
    rows.push(["RAM offloaded", ramUsable]);
  }
  rows.push(["Total", tot]);
  document.getElementById("bdDiv").innerHTML = rows
    .map(function (r) {
      var cls =
        r[0] === "RAM offloaded" ? ' style="color:var(--amber)"' : "";
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
  var fmtVal = function(val, fixed) { return fixed !== undefined ? val.toFixed(fixed) : val; };
  
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

  // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
  // PERFORMANCE ESTIMATES (v2: Connectivity-Aware Roofline Model)
  // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
  //
  // The Roofline Model determines whether inference is compute-bound
  // or bandwidth-bound by comparing arithmetic intensity to the GPU's
  // ridge point.
  //
  // Arithmetic Intensity = FLOPs / Bytes_accessed = 2/b for decode
  //   Q4:  4 FLOP/byte  (deeply bandwidth-bound)
  //   Q8:  2 FLOP/byte  (deeply bandwidth-bound)
  //   FP16: 1 FLOP/byte (deeply bandwidth-bound)
  //
  // Ridge point (H100): ~120 FLOP/byte â†’ decode is always bandwidth-bound
  //
  // Decode model: T_decode = W_vram/BW_HBM + W_ram/BW_transfer
  //   where W_vram = weights in VRAM, W_ram = weights in RAM
  //
  // Multi-GPU model: TP splits weights, PP splits layers
  //   TP: T = 1/N * W/BW_HBM + L * all_reduce / BW_interconnect
  //   PP: T = sum(stage_times) + bubble_overhead
  if (!isFT) {
    var bw = gpu.bw || 900;
    var tflops = gpu.tflops || 0;
    var tdp = gpu.tdp || 300;

    // â”€â”€â”€ 1. DECODE SPEED (bandwidth-bound) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // Each decode step reads all active weights once from HBM
    // tok/s = BW_HBM / (P_active * bytes_per_param)
    var wBytes = act * qW; // Total weight bytes in GB
    var decodeTps = MathEngine.calcDecodeSpeed(bw, wBytes); // tokens/sec (single GPU, no offload)

    // â”€â”€â”€ 2. MULTI-GPU PARALLELISM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    var tpEfficiency = 1.0;
    var tpNote = "";
    var isPipelineParallel = false;

    if (nG > 1 && interconnBW > 0) {
      if (
        effectiveInterconnType === "pipeline" ||
        interconnType === "pipeline"
      ) {
        // â”€â”€â”€ PIPELINE PARALLELISM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        isPipelineParallel = true;
        // PP: layers split across GPUs, each GPU processes its stage sequentially
        // Pipeline bubble: (N-1) micro-batches of idle time per forward pass
        // With 1 micro-batch: bubble fraction = (N-1)/N
        // With more micro-batches, bubble decreases: (N-1)/(N+M-1) for M micro-batches
        // For single-user inference with batch=1: only 1 micro-batch possible
        var bubbleFraction = (nG - 1) / nG;
        // Communication: just activation tensors (hidden_size Ã— precision)
        var effectiveH = h || hd * nh || 4096;
        var actSize = effectiveH * qW; // activation tensor size in bytes
        var commTime = (actSize / (interconnBW * 1e9)) * 1e6; // us per stage boundary
        var commLatency = INTERCONN_LATENCY["pcie"] || 40e-6;
        var tCommPP = (nG - 1) * (commTime + commLatency * 1e6); // us total communication
        // Each stage still reads its fraction of weights from HBM
        // Stage time = (P_active * qW / N) / BW_HBM
        var tStage = wBytes / nG / bw; // seconds per stage
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
        // â”€â”€â”€ TENSOR PARALLELISM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        // With TP, weights are split across N GPUs
        // Each GPU reads 1/N of weights: T_compute = wBytes / (N * BW_HBM)
        var tCompute = (wBytes * 1e9) / (nG * bw * 1e9); // seconds per token (compute)

        // All-reduce communication per layer
        var effectiveH = h || hd * nh || 4096;
        var msgSize = effectiveH * 2; // bytes per all-reduce message (FP16)

        // All-reduce depends on topology:
        // Ring all-reduce (NVLink P2P / PCIe P2P):
        //   Bandwidth: 2 * (N-1)/N * msg_size / BW_interconnect
        // NVSwitch all-reduce:
        //   Bandwidth: 2 * msg_size / BW_interconnect (all-to-all at full speed)
        var allReduceBandwidth;
        if (effectiveInterconnType === "nvswitch") {
          // NVSwitch: reduce-scatter + all-gather, each in 1 step
          allReduceBandwidth = (2 * msgSize) / (interconnBW * 1e9); // seconds
        } else {
          // Ring all-reduce: 2 * (N-1)/N factor
          allReduceBandwidth =
            (2 * ((nG - 1) / nG) * msgSize) / (interconnBW * 1e9); // seconds
        }

        // Latency per all-reduce operation
        var allReduceLatency =
          INTERCONN_LATENCY[effectiveInterconnType] || 10e-6;

        var tComm = L * (allReduceBandwidth + allReduceLatency);

        // All-reduce latency per layer (for display)
        var arLatPerLayer = (allReduceBandwidth + allReduceLatency) * 1e6; // microseconds

        tpEfficiency = tCompute / (tCompute + tComm);

        // Update TP efficiency display
        var tpEffEl = document.getElementById("dTPEff");
        if (tpEffEl)
          tpEffEl.textContent = (tpEfficiency * 100).toFixed(0) + "%";

        // Update all-reduce latency display
        var arLatEl = document.getElementById("dARLat");
        if (arLatEl) arLatEl.textContent = arLatPerLayer.toFixed(1);

        // Apply TP speedup: N GPUs Ã— efficiency
        decodeTps = ((bw * nG) / wBytes) * tpEfficiency;

        if (
          effectiveInterconnType === "none" ||
          interconnType === "none"
        ) {
          tpNote =
            "No direct GPU interconnect \u2014 Tensor Parallelism not viable. Use Pipeline Parallelism or separate instances instead.";
          tpEfficiency = 0;
          decodeTps = bw / wBytes; // Fall back to single-GPU speed
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

    // â”€â”€â”€ 3. PREFILL SPEED (compute-bound) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // Each token requires 2 * P_active FLOPs (multiply + accumulate)
    // Arithmetic intensity for prefill is high (batched matmul) â†’ compute-bound
    var prefillTps;
    if (tflops > 0) {
      prefillTps = MathEngine.calcPrefillSpeed(tflops, act, (nG > 1 && !isPipelineParallel) ? nG * tpEfficiency : tpEfficiency);
    } else {
      prefillTps = decodeTps * 4;
    }

    // â”€â”€â”€ 4. TTFT (time to first token) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // TTFT = time to read weights + time to process C tokens
    // For bandwidth-bound estimate: TTFT = P_active * qW / BW_effective
    var ttft = (wBytes / bw) * 1000; // ms, single-GPU bandwidth-bound
    if (nG > 1 && interconnBW > 0 && !isPipelineParallel) {
      ttft = ((wBytes / (bw * nG)) * 1000) / tpEfficiency; // ms with TP
    } else if (nG > 1 && isPipelineParallel) {
      ttft = ((wBytes / bw) * 1000) / tpEfficiency; // PP: each stage reads 1/N but sequentially
    }

    // â”€â”€â”€ 5. APPLY RAM OFFLOAD DEGRADATION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // Real model: T_decode = W_vram/BW_HBM + W_ram/BW_transfer
    // This is more accurate than a simple degradation factor because
    // it models the two data paths separately:
    //   - VRAM-resident layers: read from HBM at full GPU bandwidth
    //   - RAM-offloaded layers: read from RAM via PCIe at transfer bandwidth
    var ramPerfNote = document.getElementById("ramPerfNote");
    if (ramUsable > 0 && ramFraction > 0) {
      ramPerfNote.style.display = "block";
      var wVramGB = (1 - ramFraction) * wBytes; // weights in VRAM (GB)
      var wRamGB = ramFraction * wBytes; // weights in RAM (GB)

      // Effective BW for multi-GPU VRAM portion
      var effectiveBwHBM = nG > 1 ? bw * nG * tpEfficiency : bw;

      // Decode time per token = time for VRAM portion + time for RAM portion
      // This is the key formula that properly models the mixed bandwidth:
      var tVramDecode = wVramGB / effectiveBwHBM; // seconds for VRAM-resident weights
      var tRamDecode = wRamGB / transferBW; // seconds for RAM-offloaded weights
      var tDecodeTotal = tVramDecode + tRamDecode; // total seconds per token

      decodeTps = 1 / tDecodeTotal;

      // Prefill also affected by RAM offload
      if (tflops > 0) {
        // Prefill is compute-bound for VRAM layers, bandwidth-bound for RAM layers
        // The RAM-offloaded layers can't use GPU compute effectively
        // because they're waiting for data to arrive via PCIe
        var prefillDegradation =
          1 / (1 - ramFraction + ramFraction * busWallRatio);
        prefillTps = prefillTps * (1 / prefillDegradation);
      }

      // TTFT: reading prompt also needs RAM-offloaded weights
      // Each token of the prompt needs all weights, not just active ones
      ttft = tDecodeTotal * ctx * 1000; // approximate: each prompt token also needs weight reading

      ramPerfNote.innerHTML =
        "<strong>Bus Wall active:</strong> " +
        (ramFraction * 100).toFixed(0) +
        "% of weights in RAM, " +
        "running " +
        busWallRatio.toFixed(0) +
        "\u00D7 slower than HBM. " +
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
        "PCIe: " +
        pcieEffective.toFixed(1) +
        " GB/s (Gen" +
        pcieGen +
        " x" +
        pcieLanes +
        " \u00D7 " +
        PCIE_EFFICIENCY +
        "), " +
        "RAM: " +
        ramEffective.toFixed(1) +
        " GB/s (\u03B7=" +
        RAM_EFFICIENCY +
        (numaAware ? ", NUMA-aware" : ", no NUMA \u00D70.65") +
        ").";
    } else {
      ramPerfNote.style.display = "none";
      ramPerfNote.innerHTML = "";
    }

    // â”€â”€â”€ 6. DERIVED METRICS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    var latPerTok = 1000 / decodeTps;
    var time100 = ttft / 1000 + 100 / decodeTps;
    var time1000 = ttft / 1000 + 1000 / decodeTps;
    var throughput = decodeTps * uS;

    // HBM bandwidth actually used per decode step
    var bwUsed = wBytes * decodeTps; // GB/s consumed by weight reading
    if (nG > 1 && !isPipelineParallel)
      bwUsed = bwUsed / (nG * tpEfficiency || 1);

    // Arithmetic intensity (FLOP/byte) for roofline analysis
    var arithIntensity = 2 / qW; // FLOPs per byte for decode

    // Bottleneck analysis with roofline context
    var bottleneckLabel = "HBM bandwidth";
    if (ramUsable > 0) {
      bottleneckLabel =
        transferBottleneck +
        "+RAM (" +
        busWallRatio.toFixed(0) +
        "\u00D7 wall)";
    } else if (nG > 1 && effectiveInterconnType === "pcie") {
      bottleneckLabel = "PCIe interconnect";
    }

    // â”€â”€â”€ 7. DISPLAY RESULTS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    document.getElementById("pPrefill").textContent =
      prefillTps >= 1000
        ? (prefillTps / 1000).toFixed(1) + "k"
        : prefillTps.toFixed(0);
    document.getElementById("pDecode").textContent =
      decodeTps >= 1000
        ? (decodeTps / 1000).toFixed(1) + "k"
        : decodeTps.toFixed(1);
    document.getElementById("pLat").textContent =
      latPerTok >= 1000
        ? (latPerTok / 1000).toFixed(1) + "s"
        : latPerTok.toFixed(1);
    document.getElementById("pTTFT").textContent =
      ttft >= 1000 ? (ttft / 1000).toFixed(1) + "s" : ttft.toFixed(0);
    document.getElementById("p100").textContent =
      time100 >= 60
        ? (time100 / 60).toFixed(1) + "m"
        : time100.toFixed(1);
    document.getElementById("p1000").textContent =
      time1000 >= 60
        ? (time1000 / 60).toFixed(1) + "m"
        : time1000.toFixed(1);
    document.getElementById("pThr").textContent =
      throughput >= 1000
        ? (throughput / 1000).toFixed(1) + "k"
        : throughput.toFixed(0);
    document.getElementById("pBwUsed").textContent = bwUsed.toFixed(0);
    document.getElementById("pBottleneck").textContent = bottleneckLabel;
    document.getElementById("pArithInt").textContent =
      arithIntensity.toFixed(1);
    document.getElementById("perfCard").style.display = "block";

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

    // â”€â”€â”€ 8. POWER & COST CALCULATIONS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
