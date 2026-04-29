// ============================================================================
// VARIANT DISCOVERY & RENDERING
// ============================================================================

function discoverVariants(modelId, api) {
  var variants = [];
  var useTok = document.getElementById("hfUseTok").checked;
  var tok = useTok ? document.getElementById("hfTok").value.trim() : "";
  var headers = tok ? { Authorization: "Bearer " + tok } : {};
  var base = "https://huggingface.co/";
  if (api && api.siblings) {
    var ggufFiles = api.siblings.filter(function (s) {
      return s.rfilename && s.rfilename.toLowerCase().endsWith(".gguf");
    });
    ggufFiles.forEach(function (s) {
      var parsed = parseGGUFFilename(s.rfilename);
      if (parsed) {
        variants.push({ id: modelId, type: "GGUF", label: parsed.info.label, bpb: parsed.info.bpb, details: parsed.info.desc, file: s.rfilename, source: "siblings" });
      }
    });
  }
  var org = modelId.split("/")[0];
  var name = modelId.split("/").slice(1).join("/");
  var baseName = name.replace(/[-_](gguf|gptq|awq|exl2|quantized|int4|int8|nf4|fp16|bf16|q4|q8).*$/i, "");
  var searchQueries = [];
  searchQueries.push(baseName + " GGUF");
  searchQueries.push(baseName + " GPTQ");
  searchQueries.push(baseName + " AWQ");
  if (baseName !== name) searchQueries.push(name + " GGUF");
  var allSearchPromises = searchQueries.map(function (q) {
    return fetch(base + "api/models?search=" + encodeURIComponent(q) + "&sort=downloads&limit=5&direction=-1", { headers: headers })
      .then(function (r) { return r.ok ? r.json() : []; })
      .catch(function () { return []; });
  });
  return Promise.all(allSearchPromises).then(function (allResults) {
    var seen = {};
    seen[modelId] = true;
    allResults.forEach(function (results) {
      if (!results || !results.length) return;
      results.forEach(function (m) {
        if (seen[m.id]) return;
        seen[m.id] = true;
        var idL = m.id.toLowerCase();
        var isGGUF = /gguf/i.test(idL) || (m.siblings && m.siblings.some(function (s) { return s.rfilename && s.rfilename.toLowerCase().endsWith(".gguf"); }));
        var isGPTQ = /gptq/i.test(idL);
        var isAWQ = /awq/i.test(idL);
        var isEXL2 = /exl2/i.test(idL);
        if (isGGUF) {
          if (m.siblings) {
            var ggufFiles = m.siblings.filter(function (s) { return s.rfilename && s.rfilename.toLowerCase().endsWith(".gguf"); });
            ggufFiles.forEach(function (s) {
              var parsed = parseGGUFFilename(s.rfilename);
              if (parsed) {
                variants.push({ id: m.id, type: "GGUF", label: parsed.info.label, bpb: parsed.info.bpb, details: parsed.info.desc + " (" + m.id + ")", file: s.rfilename, source: "search" });
              }
            });
          }
        } else if (isGPTQ) {
          var bits = 4;
          if (/int8|8bit/i.test(idL)) bits = 8;
          if (/int4|4bit/i.test(idL)) bits = 4;
          if (/int3|3bit/i.test(idL)) bits = 3;
          if (/int2|2bit/i.test(idL)) bits = 2;
          variants.push({ id: m.id, type: "GPTQ", label: "GPTQ-" + bits + "bit", bpb: bits / 8 + 0.06, details: bits + "-bit GPTQ (" + m.id + ")", source: "search" });
        } else if (isAWQ) {
          var bits = 4;
          if (/8bit/i.test(idL)) bits = 8;
          variants.push({ id: m.id, type: "AWQ", label: "AWQ-" + bits + "bit", bpb: bits / 8 + 0.06, details: bits + "-bit AWQ (" + m.id + ")", source: "search" });
        } else if (isEXL2) {
          var bits = 4;
          if (/8bit/i.test(idL)) bits = 8;
          if (/6b?it/i.test(idL)) bits = 6;
          if (/5b?it/i.test(idL)) bits = 5;
          if (/3b?it/i.test(idL)) bits = 3;
          if (/2b?it/i.test(idL)) bits = 2;
          variants.push({ id: m.id, type: "EXL2", label: "EXL2-" + bits + "bit", bpb: bits / 8 + 0.06, details: bits + "-bit EXL2 (" + m.id + ")", source: "search" });
        }
      });
    });
    var deduped = {};
    variants.forEach(function (v) { var key = v.type + "-" + v.label; if (!deduped[key]) deduped[key] = v; });
    var sorted = Object.values(deduped);
    sorted.sort(function (a, b) {
      var typeOrder = { GGUF: 0, GPTQ: 1, AWQ: 2, EXL2: 3, BNB: 4 };
      var ta = typeOrder[a.type] !== undefined ? typeOrder[a.type] : 9;
      var tb = typeOrder[b.type] !== undefined ? typeOrder[b.type] : 9;
      if (ta !== tb) return ta - tb;
      return a.bpb - b.bpb;
    });
    return sorted;
  });
}

function renderVariants(variants) {
  var section = document.getElementById("varSection");
  var grid = document.getElementById("varGrid");
  var status = document.getElementById("varStatus");
  if (!variants || variants.length === 0) { section.className = "var-section"; return; }
  section.className = "var-section open";
  // Label for the imported model's own card: derive from HF lineage metadata.
  // Possible base_model_relation values: finetune | adapter | quantized | merge.
  // Fall back to "FINE-TUNE" when base_model exists but relation is unset,
  // and to "BASE MODEL" only when no parent reference is present at all.
  var selfLabel = "BASE MODEL";
  if (gImp) {
    if (gImp.baseModelRelation) {
      selfLabel = gImp.baseModelRelation.toUpperCase();
    } else if (gImp.baseModel) {
      selfLabel = "FINE-TUNE";
    }
  }
  var html = '<div class="var-card' + (gSelectedVar === -1 ? " selected" : "") + '" onclick="selectVariant(-1)">' +
    '<div class="vc-type">' + selfLabel + '</div>' +
    '<div class="vc-name">' + escHtml(gImp ? gImp.name : "Original") + '</div>' +
    '<div class="vc-detail">' + (gImp && gImp.tType ? gImp.tType : "BF16/FP16") + '</div>' +
    '<div class="vc-size">' + (gImp && gImp.p ? (gImp.p * (gImp.tBpb || 2)).toFixed(1) + " GB est." : "") + '</div></div>';
  variants.forEach(function (v, idx) {
    html += '<div class="var-card' + (gSelectedVar === idx ? " selected" : "") + '" onclick="selectVariant(' + idx + ')">' +
      '<div class="vc-type">' + escHtml(v.type) + '</div>' +
      '<div class="vc-name">' + escHtml(v.label) + '</div>' +
      '<div class="vc-detail">' + escHtml(v.details || "") + '</div>' +
      '<div class="vc-size">' + (gImp && gImp.p ? (gImp.p * v.bpb).toFixed(1) + " GB est." : "") + '</div></div>';
  });
  grid.innerHTML = html;
  status.textContent = variants.length + " quantized variant" + (variants.length !== 1 ? "s" : "") + " found";
}

function selectVariant(idx) {
  gSelectedVar = idx;
  if (idx === -1) { if (gImp) setClosestQW(gImp.tBpb || 2); }
  else { var v = gVariants[idx]; if (v) setClosestQW(v.bpb); }
  renderVariants(gVariants);
  calc();
}

function setClosestQW(targetBpb) {
  var qW = document.getElementById("qW");
  var opts = qW.options;
  var bestIdx = 0, bestDiff = 999;
  for (var i = 0; i < opts.length; i++) {
    var v = parseFloat(opts[i].value);
    var diff = Math.abs(v - targetBpb);
    if (diff < bestDiff) { bestDiff = diff; bestIdx = i; }
  }
  if (bestDiff > 0.05) {
    var hasCustom = false;
    for (var i = 0; i < opts.length; i++) {
      if (opts[i].getAttribute("data-custom") === "true") {
        opts[i].value = targetBpb;
        opts[i].text = "Custom (" + targetBpb.toFixed(2) + " B/param)";
        opts[i].setAttribute("data-custom", "true");
        qW.selectedIndex = i;
        hasCustom = true;
        break;
      }
    }
    if (!hasCustom) {
      var o = new Option("Custom (" + targetBpb.toFixed(2) + " B/param)", targetBpb);
      o.setAttribute("data-custom", "true");
      qW.appendChild(o);
      qW.selectedIndex = qW.options.length - 1;
    }
  } else {
    qW.selectedIndex = bestIdx;
  }
}
