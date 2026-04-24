function gemmaOv(modelId) {
  var id = modelId.toLowerCase();
  if (
    id.indexOf("qwen3.6-35b-a3b") !== -1 ||
    id.indexOf("qwen3.5-35b-a3b") !== -1
  ) {
    return {
      totalB: 35,
      activeB: 3,
      layers: 40,
      context: "256K tokens (up to 1M)",
      vocab: "248K",
      experts: "8 routed / 256 total + 1 shared",
      modalities: "Text, Image, Video",
      architecture: "MoE",
    };
  }
  if (
    id.indexOf("gemma-4-26b-a4b") !== -1 ||
    id.indexOf("gemma-4-26b_a4b") !== -1
  ) {
    return {
      totalB: 25.2,
      activeB: 3.8,
      layers: 30,
      sliding: "1024 tokens",
      context: "256K tokens",
      vocab: "262K",
      experts: "8 active / 128 total + 1 shared",
      modalities: "Text, Image",
      vision: "~550M params",
      architecture: "MoE",
    };
  }
  if (
    id.indexOf("gemma-4-31b") !== -1 ||
    id.indexOf("gemma-4-31b") !== -1
  ) {
    return {
      totalB: 30.7,
      activeB: 30.7,
      context: "128K tokens",
      vocab: "262K",
      modalities: "Text, Image",
      vision: "~550M params",
      architecture: "Dense",
    };
  }
  return null;
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// SEARCH AUTOCOMPLETE
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function onSearchInput() {
  clearTimeout(gSearchTimer);
  var q = document.getElementById("hfId").value.trim();
  var drop = document.getElementById("srchDrop");
  if (q.length < 2) {
    drop.className = "srch-drop";
    return;
  }
  gSearchTimer = setTimeout(function () {
    doSearch(q);
  }, 450);
}

function doSearch(query) {
  var drop = document.getElementById("srchDrop");
  drop.innerHTML =
    '<div class="srch-load">Searching HuggingFace...</div>';
  drop.className = "srch-drop open";
  var useTok = document.getElementById("hfUseTok").checked;
  var tok = useTok ? document.getElementById("hfTok").value.trim() : "";
  var headers = tok ? { Authorization: "Bearer " + tok } : {};
  fetch(
    "https://huggingface.co/api/models?search=" +
      encodeURIComponent(query) +
      "&sort=downloads&limit=12&direction=-1",
    { headers: headers },
  )
    .then(function (r) {
      return r.ok ? r.json() : [];
    })
    .then(function (results) {
      if (!results || !results.length) {
        drop.innerHTML = '<div class="srch-load">No models found</div>';
        return;
      }
      drop.innerHTML = results
        .map(function (m) {
          var tag = m.pipeline_tag || m.library_name || "";
          var dl = m.downloads
            ? m.downloads > 1e6
              ? (m.downloads / 1e6).toFixed(1) + "M"
              : m.downloads > 1e3
                ? (m.downloads / 1e3).toFixed(0) + "K"
                : m.downloads
            : "";
          return (
            '<div class="srch-item" onclick="pickSearch(\'' +
            escHtml(m.id) +
            "')\">" +
            '<div><div class="si-name">' +
            escHtml(m.id) +
            "</div>" +
            (dl ? '<div class="si-id">' + dl + " downloads</div>" : "") +
            "</div>" +
            (tag
              ? '<span class="si-tag">' + escHtml(tag) + "</span>"
              : "") +
            "</div>"
          );
        })
        .join("");
    })
    .catch(function () {
      drop.innerHTML = '<div class="srch-load">Search failed</div>';
    });
}

function pickSearch(modelId) {
  document.getElementById("hfId").value = modelId;
  document.getElementById("srchDrop").className = "srch-drop";
}

function escHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/'/g, "&#39;")
    .replace(/"/g, "&quot;");
}

document.addEventListener("click", function (e) {
  var wrap = document.querySelector(".srch-wrap");
  if (wrap && !wrap.contains(e.target)) {
    document.getElementById("srchDrop").className = "srch-drop";
  }
});

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// SAFETENSORS INDEX PARSER
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function parseSafIdx(text) {
  try {
    var j = JSON.parse(text);
    var meta = j.metadata || {};
    var wm = j.weight_map || {};
    var keys = Object.keys(wm);
    var shards = Object.values(wm).filter(function (v, i, a) {
      return a.indexOf(v) === i;
    }).length;
    var vis = keys.filter(function (k) {
      return k.indexOf("vision") !== -1;
    }).length;
    var rtr = keys.filter(function (k) {
      return k.indexOf("router") !== -1;
    }).length;
    var dtypes = [];
    if (meta.dtype) dtypes.push(meta.dtype);
    var dtypeKeys = Object.keys(meta).filter(function (k) {
      return (
        k.toLowerCase().indexOf("dtype") !== -1 ||
        k.toLowerCase().indexOf("type") !== -1
      );
    });
    dtypeKeys.forEach(function (k) {
      var v = meta[k];
      if (typeof v === "string" && v) dtypes.push(v);
    });
    return {
      meta: meta,
      count: keys.length,
      shards: shards,
      vis: vis,
      rtr: rtr,
      dtypes: dtypes,
    };
  } catch (e) {
    return null;
  }
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// CONFIG PARSER
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function parseCfg(c) {
  var tcfg = c.text_config || c;
  var nExp = tcfg.num_experts || tcfg.num_local_experts || 0;
  var ept = tcfg.num_experts_per_tok || 1;
  var isMoeModel =
    nExp > 1 ||
    (c.architectures &&
      c.architectures[0] &&
      c.architectures[0].indexOf("Moe") !== -1);
  var h = tcfg.hidden_size || tcfg.d_model || 0;
  var L = tcfg.num_hidden_layers || tcfg.n_layer || 0;
  var nh = tcfg.num_attention_heads || tcfg.n_head || 0;
  var nkv = tcfg.num_key_value_heads || nh;
  var V = tcfg.vocab_size || 0;
  var ctx = tcfg.max_position_embeddings || 0;
  var inter =
    tcfg.moe_intermediate_size || tcfg.intermediate_size || h * 4;
  var hd = tcfg.head_dim || (nh > 0 ? h / nh : 0);
  var embP = V * h;
  var lyrP = h * h * 4 + h * inter * 2 * (isMoeModel ? ept : 1) + h * 4;
  var totP = embP + L * lyrP;
  var quantCfg = c.quantization_config || null;
  var quantInfo = parseQuantConfig(quantCfg);
  var torchDtype = c.torch_dtype || null;
  return {
    paramCount: totP > 1e6 ? totP : 0,
    active: isMoeModel
      ? embP + L * (h * h * 4 + h * inter * 2 * ept + h * 4)
      : totP,
    h: h,
    L: L,
    nh: nh,
    nkv: nkv,
    V: V,
    ctx: ctx,
    hd: hd,
    nExp: nExp,
    arch: isMoeModel ? "MoE" : "Dense",
    src: totP > 1e6 ? "Config heuristic" : "Incomplete metadata",
    quantInfo: quantInfo,
    torchDtype: torchDtype,
  };
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// QUANTIZATION CONFIG PARSER
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function parseQuantConfig(qc) {
  if (!qc || typeof qc !== "object") return null;
  var method = qc.quant_method || qc.format || "";
  var mLow = method.toLowerCase();
  if (mLow === "gptq") {
    var bits = qc.bits || 4;
    var groupSize = qc.group_size || 128;
    var descAct = qc.desc_act || false;
    var sym = qc.sym !== undefined ? qc.sym : true;
    return {
      type: "GPTQ",
      bits: bits,
      bpb: bits / 8 + (groupSize > 0 ? 0.06 : 0),
      details:
        bits +
        "-bit, group_size=" +
        groupSize +
        (descAct ? ", desc_act" : "") +
        (sym ? ", symmetric" : ""),
      label: "GPTQ-" + bits + "bit",
    };
  }
  if (mLow === "awq") {
    var bits = qc.bits || qc.w_bit || 4;
    var groupSize = qc.group_size || qc.q_group_size || 128;
    var zeroPoint = qc.zero_point !== undefined ? qc.zero_point : true;
    return {
      type: "AWQ",
      bits: bits,
      bpb: bits / 8 + (groupSize > 0 ? 0.06 : 0),
      details:
        bits +
        "-bit, group_size=" +
        groupSize +
        (zeroPoint ? ", zero_point" : ""),
      label: "AWQ-" + bits + "bit",
    };
  }
  if (mLow.indexOf("exl2") !== -1 || mLow.indexOf("exllama") !== -1) {
    var bits = qc.bits || 4;
    return {
      type: "EXL2",
      bits: bits,
      bpb: bits / 8 + 0.06,
      details: bits + "-bit (EXL2 format)",
      label: "EXL2-" + bits + "bit",
    };
  }
  if (mLow === "gguf" || mLow === "llama.cpp") {
    return {
      type: "GGUF",
      bits: 0,
      bpb: 0,
      details: "GGUF quantized model",
      label: "GGUF",
    };
  }
  if (mLow.indexOf("bitsandbytes") !== -1 || mLow.indexOf("nf4") !== -1) {
    var bits = qc.bits || 4;
    return {
      type: "BNB/NF4",
      bits: bits,
      bpb: bits / 8,
      details: bits + "-bit (bitsandbytes/NF4)",
      label: "NF4-" + bits + "bit",
    };
  }
  if (method) {
    var bits = qc.bits || 0;
    return {
      type: method.toUpperCase(),
      bits: bits,
      bpb: bits > 0 ? bits / 8 : 0,
      details: method + (bits ? " " + bits + "-bit" : ""),
      label: method.toUpperCase() + (bits ? "-" + bits + "bit" : ""),
    };
  }
  return null;
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// GGUF FILENAME PARSER
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function parseGGUFFilename(filename) {
  var fl = filename.toLowerCase();
  var match = fl.match(/[.\-_](q\d+_?k?[msl]?|f16|f32|bf16)\.gguf$/i);
  if (!match) return null;
  var key = match[1].replace(/[\-_]/g, "_").toLowerCase();
  if (key === "q4km") key = "q4_k_m";
  if (key === "q4ks") key = "q4_k_s";
  if (key === "q5km") key = "q5_k_m";
  if (key === "q5ks") key = "q5_k_s";
  if (key === "q3km") key = "q3_k_m";
  if (key === "q3ks") key = "q3_k_s";
  if (key === "q3kl") key = "q3_k_l";
  if (key === "q6k") key = "q6_k";
  if (key === "q2k") key = "q2_k";
  if (key === "q8_0" || key === "q8") key = "q8_0";
  if (key === "q4_0") key = "q4_0";
  return GGUF_QUANTS[key]
    ? { key: key, info: GGUF_QUANTS[key], file: filename }
    : null;
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// TENSOR TYPE DETECTION
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function detectTensorType(api, cfgData, idxMeta) {
  var result = {
    type: "Unknown",
    bpb: 2,
    source: "default",
    isQuantized: false,
    quantInfo: null,
  };
  if (cfgData && cfgData.quantization_config) {
    var qi = parseQuantConfig(cfgData.quantization_config);
    if (qi) {
      result.type = qi.label;
      result.bpb = qi.bpb || qi.bits / 8;
      result.source = "config.json quantization_config";
      result.isQuantized = true;
      result.quantInfo = qi;
      return result;
    }
  }
  if (api && api.safetensors && api.safetensors.parameters) {
    var pTypes = Object.keys(api.safetensors.parameters);
    if (pTypes.length > 0) {
      var allTypes = pTypes.join(", ");
      var mainType = pTypes[0];
      var isQuant = /int4|int8|fp4|nf4|u4|u8|q4|q8/i.test(allTypes);
      if (isQuant) {
        result.type = allTypes;
        result.isQuantized = true;
        result.source = "API safetensors.parameters";
        var tl = allTypes.toLowerCase();
        if (/int4|fp4|nf4|u4|q4/.test(tl)) result.bpb = 0.5;
        else if (/int8|fp8|u8|q8/.test(tl)) result.bpb = 1;
        else result.bpb = 0.5;
      } else {
        result.type = allTypes;
        result.bpb = bpbFromDtype(mainType);
        result.source = "API safetensors.parameters";
      }
      return result;
    }
  }
  if (idxMeta && idxMeta.dtypes && idxMeta.dtypes.length > 0) {
    var dtypeStr = idxMeta.dtypes.join(", ");
    result.type = dtypeStr;
    result.bpb = bpbFromDtype(idxMeta.dtypes[0]);
    result.source = "safetensors index metadata";
    return result;
  }
  if (cfgData && cfgData.torch_dtype) {
    result.type = cfgData.torch_dtype;
    result.bpb = bpbFromDtype(cfgData.torch_dtype);
    result.source = "config.json torch_dtype";
    return result;
  }
  return result;
}

function bpbFromDtype(dtype) {
  var d = (dtype || "").toLowerCase();
  if (d.indexOf("fp32") !== -1 || d.indexOf("float32") !== -1) return 4;
  if (d.indexOf("bf16") !== -1 || d.indexOf("bfloat16") !== -1) return 2;
  if (d.indexOf("fp16") !== -1 || d.indexOf("float16") !== -1) return 2;
  if (d.indexOf("fp8") !== -1 || d.indexOf("float8") !== -1) return 1;
  if (d.indexOf("int8") !== -1) return 1;
  if (
    d.indexOf("int4") !== -1 ||
    d.indexOf("nf4") !== -1 ||
    d.indexOf("fp4") !== -1
  )
    return 0.5;
  if (d.indexOf("int2") !== -1) return 0.25;
  return 2;
}
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// BRANCH DETECTION HELPER
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function fetchWithBranches(urlBase, branches, opts) {
  var chain = Promise.reject();
  branches.forEach(function (br) {
    chain = chain.catch(function () {
      return fetch(urlBase + "/raw/" + br + "/config.json", opts).then(
        function (r) {
          if (!r.ok) throw new Error("not found on branch " + br);
          return { response: r, branch: br };
        },
      );
    });
  });
  return chain;
}
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// HF FETCH
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function hfFetch() {
  var modelId = document.getElementById("hfId").value.trim();
  if (!modelId) return;
  var btn = document.getElementById("hfBtn");
  var wbox = document.getElementById("hfW");
  wbox.style.display = "none";
  wbox.textContent = "";
  btn.disabled = true;
  btn.textContent = "Importing\u2026";
  var useTok = document.getElementById("hfUseTok").checked;
  var tok = useTok ? document.getElementById("hfTok").value.trim() : "";
  var opts = tok ? { headers: { Authorization: "Bearer " + tok } } : {};
  var base = "https://huggingface.co/";
  fetch(base + "api/models/" + modelId, opts)
    .then(function (r) {
      if (!r.ok) throw new Error("Model not found: " + modelId);
      return r.json();
    })
    .then(function (api) {
      gBranch = api.defaultBranch || "main";
      var pConf = fetch(
        base + modelId + "/raw/" + gBranch + "/config.json",
        opts,
      )
        .then(function (r) {
          return r.ok
            ? r.json().catch(function () {
                return null;
              })
            : null;
        })
        .catch(function () {
          return null;
        });
      var pIdx = fetch(
        base +
          modelId +
          "/resolve/" +
          gBranch +
          "/model.safetensors.index.json",
        opts,
      )
        .then(function (r) {
          return r.ok
            ? r.text().catch(function () {
                return null;
              })
            : null;
        })
        .catch(function () {
          return null;
        });
      return Promise.all([pConf, pIdx]).then(function (data) {
        return { api: api, cfgData: data[0], idxTxt: data[1] };
      });
    })
    .then(function (result) {
      var api = result.api;
      var cfgData = result.cfgData;
      var idxTxt = result.idxTxt;
      var card = api && api.cardData ? api.cardData : {};
      var idx = idxTxt ? parseSafIdx(idxTxt) : null;
      var meta = cfgData
        ? parseCfg(cfgData)
        : {
            paramCount: 0,
            active: 0,
            h: 0,
            L: 0,
            nh: 0,
            nkv: 0,
            V: 0,
            ctx: 0,
            hd: 0,
            nExp: 0,
            arch: "Dense",
            src: "Incomplete metadata",
            quantInfo: null,
            torchDtype: null,
          };
      var fallbackId = null;
      if (!cfgData && card.base_model) {
        fallbackId = Array.isArray(card.base_model)
          ? card.base_model[0]
          : card.base_model;
        meta.src = "Base-model fallback (" + fallbackId + ")";
      }
      if (idx && idx.meta && idx.meta.total_parameters) {
        meta.paramCount = idx.meta.total_parameters;
        meta.src = "Exact safetensors index";
      } else if (api && api.safetensors && api.safetensors.total) {
        meta.paramCount = api.safetensors.total;
        meta.src = "Exact API metadata";
      }
      var ov = gemmaOv(modelId);
      var totalB =
        ov && ov.totalB
          ? ov.totalB
          : meta.paramCount
            ? meta.paramCount / 1e9
            : null;
      var tensorInfo = detectTensorType(api, cfgData, idx);
      var tType = tensorInfo.type;
      var tBpb = tensorInfo.bpb;
      var idL = modelId.toLowerCase();
      var isMoE =
        (ov && ov.architecture === "MoE") ||
        /a[0-9]+b|moe/i.test(idL) ||
        (meta.arch && meta.arch.indexOf("MoE") !== -1) ||
        (idx && idx.rtr > 5);
      var activeB =
        ov && ov.activeB
          ? ov.activeB
          : meta.active && meta.active / 1e9 !== totalB
            ? meta.active / 1e9
            : null;
      if (!activeB) activeB = totalB;
      if (!isMoE && activeB && totalB && activeB > totalB)
        activeB = totalB;
      function st(id, v) {
        document.getElementById(id).textContent =
          v !== null && v !== undefined && v !== "" ? v : "\u2014";
      }
      var tensorDisplay = tType;
      if (tensorInfo.isQuantized) {
        tensorDisplay = tType + " (quantized)";
      } else if (tType !== "Unknown") {
        tensorDisplay = tType;
      }
      if (tensorInfo.quantInfo && tensorInfo.quantInfo.details) {
        tensorDisplay =
          tensorInfo.quantInfo.label +
          " \u2014 " +
          tensorInfo.quantInfo.details;
      }
      st("mPS", meta.src);
      st("mTask", api ? api.pipeline_tag || card.pipeline_tag : null);
      st(
        "mDL",
        api && api.downloads ? api.downloads.toLocaleString() : null,
      );
      st("mLib", api && api.library_name);
      st("mLic", card.license || (api && api.license));
      st(
        "mBase",
        Array.isArray(card.base_model)
          ? card.base_model.join(", ")
          : card.base_model,
      );
      st("mBaseR", card.base_model_relation);
      st(
        "mMod",
        api && api.lastModified ? api.lastModified.slice(0, 10) : null,
      );
      st("mTP", totalB ? totalB.toFixed(2) + "B" : null);
      st("mAP", activeB ? activeB.toFixed(2) + "B" : null);
      st(
        "mExp",
        ov && ov.experts
          ? ov.experts
          : meta.nExp > 1
            ? meta.nExp + " total"
            : null,
      );
      st("mSW", ov && ov.sliding);
      st(
        "mCtx",
        ov && ov.context
          ? ov.context
          : meta.ctx
            ? (meta.ctx / 1024).toFixed(0) + "K tokens"
            : null,
      );
      st(
        "mVoc",
        ov && ov.vocab
          ? ov.vocab
          : meta.V
            ? (meta.V / 1000).toFixed(0) + "K"
            : null,
      );
      st("mTensor", tensorDisplay);
      st(
        "mMod2",
        ov && ov.modalities
          ? ov.modalities
          : api &&
              api.tags &&
              /vision|image|visual|multi|video/i.test(api.tags.join(" "))
            ? "Multimodal"
            : "Text",
      );
      st("mVis", ov && ov.vision);
      st(
        "mLang",
        Array.isArray(card.language)
          ? card.language.slice(0, 4).join(", ")
          : card.language,
      );
      st(
        "mDS",
        Array.isArray(card.datasets)
          ? card.datasets.slice(0, 4).join(", ")
          : card.datasets,
      );
      st(
        "mTags",
        api && Array.isArray(api.tags)
          ? api.tags.slice(0, 7).join(", ")
          : null,
      );
      st(
        "mEval",
        Array.isArray(card.eval_results)
          ? card.eval_results
              .slice(0, 2)
              .map(function (r) {
                return (
                  (r.task && r.task.name) ||
                  (r.metric && r.metric.name) ||
                  ""
                );
              })
              .join(", ")
          : null,
      );
      var ws = [];
      if (!meta.paramCount) ws.push("parameter count missing");
      if (!meta.h) ws.push("hidden size missing");
      if (!meta.L) ws.push("layer count missing");
      if (!meta.nh) ws.push("attention heads missing");
      if (meta.h && meta.nh && meta.h % meta.nh !== 0)
        ws.push("hidden_size not divisible by num_attention_heads");
      if (!isMoE && activeB && totalB && activeB > totalB)
        ws.push("active params clamped to total (dense model)");
      if (fallbackId)
        ws.push(
          "no direct config.json \u2014 using base model: " + fallbackId,
        );
      if (idx && idx.shards)
        ws.push(idx.count + " tensors across " + idx.shards + " shards");
      if (idx && idx.vis > 0)
        ws.push("vision tensors detected in safetensors index");
      if (tensorInfo.isQuantized)
        ws.push(
          "quantized model detected (" +
            tType +
            "), weight precision auto-set",
        );
      if (tensorInfo.source !== "default")
        ws.push("tensor type from: " + tensorInfo.source);
      if (ws.length) {
        wbox.textContent = ws.join(" \u00b7 ");
        wbox.style.display = "block";
      }
      gImp = {
        tType: tType,
        tBpb: tBpb,
        isQuant: tensorInfo.isQuantized,
        quantInfo: tensorInfo.quantInfo,
        name: modelId,
        p: totalB || 0,
        active: activeB || totalB || 0,
        L: ov && ov.layers ? ov.layers : meta.L || 0,
        h: meta.h || 0,
        nh: meta.nh || 0,
        nkv: meta.nkv || meta.nh || 0,
        headDim: meta.hd || 0,
        arch:
          ov && ov.architecture
            ? ov.architecture
            : meta.arch || (isMoE ? "MoE" : "Dense"),
        src: meta.src || "Imported",
        task: api && api.pipeline_tag ? api.pipeline_tag : "\u2014",
      };
      gRaw = JSON.parse(JSON.stringify(gImp));
      gSelectedVar = -1;
      hfFill(gImp);
      hfApply();
      document.getElementById("varStatus").textContent =
        "Searching for quantized variants...";
      discoverVariants(modelId, api)
        .then(function (variants) {
          gVariants = variants;
          renderVariants(variants);
        })
        .catch(function () {
          gVariants = [];
          renderVariants([]);
        });
    })
    .catch(function (err) {
      wbox.textContent = "Import failed: " + err.message;
      wbox.style.display = "block";
    })
    .finally(function () {
      btn.disabled = false;
      btn.textContent = "Import model metadata";
    });
}

function hfFill(m) {
  if (!m) return;
  function sv(id, v) {
    var el = document.getElementById(id);
    if (el) el.value = v !== null && v !== undefined ? v : "";
  }
  sv("eP", m.p ? m.p.toFixed(2) : "");
  sv("ePA", m.active ? m.active.toFixed(2) : "");
  sv("eL", m.L || "");
  sv("eH", m.h || "");
  sv("eNH", m.nh || "");
  sv("eNKV", m.nkv || "");
  sv("eHD", m.headDim || "");
  sv("eTask", m.task || "");
  if (m.tBpb) setClosestQW(m.tBpb);
  var a = document.getElementById("eArch");
  if (a)
    a.value =
      (m.arch || "Dense").toLowerCase() === "moe" ? "moe" : "dense";
}

function hfSync() {
  if (!gImp) return;
  gImp.p = parseFloat(document.getElementById("eP").value) || gImp.p;
  gImp.active =
    parseFloat(document.getElementById("ePA").value) || gImp.active;
  gImp.L = parseInt(document.getElementById("eL").value) || gImp.L;
  gImp.h = parseInt(document.getElementById("eH").value) || gImp.h;
  gImp.nh = parseInt(document.getElementById("eNH").value) || gImp.nh;
  gImp.nkv = parseInt(document.getElementById("eNKV").value) || gImp.nkv;
  gImp.headDim =
    parseInt(document.getElementById("eHD").value) || gImp.headDim;
  gImp.arch =
    document.getElementById("eArch").value === "moe" ? "MoE" : "Dense";
  gImp.task = document.getElementById("eTask").value || gImp.task;
  hfApply();
}

function hfReset() {
  if (!gRaw) return;
  gImp = JSON.parse(JSON.stringify(gRaw));
  gSelectedVar = -1;
  hfFill(gImp);
  hfApply();
  renderVariants(gVariants);
}

function hfApply() {
  var conf = document.getElementById("hfConf");
  if (!gImp) {
    conf.textContent = "Confidence: not imported";
    return;
  }
  var m = gImp;
  document.getElementById("dP").textContent = m.p
    ? m.p.toFixed(2) + "B"
    : "\u2014";
  document.getElementById("dA").textContent = m.arch || "\u2014";
  document.getElementById("dL").textContent = m.L || "\u2014";
  document.getElementById("dKV").textContent = m.nkv || "\u2014";
  var complete = m.p > 0 && m.L > 0 && m.h > 0 && m.nh > 0 && m.nkv > 0;
  var s = m.src || "";
  if (
    complete &&
    (s.indexOf("safetensors") !== -1 || s.indexOf("API") !== -1)
  )
    conf.textContent =
      "\u2605\u2605\u2605 architecture-aware \u00b7 exact metadata";
  else if (complete && s.indexOf("config") !== -1)
    conf.textContent =
      "\u2605\u2605\u2606 architecture-aware \u00b7 config derived";
  else if (complete)
    conf.textContent =
      "\u2605\u2605\u2606 architecture-aware \u00b7 manually completed";
  else
    conf.textContent =
      "\u2605\u2606\u2606 rule-of-thumb \u00b7 missing architecture fields";
  var sel = document.getElementById("mSel");
  var exists = false;
  for (var i = 0; i < sel.options.length; i++) {
    if (sel.options[i].value === "_imported") {
      exists = true;
      break;
    }
  }
  if (!exists) {
    var o = new Option(m.name + " (imported)", "_imported");
    sel.insertBefore(o, sel.options[0]);
  } else {
    sel.options[0].text = m.name + " (imported)";
  }
  sel.selectedIndex = 0;
  calc();
}
