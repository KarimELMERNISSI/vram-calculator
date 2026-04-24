// ═══════════════════════════════════════════════════════════
// GLOBAL STATE
// ═══════════════════════════════════════════════════════════
var gImp = null;
var gRaw = null;
var gVariants = [];
var gSelectedVar = -1;
var gSearchTimer = null;
var gBranch = "main";

// ═══════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════
function ss(from, to, comma) {
  var v = parseFloat(document.getElementById(from).value);
  document.getElementById(to).textContent = comma ? v.toLocaleString() : v;
}
function fmt(gb) {
  if (gb >= 1000) return (gb / 1000).toFixed(1) + " TB";
  return gb.toFixed(1) + " GB";
}
function fmtMoney(v) {
  if (v >= 1000) return "$" + v.toFixed(0);
  if (v >= 1) return "$" + v.toFixed(2);
  if (v >= 0.01) return "$" + v.toFixed(3);
  return "$" + v.toFixed(4);
}
function setMode(m) {
  document.getElementById("ftSel").style.display = m === "ft" ? "block" : "none";
  document.getElementById("uSel").style.display = m === "ft" ? "none" : "block";
  document.querySelectorAll(".tab").forEach(function (b, i) {
    b.classList.toggle("on", m === "ft" ? i === 1 : i === 0);
  });
  calc();
}
function getModel() {
  var sel = document.getElementById("mSel");
  if (sel.value === "_imported" && gImp) return gImp;
  return MODELS.find(function (m) { return m.name === sel.value; }) || MODELS[0];
}
function onMC() {
  var m = getModel();
  document.getElementById("dP").textContent = m.p ? (+m.p).toFixed(2) + "B" : "\u2014";
  document.getElementById("dA").textContent = m.arch || "\u2014";
  document.getElementById("dL").textContent = m.L || "\u2014";
  document.getElementById("dKV").textContent = m.nkv || "\u2014";
  calc();
}

// ═══════════════════════════════════════════════════════════
// GPU CHANGE HANDLER
// ═══════════════════════════════════════════════════════════
function onGpuChange() {
  var gName = document.getElementById("gSel").value;
  var customRow = document.getElementById("customGpuRow");
  if (gName === "Custom GPU") {
    customRow.className = "custom-gpu-row open";
  } else {
    customRow.className = "custom-gpu-row";
  }
  var gpu = GPUS.find(function (g) { return g.name === gName; });
  var icSel = document.getElementById("interconnType");
  if (icSel) {
    if (gpu && gpu.nvswitch) icSel.value = "nvswitch";
    else if (gpu && gpu.nvlink > 0) icSel.value = "nvlink";
    else icSel.value = "pipeline";
  }
  calc();
}

function toggleRamOff() {
  var checked = document.getElementById("ramOff").checked;
  var ramRow = document.getElementById("ramRow");
  var ramWarn = document.getElementById("ramWarn");
  ramRow.className = checked ? "ram-row open" : "ram-row";
  ramWarn.style.display = checked ? "block" : "none";
}

function updateTdpFromGpu() {
  var gName = document.getElementById("gSel").value;
  if (gName !== "Custom GPU") {
    var gpu = GPUS.find(function (g) { return g.name === gName; });
    if (gpu && gpu.tdp) document.getElementById("pTdp").value = gpu.tdp;
  }
}

// ═══════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════
(function () {
  // Move popover to body
  var pop = document.getElementById("vc-popover");
  if (pop && pop.parentNode !== document.body) {
    document.body.appendChild(pop);
  }

  var mSel = document.getElementById("mSel");
  var gSel = document.getElementById("gSel");
  MODELS.forEach(function (m) { mSel.appendChild(new Option(m.name, m.name)); });
  GPUS.forEach(function (g) {
    var label = g.name === "Custom GPU" ? "Custom GPU" : g.name + " (" + g.vram + "GB)";
    gSel.appendChild(new Option(label, g.name));
  });

  document.getElementById("hfId").value = "Qwen/Qwen3.6-35B-A3B";
  mSel.value = "Qwen 3.6 35B A3B";
  gSel.value = GPUS[0].name;

  updateTdpFromGpu();
  onMC();

  // Wire up "?" popover buttons
  document.querySelectorAll(".qbtn").forEach(function (btn) {
    btn.addEventListener("mouseenter", function (e) {
      cancelHidePopover();
      var key = btn.getAttribute("data-pop");
      _popoverShowTimer = setTimeout(function () { openPopover(key, btn); }, 80);
    });
    btn.addEventListener("mouseleave", function (e) {
      clearTimeout(_popoverShowTimer);
      scheduleHidePopover();
    });
    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      e.preventDefault();
      var key = btn.getAttribute("data-pop");
      openPopover(key, btn);
    });
    btn.addEventListener("mousedown", function (e) { e.preventDefault(); });
  });
})();

// Override onGpuChange to also update TDP
var _origOnGpuChange = onGpuChange;
onGpuChange = function () {
  _origOnGpuChange();
  updateTdpFromGpu();
};
