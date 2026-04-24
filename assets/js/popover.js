// ═══════════════════════════════════════════════════════════
// POPOVER SYSTEM (hover-based)
// ═══════════════════════════════════════════════════════════
var _popoverHideTimer = null;
var _popoverShowTimer = null;
var _popoverCurrentBtn = null;

function openPopover(key, btnEl) {
  var def = POPDEFS[key];
  if (!def) return;
  clearTimeout(_popoverHideTimer);
  clearTimeout(_popoverShowTimer);
  _popoverCurrentBtn = btnEl;

  document.querySelectorAll(".qbtn.active").forEach(function (b) {
    b.classList.remove("active");
  });
  btnEl.classList.add("active");

  var pop = document.getElementById("vc-popover");
  document.getElementById("vc-popTitle").textContent = def.title;
  var formulaEl = document.getElementById("vc-popFormula");
  var tex = STATIC_FORMULAS[key] || "";
  try {
    if (typeof katex !== "undefined") {
      katex.render(tex, formulaEl, { displayMode: true, throwOnError: false });
    } else {
      formulaEl.textContent = tex;
    }
  } catch (e) {
    formulaEl.textContent = tex;
  }
  document.getElementById("vc-popExplain").innerHTML = def.explain;
  document.getElementById("vc-popRefs").textContent = def.refs;
  var rect = btnEl.getBoundingClientRect();
  var popW = 320;
  var left = rect.left + 20;
  if (left + popW > window.innerWidth - 10) left = window.innerWidth - popW - 10;
  if (left < 10) left = 10;
  pop.style.left = "-9999px";
  pop.style.top = "0px";
  pop.className = "vc-popover vc-open";
  var popH = pop.offsetHeight;
  var top = rect.bottom + 8;
  if (top + popH + 10 > window.innerHeight) top = rect.top - popH - 8;
  if (top < 10) top = 10;
  pop.style.left = left + "px";
  pop.style.top = top + "px";
}

function scheduleHidePopover() {
  _popoverHideTimer = setTimeout(function () {
    closePopover();
  }, 250);
}

function cancelHidePopover() {
  clearTimeout(_popoverHideTimer);
}

function closePopover() {
  var pop = document.getElementById("vc-popover");
  if (pop) pop.className = "vc-popover";
  document.querySelectorAll(".qbtn.active").forEach(function (b) {
    b.classList.remove("active");
  });
  _popoverCurrentBtn = null;
}

document.addEventListener("keydown", function (e) {
  if (e.key === "Escape") closePopover();
});

(function () {
  var pop = document.getElementById("vc-popover");
  if (pop) {
    pop.addEventListener("mouseenter", function () { cancelHidePopover(); });
    pop.addEventListener("mouseleave", function () { scheduleHidePopover(); });
  }
})();

// ═══════════════════════════════════════════════════════════
// KATEX RENDER HELPER
// ═══════════════════════════════════════════════════════════
function renderKatex(element, tex) {
  if (typeof katex !== "undefined") {
    try {
      katex.render(tex, element, { displayMode: false, throwOnError: false });
    } catch (e) {
      element.textContent = tex;
    }
  } else {
    element.textContent = tex;
  }
}
