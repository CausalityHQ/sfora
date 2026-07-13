// Interactivity for the "How it separates" panels:
//  • hover / tap a dot -> floating image tooltip at the cursor + the side
//    inspector, and the *same* dot (same image index) lights up in every panel;
//  • click a legend species -> isolate it in every panel;
//  • ⤢ button -> enlarge one space to explore its dots up close.
// Points align by index k across panels because make_projection samples the
// same rows for every model (see scripts/make_projection.py `indices`).

const section = document.getElementById("separates");
if (section) {
  const base = section.dataset.base || "";
  const thumbs = JSON.parse(section.dataset.thumbs || "[]");
  const thumbBase = section.dataset.thumbbase || "";
  const names = JSON.parse(section.dataset.names || "[]");
  const src = (k) => `${base}/${thumbBase}/${thumbs[k]}`;

  const layout = section.querySelector(".proj-layout");
  const grid = section.querySelector(".proj-grid");
  const img = document.getElementById("insp-img");
  const nameEl = document.getElementById("insp-name");
  const subEl = document.getElementById("insp-sub");
  const tip = document.getElementById("proj-tip");
  const tipImg = document.getElementById("tip-img");
  const tipName = document.getElementById("tip-name");

  let active = [];
  const clearHi = () => {
    for (const c of active) c.classList.remove("hi");
    active = [];
  };

  const placeTip = (e) => {
    const pad = 16;
    const w = tip.offsetWidth || 150;
    const h = tip.offsetHeight || 170;
    let x = e.clientX + pad;
    let y = e.clientY + pad;
    if (x + w > window.innerWidth) x = e.clientX - w - pad;
    if (y + h > window.innerHeight) y = e.clientY - h - pad;
    tip.style.transform = `translate(${x}px, ${y}px)`;
  };

  const show = (k, c, e) => {
    clearHi();
    active = Array.from(grid.querySelectorAll(`circle[data-k="${k}"]`));
    for (const el of active) el.classList.add("hi");
    layout.classList.add("inspecting");
    if (img) {
      img.src = src(k);
      img.alt = names[c] || "bird";
      img.classList.add("shown");
    }
    if (nameEl) nameEl.textContent = names[c] || "";
    if (subEl) subEl.textContent = "the same photo, lit up in all five spaces";
    // floating tooltip
    tipImg.src = src(k);
    tipName.textContent = names[c] || "";
    tip.classList.add("on");
    if (e) placeTip(e);
  };

  const hideTip = () => tip.classList.remove("on");

  const dotFrom = (e) => {
    const t = e.target;
    return t && t.tagName === "circle" && t.dataset.k !== undefined ? t : null;
  };

  grid.addEventListener("pointerover", (e) => {
    const d = dotFrom(e);
    if (d) show(+d.dataset.k, +d.dataset.c, e);
  });
  grid.addEventListener("pointermove", (e) => {
    if (tip.classList.contains("on")) placeTip(e);
  });
  grid.addEventListener("click", (e) => {
    const d = dotFrom(e);
    if (d) show(+d.dataset.k, +d.dataset.c, e);
  });
  grid.addEventListener("pointerleave", () => {
    clearHi();
    layout.classList.remove("inspecting");
    hideTip();
  });

  // Legend: click a species to isolate it in every panel (toggle).
  let isolated = null;
  for (const chip of section.querySelectorAll(".leg-chip")) {
    chip.addEventListener("click", () => {
      const c = +chip.dataset.c;
      const already = isolated === c;
      grid.classList.remove(...Array.from({ length: 8 }, (_, i) => `iso-${i}`));
      for (const b of section.querySelectorAll(".leg-chip")) b.classList.remove("on");
      if (already) {
        isolated = null;
      } else {
        isolated = c;
        grid.classList.add(`iso-${c}`);
        chip.classList.add("on");
      }
    });
  }

  // ⤢ Enlarge one space (toggle). Others collapse so it can breathe.
  for (const btn of section.querySelectorAll(".proj-expand")) {
    btn.addEventListener("click", () => {
      const card = btn.closest(".proj-card");
      const wasOpen = card.classList.contains("enlarged");
      for (const c of grid.querySelectorAll(".proj-card")) c.classList.remove("enlarged");
      grid.classList.toggle("has-enlarged", !wasOpen);
      if (!wasOpen) card.classList.add("enlarged");
      hideTip();
    });
  }
}
