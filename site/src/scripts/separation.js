// Interactivity for the "How it separates" panels:
//  • hover / tap a dot -> show that bird's photo and light up the *same* dot
//    (same image index) across all panels;
//  • click a legend species -> isolate it in every panel.
// Points align by index k across panels because make_projection samples the
// same rows for every model (see scripts/make_projection.py `indices`).

const section = document.getElementById("separates");
if (section) {
  const base = section.dataset.base || "";
  const thumbs = JSON.parse(section.dataset.thumbs || "[]");
  const thumbBase = section.dataset.thumbbase || "";
  const names = JSON.parse(section.dataset.names || "[]");

  const layout = section.querySelector(".proj-layout");
  const grid = section.querySelector(".proj-grid");
  const img = document.getElementById("insp-img");
  const nameEl = document.getElementById("insp-name");
  const subEl = document.getElementById("insp-sub");

  let active = []; // circles currently highlighted

  const clearHi = () => {
    for (const c of active) c.classList.remove("hi");
    active = [];
  };

  const show = (k, c) => {
    clearHi();
    active = Array.from(grid.querySelectorAll(`circle[data-k="${k}"]`));
    for (const el of active) el.classList.add("hi");
    layout.classList.add("inspecting");
    if (img) {
      img.src = `${base}/${thumbBase}/${thumbs[k]}`;
      img.alt = names[c] || "bird";
      img.classList.add("shown");
    }
    if (nameEl) nameEl.textContent = names[c] || "";
    if (subEl) subEl.textContent = `the same photo, lit up in all five spaces`;
  };

  // Pointer (mouse + touch) over a dot.
  grid.addEventListener("pointerover", (e) => {
    const t = e.target;
    if (t && t.tagName === "circle" && t.dataset.k !== undefined) {
      show(+t.dataset.k, +t.dataset.c);
    }
  });
  // Tap on touch devices fires click; keep the last selection sticky otherwise.
  grid.addEventListener("click", (e) => {
    const t = e.target;
    if (t && t.tagName === "circle" && t.dataset.k !== undefined) {
      show(+t.dataset.k, +t.dataset.c);
    }
  });
  grid.addEventListener("pointerleave", () => {
    clearHi();
    layout.classList.remove("inspecting");
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
}
