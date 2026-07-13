// SFORA hero — a live 3D point cloud of a HERD model's learned embedding space.
import * as THREE from "three";

// Earthy, cohesive palette (no neon) — matches the site's ember/moss identity.
const PALETTE = [
  0xc2542a, 0xd99a3a, 0x5f7d55, 0x3f7d73, 0xa86b4c, 0x7d5568, 0xb89b3e, 0x5a6b86,
];

function roundSprite() {
  const s = 64;
  const c = document.createElement("canvas");
  c.width = c.height = s;
  const g = c.getContext("2d");
  const grd = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  grd.addColorStop(0, "rgba(255,255,255,1)");
  grd.addColorStop(0.4, "rgba(255,255,255,1)");
  grd.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grd;
  g.beginPath();
  g.arc(s / 2, s / 2, s / 2, 0, Math.PI * 2);
  g.fill();
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// A great circle of radius R in one of three orthogonal planes.
function circle(R, plane, color) {
  const pts = [];
  for (let i = 0; i <= 96; i++) {
    const a = (i / 96) * Math.PI * 2;
    const x = Math.cos(a) * R;
    const y = Math.sin(a) * R;
    if (plane === 0) pts.push(new THREE.Vector3(x, y, 0));
    else if (plane === 1) pts.push(new THREE.Vector3(x, 0, y));
    else pts.push(new THREE.Vector3(0, x, y));
  }
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  return new THREE.Line(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.2 }));
}

function boot(canvas, data) {
  const points = Array.isArray(data.points) ? data.points : [];
  if (!points.length) return;

  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
  camera.position.set(0, 0, 3.2);

  const world = new THREE.Group();
  scene.add(world);
  const R = 1.32;

  // Context: a faint great-circle cage suggesting the unit hypersphere.
  const cage = new THREE.Group();
  cage.add(circle(R, 0, 0x9a8f7d));
  cage.add(circle(R, 1, 0x9a8f7d));
  cage.add(circle(R, 2, 0x9a8f7d));
  world.add(cage);

  // The embedding point cloud (real data), colored by class, on the sphere shell.
  const N = points.length;
  const positions = new Float32Array(N * 3);
  const colors = new Float32Array(N * 3);
  const col = new THREE.Color();
  const v = new THREE.Vector3();
  for (let i = 0; i < N; i++) {
    const p = points[i];
    v.set(p.x || 0, p.y || 0, p.z || 0);
    if (v.lengthSq() < 1e-6) v.set(0, 0, 1);
    v.normalize().multiplyScalar(R);
    positions[i * 3] = v.x;
    positions[i * 3 + 1] = v.y;
    positions[i * 3 + 2] = v.z;
    col.setHex(PALETTE[(p.c || 0) % PALETTE.length]);
    colors[i * 3] = col.r;
    colors[i * 3 + 1] = col.g;
    colors[i * 3 + 2] = col.b;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const cloud = new THREE.Points(
    geo,
    new THREE.PointsMaterial({
      size: 0.12,
      map: roundSprite(),
      vertexColors: true,
      transparent: true,
      alphaTest: 0.2,
      sizeAttenuation: true,
      depthWrite: false,
    }),
  );
  world.add(cloud);
  world.rotation.x = -0.35;

  // Interaction: drag to rotate, gentle auto-spin when idle.
  let dragging = false;
  let px = 0;
  let py = 0;
  const spin = reduced ? 0 : 0.0024;
  const getX = (e) => (e.touches ? e.touches[0].clientX : e.clientX);
  const getY = (e) => (e.touches ? e.touches[0].clientY : e.clientY);
  canvas.addEventListener("pointerdown", (e) => { dragging = true; px = getX(e); py = getY(e); });
  window.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const x = getX(e);
    const y = getY(e);
    world.rotation.y += (x - px) * 0.006;
    world.rotation.x += (y - py) * 0.006;
    px = x;
    py = y;
  });
  window.addEventListener("pointerup", () => { dragging = false; });

  const resize = () => {
    const s = Math.max(1, canvas.clientWidth || 400);
    renderer.setSize(s, s, false);
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(canvas);
  resize();

  let running = true;
  const frame = () => {
    if (!running) return;
    if (!dragging) world.rotation.y += spin;
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  };
  frame();

  new IntersectionObserver((entries) => {
    for (const en of entries) {
      const was = running;
      running = en.isIntersecting;
      if (running && !was) frame();
    }
  }).observe(canvas);
}

// Run AFTER all module-level consts are initialized (avoids a TDZ race on PALETTE
// once the bundler hoists declarations).
const canvas = document.getElementById("sphere");
const dataEl = document.getElementById("embedding-data");
if (canvas && dataEl) {
  try {
    boot(canvas, JSON.parse(dataEl.textContent || "{}"));
  } catch (err) {
    console.warn("sphere viz failed to start", err);
  }
}
