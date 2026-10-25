/** Dependency-free 3D layer scene, projected onto a high-DPI canvas.
 * Each sheet occupies a distinct Z coordinate. Drag changes the camera's
 * yaw/pitch; wheel and controls change camera distance. No semantic or causal
 * relationships are inferred from index-based neuron positions.
 */
export class LayerAtlas {
  constructor(canvas, onSelect) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.onSelect = onSelect;
    this.nodes = [];
    this.layers = [];
    this.widths = [];
    this.hits = [];
    this.drag = null;
    this.hover = null;
    this.reset();
    this.observer = new ResizeObserver(() => this.draw());
    this.observer.observe(canvas);
    canvas.addEventListener("pointerdown", (event) => {
      this.drag = {
        x: event.clientX,
        y: event.clientY,
        yaw: this.yaw,
        pitch: this.pitch,
        moved: false,
      };
      canvas.setPointerCapture(event.pointerId);
      canvas.style.cursor = "grabbing";
    });
    canvas.addEventListener("pointermove", (event) => {
      const rect = canvas.getBoundingClientRect();
      if (this.drag) {
        const dx = event.clientX - this.drag.x,
          dy = event.clientY - this.drag.y;
        if (Math.hypot(dx, dy) > 4) this.drag.moved = true;
        this.yaw = this.drag.yaw + dx * 0.007;
        this.pitch = Math.max(
          -1.05,
          Math.min(1.05, this.drag.pitch + dy * 0.006),
        );
        this.draw();
      } else {
        const hit = this.nearest(
          event.clientX - rect.left,
          event.clientY - rect.top,
        );
        this.hover = hit?.node || null;
        canvas.style.cursor = hit ? "pointer" : "grab";
        canvas.title = hit
          ? `L${hit.node.layer} · N${hit.node.neuron} — ${hit.node.title}`
          : "Drag to rotate · scroll to zoom · click a neuron";
        this.draw();
      }
    });
    canvas.addEventListener("pointerup", (event) => {
      if (this.drag && !this.drag.moved) {
        const rect = canvas.getBoundingClientRect();
        const hit = this.nearest(
          event.clientX - rect.left,
          event.clientY - rect.top,
        );
        if (hit) this.onSelect(hit.node);
      }
      this.drag = null;
      canvas.style.cursor = "grab";
    });
    canvas.addEventListener("pointercancel", () => {
      this.drag = null;
      canvas.style.cursor = "grab";
    });
    canvas.addEventListener("pointerleave", () => {
      if (!this.drag) {
        this.hover = null;
        this.draw();
      }
    });
    canvas.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        this.zoomBy(Math.exp(-event.deltaY * 0.001));
      },
      { passive: false },
    );
    canvas.addEventListener("keydown", (event) => {
      if (
        ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(
          event.key,
        ) &&
        this.nodes.length
      ) {
        event.preventDefault();
        const current = this.nodes.findIndex((node) =>
          this.matches(node, this.selected),
        );
        const next =
          (current +
            (["ArrowLeft", "ArrowUp"].includes(event.key) ? -1 : 1) +
            this.nodes.length) %
          this.nodes.length;
        this.onSelect(this.nodes[next]);
      } else if (event.key === "+" || event.key === "=") {
        event.preventDefault();
        this.zoomBy(1.15);
      } else if (event.key === "-") {
        event.preventDefault();
        this.zoomBy(1 / 1.15);
      } else if (event.key.toLowerCase() === "r") {
        event.preventDefault();
        this.reset();
      }
    });
  }
  matches(a, b) {
    return a && b && a.layer === b.layer && a.neuron === b.neuron;
  }
  reset() {
    this.yaw = 0.78;
    this.pitch = -0.24;
    this.zoom = 1;
    this.draw();
  }
  zoomBy(factor) {
    this.zoom = Math.max(0.55, Math.min(2.6, this.zoom * factor));
    this.draw();
  }
  setData(nodes, layers, widths, selected) {
    this.nodes = nodes;
    this.layers = layers;
    this.widths = widths;
    this.selected = selected;
    this.canvas.setAttribute(
      "aria-label",
      `Interactive 3D layer view. ${layers.length} sampled layers, ${nodes.length} neurons. ${selected ? `Selected layer ${selected.layer}, neuron ${selected.neuron}. ` : ""}Drag to rotate, scroll to zoom, arrow keys to select neurons, plus and minus to zoom, R to reset. List view is also available.`,
    );
    this.draw();
  }
  nearest(x, y) {
    return this.hits
      .map((hit) => ({ ...hit, distance: Math.hypot(hit.x - x, hit.y - y) }))
      .filter((hit) => hit.distance < 13)
      .sort((a, b) => a.distance - b.distance || a.depth - b.depth)[0];
  }
  transform([x, y, z]) {
    const cy = Math.cos(this.yaw),
      sy = Math.sin(this.yaw),
      cx = Math.cos(this.pitch),
      sx = Math.sin(this.pitch);
    const rx = x * cy + z * sy,
      rz = -x * sy + z * cy;
    return [rx, y * cx - rz * sx, y * sx + rz * cx];
  }
  draw() {
    const canvas = this.canvas,
      ctx = this.ctx;
    if (!ctx || !canvas.clientWidth || !canvas.clientHeight) return;
    const w = canvas.clientWidth,
      h = canvas.clientHeight,
      dpr = Math.min(devicePixelRatio || 1, 2);
    if (
      canvas.width !== Math.round(w * dpr) ||
      canvas.height !== Math.round(h * dpr)
    ) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const glow = ctx.createRadialGradient(
      w * 0.5,
      h * 0.5,
      10,
      w * 0.5,
      h * 0.5,
      w * 0.55,
    );
    glow.addColorStop(0, "#24304b");
    glow.addColorStop(0.65, "#182338");
    glow.addColorStop(1, "#131c2d");
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, w, h);
    if (!this.layers.length) return;
    const span = Math.max(1, this.layers.length - 1) * 31;
    const scale =
      Math.min((w - 60) / (span * 0.78 + 175), (h - 76) / (190 + span * 0.19)) *
      this.zoom;
    const project = (point) => {
      const [x, y, z] = this.transform(point);
      const perspective = 900 / (900 + z);
      return {
        x: w / 2 + x * scale * perspective,
        y: h / 2 + y * scale * perspective,
        depth: z,
        k: perspective,
      };
    };
    const line = (points, stroke, width = 1, fill = null) => {
      ctx.beginPath();
      points.forEach((point, i) => {
        const p = project(point);
        i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y);
      });
      if (fill) {
        ctx.closePath();
        ctx.fillStyle = fill;
        ctx.fill();
      }
      ctx.strokeStyle = stroke;
      ctx.lineWidth = width;
      ctx.stroke();
    };
    const sheets = this.layers
      .map((layer, index) => ({
        layer,
        z: (index - (this.layers.length - 1) / 2) * 31,
      }))
      .sort((a, b) => project([0, 0, b.z]).depth - project([0, 0, a.z]).depth);
    this.hits = [];
    for (const sheet of sheets) {
      const { layer, z } = sheet;
      const active = this.selected?.layer === layer;
      line(
        [
          [-83, -88, z],
          [83, -88, z],
          [83, 88, z],
          [-83, 88, z],
        ],
        active ? "#71acbb8a" : "#74a9ba3d",
        active ? 1.1 : 0.7,
        active ? "#6fb2d314" : "#74a9ba08",
      );
      for (let i = 1; i < 9; i++) {
        const x = -83 + (i * 166) / 9,
          y = -88 + (i * 176) / 9;
        line(
          [
            [x, -88, z],
            [x, 88, z],
          ],
          "#87adbf12",
          0.5,
        );
        line(
          [
            [-83, y, z],
            [83, y, z],
          ],
          "#87adbf12",
          0.5,
        );
      }
      ctx.fillStyle = "#b4dce03d";
      for (let x = 0; x < 11; x++)
        for (let y = 0; y < 13; y++) {
          const p = project([-74 + x * 14.8, -78 + y * 13, z]);
          ctx.beginPath();
          ctx.arc(p.x, p.y, Math.max(0.45, 0.62 * scale * p.k), 0, Math.PI * 2);
          ctx.fill();
        }
      const label = project([-83, -103, z]);
      ctx.font = "10px ui-monospace, Consolas, monospace";
      ctx.textAlign = "left";
      ctx.fillStyle = active ? "#dbd1f5" : "#95b2c5";
      ctx.fillText(`L${layer}`, label.x, label.y);
      for (const node of this.nodes.filter((node) => node.layer === layer)) {
        const width = this.widths[layer];
        const normalized = node.neuron / Math.max(width - 1, 1);
        const position = Math.floor(normalized * 399);
        const p = project([
          -74 + ((position % 20) * 148) / 19,
          -78 + (Math.floor(position / 20) * 156) / 19,
          z,
        ]);
        const selected = this.matches(node, this.selected),
          hovered = this.matches(node, this.hover);
        this.hits.push({ ...p, node });
        if (selected || hovered) {
          ctx.beginPath();
          ctx.arc(p.x, p.y, selected ? 10 : 7, 0, Math.PI * 2);
          ctx.fillStyle = selected ? "#c7b8f02e" : "#ffffff19";
          ctx.fill();
          ctx.strokeStyle = selected ? "#c6b0f2" : "#abcee18a";
          ctx.lineWidth = 1;
          ctx.stroke();
        }
        const radius = selected ? 4 : Math.max(1.5, 2.2 * scale * p.k);
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = selected
          ? "#f4eaff"
          : node.score === null
            ? "#aad0e0"
            : `hsla(${257 + node.score * 30},${45 + node.score * 20}%,${78 + node.score * 10}%,${0.5 + node.score * 0.5})`;
        if (selected) {
          ctx.shadowColor = "#c2a4fa";
          ctx.shadowBlur = 14;
        }
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }
    // Draw the selected marker last so it remains legible through the sheets.
    const selectedHit = this.hits.find((hit) =>
      this.matches(hit.node, this.selected),
    );
    if (selectedHit) {
      const { x, y, node } = selectedHit;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = "#f3eaff";
      ctx.shadowColor = "#c7a9ff";
      ctx.shadowBlur = 14;
      ctx.fill();
      ctx.shadowBlur = 0;
      const label = `L${node.layer} · N${node.neuron}`;
      ctx.font = "10px ui-monospace, Consolas, monospace";
      const tw = ctx.measureText(label).width;
      const lx = Math.min(w - tw - 15, Math.max(8, x + 14)),
        ly = Math.max(18, Math.min(h - 12, y - 14));
      ctx.fillStyle = "#172135e8";
      ctx.fillRect(lx - 6, ly - 12, tw + 12, 20);
      ctx.strokeStyle = "#8b7ca34d";
      ctx.lineWidth = 0.6;
      ctx.strokeRect(lx - 6, ly - 12, tw + 12, 20);
      ctx.fillStyle = "#ded3f3";
      ctx.textAlign = "left";
      ctx.fillText(label, lx, ly + 1);
    }
    ctx.font = '9px "Segoe UI", sans-serif';
    ctx.fillStyle = "#70849c";
    ctx.textAlign = "left";
    ctx.fillText("LAYER STACK  /  INDEX-BASED POSITIONS", 18, h - 16);
  }
}
