import { LayerAtlas } from "./atlas3d.js";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const escapeHTML = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        char
      ],
  );
const paths = {
  orbit:
    '<ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(-35 12 12)"/><ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(35 12 12)"/><circle cx="12" cy="12" r="2"/>',
  chat: '<path d="M21 11.5a8.5 8.5 0 0 1-8.5 8.5H4l-2 2V11.5A8.5 8.5 0 0 1 10.5 3h2a8.5 8.5 0 0 1 8.5 8.5Z"/><path d="M7 9h9M7 13h6"/>',
  flask:
    '<path d="M9 3h6M10 3v6l-6 10a1.4 1.4 0 0 0 1.2 2h13.6a1.4 1.4 0 0 0 1.2-2L14 9V3M8 14h8"/><path d="m10 17 1 .5M14 18h.1"/>',
  sliders:
    '<path d="M4 7h6m4 0h6M4 17h10m4 0h2"/><circle cx="12" cy="7" r="2"/><circle cx="16" cy="17" r="2"/>',
  shield:
    '<path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Z"/><path d="m9 12 2 2 4-4"/>',
  help: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 0 1 5 0c0 2-2.5 2-2.5 4m0 3h.01"/>',
  chevron: '<path d="m8 10 4 4 4-4"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10h.01"/>',
  reset: '<path d="M3 10a9 9 0 1 1 2 8M3 4v6h6"/>',
  bulb: '<path d="M9 18h6m-5 3h4M8.5 15.5a6 6 0 1 1 7 0c-.7.5-.8 1.5-.8 2.5H9.3c0-1-.1-2-.8-2.5Z"/>',
  play: '<path d="m8 4 12 8-12 8V4Z"/>',
  spark:
    '<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Z"/>',
  close: '<path d="m6 6 12 12M6 18 18 6"/>',
  download: '<path d="M12 3v12m-4-4 4 4 4-4M4 16v5h16v-5"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
};
const icon = (name) =>
  `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">${paths[name] || paths.spark}</svg>`;
$$("[data-icon]").forEach((element) => {
  element.outerHTML = icon(element.dataset.icon);
});

const state = {
  models: [],
  workspace: null,
  selected: null,
  view: "map",
  page: "explorer",
  experiments: [],
  busy: false,
  workspaceRequest: 0,
};
let toastTimer;

const sceneCanvas = document.createElement("canvas");
sceneCanvas.id = "layer-scene";
sceneCanvas.tabIndex = 0;
sceneCanvas.setAttribute("role", "group");
sceneCanvas.textContent =
  "Interactive 3D layer stack. Use List view to select neurons without the canvas.";
$("#atlas-stage").prepend(sceneCanvas);
const layerScene = new LayerAtlas(sceneCanvas, (node) => selectNeuron(node));
$("#map-view").textContent = "3D layers";
const cameraControls = document.createElement("div");
cameraControls.className = "camera-controls";
cameraControls.innerHTML =
  '<span>Drag to rotate · scroll to zoom</span><button id="zoom-out" type="button" aria-label="Zoom out of 3D layers">−</button><button id="zoom-in" type="button" aria-label="Zoom into 3D layers">＋</button>';
$("#atlas-stage").append(cameraControls);
$("#zoom-out").addEventListener("click", () => layerScene.zoomBy(1 / 1.2));
$("#zoom-in").addEventListener("click", () => layerScene.zoomBy(1.2));

// Keep the common atlas compact, while allowing direct access to every neuron.
const jumpTrigger = document.createElement("button");
jumpTrigger.id = "jump-trigger";
jumpTrigger.className = "text-button";
jumpTrigger.textContent = "Go to neuron ↗";
$(".atlas-card .card-header").insertBefore(jumpTrigger, $("#reset-atlas"));
jumpTrigger.addEventListener("click", () => {
  if (state.busy || !state.selected) return;
  $("#jump-layer").max = state.workspace.info.layers - 1;
  $("#jump-layer").value = state.selected.layer;
  $("#jump-neuron").value = state.selected.neuron;
  $("#jump-error").textContent = "";
  updateJumpBounds();
  $("#neuron-dialog").showModal();
});
function updateJumpBounds() {
  const layer = Number($("#jump-layer").value);
  const width = state.workspace?.info.widths[layer];
  $("#jump-neuron").max = width ? width - 1 : 0;
  $("#jump-bounds").textContent = width
    ? `Layer ${layer} has ${width.toLocaleString()} neurons (0–${width - 1}).`
    : "Choose a layer in this model.";
}
$("#jump-layer").addEventListener("input", updateJumpBounds);
$("#jump-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.busy) return;
  $("#jump-error").textContent = "";
  $("#jump-button").disabled = true;
  try {
    const node = await api("/api/neuron", {
      model_id: state.workspace.profile.id,
      layer: Number($("#jump-layer").value),
      neuron: Number($("#jump-neuron").value),
    });
    const modelId = state.workspace.profile.id;
    if (!state.workspace.atlas.layers.includes(node.layer)) {
      await loadWorkspace(modelId, node.layer);
    }
    if (
      !state.workspace.atlas.nodes.some(
        (item) => item.layer === node.layer && item.neuron === node.neuron,
      )
    ) {
      state.workspace.atlas.nodes.push(node);
    }
    $("#neuron-search").value = "";
    selectNeuron(node);
    $("#neuron-dialog").close();
  } catch (error) {
    $("#jump-error").textContent = error.message;
  } finally {
    $("#jump-button").disabled = false;
  }
});

async function api(path, body, method = body ? "POST" : "GET") {
  const response = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error(
      "The server returned an unreadable response. Check the server terminal.",
    );
  }
  if (!response.ok)
    throw new Error(payload.error || `Request failed (${response.status}).`);
  return payload;
}
function showError(error) {
  $("#global-error").textContent = error.message || String(error);
  $("#global-error").hidden = false;
}
function clearError() {
  $("#global-error").hidden = true;
}
function toast(message) {
  clearTimeout(toastTimer);
  $("#toast").textContent = message;
  $("#toast").hidden = false;
  toastTimer = setTimeout(() => {
    $("#toast").hidden = true;
  }, 4000);
}
function setBusy(value) {
  state.busy = value;
  $$(
    "#compare-button, #probe-button, #chat-button, #model-picker, #connect-model, #manage-models, #notice-connect, #layer-filter, #compare-prompt, #strength, #max-tokens, #probe-text, #chat-prompt, #jump-trigger, #reset-atlas",
  ).forEach((button) => {
    button.disabled = value;
  });
}
function setPage(page) {
  state.page = page;
  $$(".page").forEach((element) => {
    element.hidden = element.id !== `page-${page}`;
  });
  $$("[data-page]").forEach((button) =>
    button.classList.toggle("active", button.dataset.page === page),
  );
  $("#page-label").textContent = {
    explorer: "Explorer",
    playground: "Playground",
    experiments: "Experiments",
  }[page];
  if (page === "experiments") renderExperiments();
}
$$("[data-page]").forEach((button) =>
  button.addEventListener("click", () => setPage(button.dataset.page)),
);
$("#go-playground").addEventListener("click", () => setPage("playground"));

function openModels(add = false) {
  renderModels();
  $("#add-model-details").open = add;
  $("#model-dialog").showModal();
}
$("#model-picker").addEventListener("click", () => openModels());
$("#connect-model").addEventListener("click", () => openModels(true));
$("#manage-models").addEventListener("click", () => openModels());
$("#notice-connect").addEventListener("click", () => openModels(true));
["#open-guide", "#help-button"].forEach((selector) =>
  $(selector).addEventListener("click", () => $("#guide-dialog").showModal()),
);
$$(".dialog-close").forEach((button) =>
  button.addEventListener("click", () => button.closest("dialog").close()),
);
$$("dialog").forEach((dialog) =>
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) {
      const rect = dialog.getBoundingClientRect();
      if (
        event.clientX < rect.left ||
        event.clientX > rect.right ||
        event.clientY < rect.top ||
        event.clientY > rect.bottom
      )
        dialog.close();
    }
  }),
);

function renderModels() {
  $("#model-list").innerHTML = state.models
    .map(
      (profile) =>
        `<button class="model-option ${state.workspace?.profile.id === profile.id ? "current" : ""}" data-model="${escapeHTML(profile.id)}"><span class="option-avatar">${profile.provider === "demo" ? "✧" : profile.provider === "local" ? "H" : "↗"}</span><span class="option-name">${escapeHTML(profile.name)}<small class="option-detail">${escapeHTML(profile.provider === "demo" ? "Synthetic dataset · no download needed" : profile.model)} · ${profile.provider === "local" ? "Local weights" : profile.provider === "demo" ? "Demo" : "Chat API"}</small></span><span class="option-status">${state.workspace?.profile.id === profile.id ? "Active" : "Select →"}</span></button>`,
    )
    .join("");
  $$("#model-list [data-model]").forEach((button) =>
    button.addEventListener("click", async () => {
      $("#model-dialog").close();
      if (state.busy) return;
      await loadWorkspace(button.dataset.model);
    }),
  );
}
function updateProviderForm() {
  const provider = $("#provider-select").value;
  const http = provider !== "local";
  $("#http-fields").hidden = !http;
  $("#model-form [name=endpoint]").required = http;
  $("#model-form [name=endpoint]").placeholder =
    provider === "ollama"
      ? "http://localhost:11434"
      : "http://localhost:8000/v1";
  $("#model-form [name=model]").placeholder =
    provider === "local"
      ? "Qwen/Qwen3-0.6B"
      : provider === "ollama"
        ? "llama3.2:3b"
        : "Your served model ID";
  $("#provider-hint").textContent = http
    ? "Chat only: API endpoints do not expose internal neurons. Set the API key in your server environment; only its variable name is stored here."
    : "Requires the local inference extra. First use downloads model weights. Dense MLP architecture is detected automatically.";
}
$("#provider-select").addEventListener("change", updateProviderForm);
$("#model-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("#save-model");
  button.disabled = true;
  $("#model-form-error").textContent = "";
  const body = Object.fromEntries(new FormData(event.target));
  if (body.provider === "local") {
    body.endpoint = "";
    body.key_env = "";
  }
  try {
    const profile = await api("/api/models", body);
    state.models.push(profile);
    renderModels();
    event.target.reset();
    updateProviderForm();
    $("#add-model-details").open = false;
    toast("Connection added. Select it to open the model.");
  } catch (error) {
    $("#model-form-error").textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

async function loadWorkspace(modelId = "demo", layer = "") {
  const request = ++state.workspaceRequest;
  clearError();
  setBusy(true);
  $("#loading").hidden = false;
  if (modelId === "demo")
    $("#loading span:last-child").textContent = "Opening demo workspace…";
  else
    $("#loading span:last-child").textContent =
      "Opening model… First use of local weights may download the model and take several minutes.";
  try {
    const workspace = await api(
      `/api/workspace?model=${encodeURIComponent(modelId)}${layer !== "" ? `&layer=${encodeURIComponent(layer)}` : ""}`,
    );
    if (request !== state.workspaceRequest) return;
    const changed = state.workspace?.profile.id !== workspace.profile.id;
    state.workspace = workspace;
    const nodes = workspace.atlas.nodes;
    state.selected =
      nodes.find(
        (node) =>
          node.layer === state.selected?.layer &&
          node.neuron === state.selected?.neuron &&
          !changed,
      ) ||
      nodes.find((node) => node.layer === 8 && node.neuron === 2742) ||
      nodes[0] ||
      null;
    if (changed) {
      $("#comparison-results").hidden = true;
      $("#comparison-results").replaceChildren();
      $("#chat-output").replaceChildren();
      $("#neuron-search").value = "";
    }
    $("#active-model").textContent = workspace.profile.name;
    $("#active-model").title = [
      workspace.info.status,
      workspace.info.architecture,
    ]
      .filter(Boolean)
      .join(" · ");
    $(".model-avatar").textContent =
      workspace.profile.provider === "demo"
        ? "Q"
        : workspace.profile.provider === "local"
          ? "H"
          : "↗";
    const synthetic = workspace.info.source === "synthetic";
    $("#source-badge").textContent = synthetic
      ? "Demo mode"
      : workspace.profile.provider === "local"
        ? "Local model"
        : "Chat API";
    $("#source-badge").className = `badge ${synthetic ? "demo" : "live"}`;
    $("#chat-model-label").textContent = workspace.profile.name;
    $("#capabilities").innerHTML = ["chat", "inspect", "steer"]
      .map(
        (capability) =>
          `<span class="capability ${workspace.info.capabilities[capability] ? "" : "disabled"}" title="${workspace.info.capabilities[capability] ? "Available" : "Unavailable for this provider"}">${icon(workspace.info.capabilities[capability] ? "check" : "close")}${{ chat: "Chat", inspect: "Activations", steer: "Steering" }[capability]}</span>`,
      )
      .join("");
    $("#demo-notice").hidden = !synthetic;
    $("#inspection-workspace").hidden = !workspace.info.capabilities.inspect;
    $("#api-empty").hidden = workspace.info.capabilities.inspect;
    $("#layer-filter").innerHTML =
      '<option value="">All layers</option>' +
      Array.from(
        { length: workspace.info.layers },
        (_, i) => `<option value="${i}">Layer ${i}</option>`,
      ).join("");
    $("#layer-filter").value = String(layer);
    $("#atlas-caption").textContent =
      `${workspace.atlas.layers.length} of ${workspace.info.layers} layers · positions reflect neuron index`;
    $("#atlas-legend").textContent = synthetic
      ? "Illustrative activity"
      : "Sampled neurons";
    // The range legend applies only to the explicitly synthetic activity field.
    const footer = $(".atlas-footer>span");
    footer.innerHTML = synthetic
      ? '<i class="legend-dot lilac"></i><span id="atlas-legend">Illustrative activity</span><span class="legend-scale"></span>Low <span class="legend-high">High</span>'
      : '<i class="legend-dot lilac"></i><span id="atlas-legend">Index-based sample · inspect to measure</span>';
    renderAtlas();
    renderNeuron();
  } catch (error) {
    if (request === state.workspaceRequest) showError(error);
  } finally {
    if (request === state.workspaceRequest) {
      setBusy(false);
      $("#loading").hidden = true;
    }
  }
}

function filteredNodes() {
  const query = $("#neuron-search").value.trim().toLowerCase();
  return (state.workspace?.atlas.nodes || []).filter(
    (node) =>
      !query ||
      `${node.title} ${node.neuron} L${node.layer} N${node.neuron} ${node.tokens.join(" ")}`
        .toLowerCase()
        .includes(query),
  );
}
function renderAtlas() {
  if (!state.workspace) return;
  const nodes = filteredNodes();
  const layers = state.workspace.atlas.layers;
  $("#atlas-empty").hidden = nodes.length > 0;
  $("#atlas").style.display = "none";
  sceneCanvas.hidden = state.view !== "map" || !nodes.length;
  cameraControls.hidden = state.view !== "map" || !nodes.length;
  $("#atlas-stage").classList.toggle(
    "scene-active",
    state.view === "map" && nodes.length > 0,
  );
  $("#atlas-list").hidden = state.view !== "list" || !nodes.length;
  $("#map-view").classList.toggle("selected", state.view === "map");
  $("#list-view").classList.toggle("selected", state.view === "list");
  const selected = state.selected;
  const matches = (node) =>
    node.layer === selected?.layer && node.neuron === selected?.neuron;
  if (state.view === "list") {
    $("#atlas-list").innerHTML = nodes
      .map(
        (node, index) =>
          `<button class="${matches(node) ? "selected" : ""}" data-node="${index}"><span>L${node.layer} · N${node.neuron}</span><small>${escapeHTML(node.title)}</small></button>`,
      )
      .join("");
    $$("#atlas-list [data-node]").forEach((button) =>
      button.addEventListener("click", () =>
        selectNeuron(nodes[Number(button.dataset.node)]),
      ),
    );
    return;
  }
  layerScene.setData(nodes, layers, state.workspace.info.widths, selected);
}
function selectNeuron(node) {
  if (state.busy) return;
  state.selected = node;
  $("#comparison-results").hidden = true;
  renderAtlas();
  renderNeuron();
}
function renderNeuron() {
  const node = state.selected;
  if (!node) return;
  $("#neuron-id").textContent = `L${node.layer} · N${node.neuron}`;
  $("#control-neuron").textContent = `L${node.layer} · N${node.neuron}`;
  $("#neuron-title").textContent = node.title;
  $("#neuron-description").textContent = node.description;
  $("#neuron-source").textContent =
    node.source === "synthetic" ? "ILLUSTRATIVE" : "UNLABELED";
  $("#token-chips").innerHTML = node.tokens.length
    ? node.tokens.map((token) => `<span>${escapeHTML(token)}</span>`).join("")
    : '<span class="token-empty">No labels yet. Inspect this neuron on a few examples to find a pattern.</span>';
  $("#neuron-insight").textContent =
    node.source === "synthetic"
      ? "A label is a starting point. Test a few different prompts to see whether the pattern holds."
      : "Capture signed activations in the Activations tab. Higher magnitude does not establish what a neuron means.";
  $("#probe-result").replaceChildren();
}
$("#map-view").addEventListener("click", () => {
  state.view = "map";
  renderAtlas();
});
$("#list-view").addEventListener("click", () => {
  state.view = "list";
  renderAtlas();
});
$("#neuron-search").addEventListener("input", renderAtlas);
$("#layer-filter").addEventListener("change", () =>
  loadWorkspace(state.workspace.profile.id, $("#layer-filter").value),
);
$("#reset-atlas").addEventListener("click", () => {
  if (state.busy) return;
  $("#neuron-search").value = "";
  state.view = "map";
  layerScene.reset();
  loadWorkspace(state.workspace.profile.id);
});
function setDetailTab(activations) {
  $("#overview-tab").classList.toggle("active", !activations);
  $("#activation-tab").classList.toggle("active", activations);
  $("#overview-tab").setAttribute("aria-selected", String(!activations));
  $("#activation-tab").setAttribute("aria-selected", String(activations));
  $("#overview-panel").hidden = activations;
  $("#activation-panel").hidden = !activations;
}
$("#overview-tab").addEventListener("click", () => setDetailTab(false));
$("#activation-tab").addEventListener("click", () => setDetailTab(true));
$("#try-neuron").addEventListener("click", () => {
  $("#compare-prompt").focus();
  $("#compare-form").scrollIntoView({
    behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
      ? "instant"
      : "smooth",
    block: "center",
  });
});
$("#strength").addEventListener("input", () => {
  $("#strength-value").textContent =
    `${Number($("#strength").value).toFixed(1)}×`;
});

$("#probe-button").addEventListener("click", async () => {
  if (state.busy || !state.selected) return;
  const prompt = $("#probe-text").value.trim();
  if (!prompt) {
    toast("Enter some text to inspect.");
    return;
  }
  clearError();
  setBusy(true);
  const button = $("#probe-button");
  const old = button.innerHTML;
  button.textContent = "Measuring…";
  const node = { ...state.selected };
  try {
    const result = await api("/api/probe", {
      model_id: state.workspace.profile.id,
      layer: node.layer,
      neuron: node.neuron,
      prompt,
    });
    if (
      node.layer !== state.selected.layer ||
      node.neuron !== state.selected.neuron
    )
      return;
    const max = Math.max(
      ...result.tokens.map((token) => Math.abs(token.value)),
      0.001,
    );
    $("#probe-result").innerHTML =
      result.tokens
        .map(
          (token) =>
            `<mark style="background:hsla(${token.value >= 0 ? "272,45%,68%" : "191,43%,65%"},${0.06 + (Math.abs(token.value) / max) * 0.38})" title="Activation: ${token.value}">${escapeHTML(token.text)}</mark>`,
        )
        .join("") +
      `<span class="probe-legend">${result.source === "synthetic" ? "Synthetic activity" : "Raw MLP activations"} · purple: positive · teal: negative · hover for value</span>`;
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
    button.innerHTML = old;
  }
});

function resultHTML(result) {
  return `<div class="results-grid"><div class="response-card"><div class="response-heading"><i class="legend-dot" style="background:#c7bfce"></i><strong>Baseline</strong><span>· 1×</span></div><div class="response-text">${escapeHTML(result.baseline)}</div></div><div class="response-card steered"><div class="response-heading"><i class="legend-dot"></i><strong>Steered</strong><span>· ${Number(result.strength).toFixed(1)}×</span></div><div class="response-text">${escapeHTML(result.steered)}</div></div></div><p class="result-note">${escapeHTML(result.note || "")} ${result.source === "synthetic" ? "Demo · synthetic results." : `Live results · ${result.seconds}s.`}</p>`;
}
$("#compare-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.busy || !state.selected) return;
  const prompt = $("#compare-prompt").value.trim();
  if (!prompt) return;
  clearError();
  setBusy(true);
  $("#compare-button").innerHTML = '<span class="spinner"></span>Comparing…';
  $("#experiment-hint").innerHTML =
    icon("info") +
    "<span>Generating the baseline and steered response. Local CPU inference can take several minutes.</span>";
  try {
    const result = await api("/api/compare", {
      model_id: state.workspace.profile.id,
      layer: state.selected.layer,
      neuron: state.selected.neuron,
      prompt,
      strength: Number($("#strength").value),
      max_tokens: Number($("#max-tokens").value),
    });
    $("#comparison-results").innerHTML =
      `<div class="results-header"><span>L${result.layer} · N${result.neuron} · ${Number(result.strength).toFixed(1)}×</span><span class="saved-label">✓ Saved to Experiments</span></div>${resultHTML(result)}`;
    $("#comparison-results").hidden = false;
    await refreshExperiments();
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
    $("#compare-button").innerHTML = icon("play") + "Run comparison";
    $("#experiment-hint").innerHTML =
      icon("spark") +
      "<span>A small change can reveal an interesting pattern. Try a gain of 0× to suppress the neuron.</span>";
  }
});
$("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.busy || !state.workspace) return;
  const prompt = $("#chat-prompt").value.trim();
  if (!prompt) return;
  clearError();
  setBusy(true);
  $("#chat-button").innerHTML = '<span class="spinner"></span>Generating…';
  try {
    const result = await api("/api/chat", {
      model_id: state.workspace.profile.id,
      prompt,
      max_tokens: 160,
    });
    $("#chat-output").innerHTML =
      `<div class="response-heading">${escapeHTML(state.workspace.profile.name)} · ${result.source === "synthetic" ? "Synthetic demo" : "Live response"} · ${result.seconds}s</div><div class="response-text">${escapeHTML(result.text)}</div>`;
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
    $("#chat-button").innerHTML = "Generate response <span>→</span>";
  }
});

async function refreshExperiments() {
  const result = await api("/api/experiments");
  state.experiments = result.experiments;
  $("#experiment-count").textContent = state.experiments.length;
}
function exportJSON(data, name) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("#export-all").addEventListener("click", () =>
  exportJSON(state.experiments, "llm-explore-experiments.json"),
);
function renderExperiments() {
  $("#export-all").disabled = !state.experiments.length;
  if (!state.experiments.length) {
    $("#experiments-list").innerHTML =
      '<div class="card empty-experiments"><span>✧</span><h2>Your next discovery starts small.</h2><p>Run your first neuron comparison in Explorer.<br>The prompt, settings, and both responses will be saved here.</p><button class="primary" id="first-experiment">Explore a neuron →</button></div>';
    $("#first-experiment").addEventListener("click", () => setPage("explorer"));
    return;
  }
  $("#experiments-list").innerHTML = state.experiments
    .map(
      (result) =>
        `<article class="card experiment-entry"><div class="card-header"><div><h2>${escapeHTML(result.prompt)}</h2><div class="experiment-meta"><span>${escapeHTML(result.model_name)}</span><span>L${Number(result.layer)} · N${Number(result.neuron)} · ${Number(result.strength).toFixed(1)}×</span><span>${escapeHTML(new Date(result.created).toLocaleString())}</span><span class="badge ${result.source === "synthetic" ? "demo" : "live"}">${result.source === "synthetic" ? "Synthetic" : "Live"}</span></div></div><div class="entry-actions"><button data-export="${escapeHTML(result.id)}">Export</button><button data-delete="${escapeHTML(result.id)}" aria-label="Delete experiment ${escapeHTML(result.prompt)}">Delete</button></div></div>${resultHTML(result)}</article>`,
    )
    .join("");
  $$("#experiments-list [data-export]").forEach((button) =>
    button.addEventListener("click", () =>
      exportJSON(
        state.experiments.find((item) => item.id === button.dataset.export),
        `experiment-${button.dataset.export}.json`,
      ),
    ),
  );
  $$("#experiments-list [data-delete]").forEach((button) =>
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        await api(
          `/api/experiments/${encodeURIComponent(button.dataset.delete)}`,
          null,
          "DELETE",
        );
        await refreshExperiments();
        renderExperiments();
        toast("Experiment deleted.");
      } catch (error) {
        showError(error);
        button.disabled = false;
      }
    }),
  );
}

async function init() {
  try {
    const [models] = await Promise.all([
      api("/api/models"),
      refreshExperiments(),
    ]);
    state.models = models.models;
    await loadWorkspace();
  } catch (error) {
    showError(error);
  }
}
updateProviderForm();
init();
