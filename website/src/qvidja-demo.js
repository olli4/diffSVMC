import { WebR } from "https://webr.r-wasm.org/latest/webr.mjs";
import {
  buildJaxInputs,
  getDefaultWebsiteControls,
  sliceReferenceData,
} from "./qvidja-jax-core.js";

const ENGINE_META = {
  webr: {
    label: "WebR",
    className: "engine-webr",
  },
  jax: {
    label: "jax-js-nonconsuming",
    className: "engine-jax",
  },
};

const SUMMARY_FIELDS = [
  { key: "annualGpp", label: "Mean Annual GPP (g C/m²/yr)", digits: 1 },
  { key: "annualNee", label: "Mean Annual NEE (g C/m²/yr)", digits: 1 },
  { key: "finalSoc", label: "Final SOC (kg/m²)", digits: 2 },
  { key: "peakLai", label: "Peak LAI", digits: 2 },
  { key: "peakCleaf", label: "Peak C_leaf (g/m²)", digits: 2 },
];

const CHART_DEFS = [
  {
    title: "Fluxes",
    subtitle: "Daily carbon fluxes over the replay window.",
    yLabel: "g C m^-2 d^-1",
    buildDatasets(result) {
      return [
        lineDataset("GPP", scaleSeries(result.daily.gpp, 86400 * 1000), "#198754"),
        lineDataset("NEE", scaleSeries(result.daily.nee, 86400 * 1000), "#dc3545"),
      ];
    },
  },
  {
    title: "Respiration",
    subtitle: "Autotrophic and heterotrophic respiration.",
    yLabel: "g C m^-2 d^-1",
    buildDatasets(result) {
      return [
        lineDataset("Heterotrophic", scaleSeries(result.daily.heteroResp, 86400 * 1000), "#fd7e14"),
        lineDataset("Autotrophic", scaleSeries(result.daily.autoResp, 86400 * 1000), "#6f42c1"),
      ];
    },
  },
  {
    title: "Carbon Pools",
    subtitle: "Daily vegetation carbon pools.",
    yLabel: "g C m^-2",
    buildDatasets(result) {
      return [
        lineDataset("C_leaf", scaleSeries(result.daily.cleaf, 1000), "#198754"),
        lineDataset("C_root", scaleSeries(result.daily.croot, 1000), "#0d6efd"),
        lineDataset("C_stem", scaleSeries(result.daily.cstem, 1000), "#fd7e14"),
      ];
    },
  },
  {
    title: "Soil Carbon and LAI",
    subtitle: "Shared daily state exposed by both engines.",
    yLabel: "kg C m^-2",
    y2Label: "LAI",
    buildDatasets(result) {
      return [
        lineDataset("SOC", result.daily.socTotal, "#795548"),
        lineDataset("LAI alloc", result.daily.laiAlloc, "#20c997", "y2"),
      ];
    },
  },
];

const JAX_DEBUG_PREFIX = "[svmc-demo:jax]";
/** Base timeout per day (ms): ~600ms/day for WASM JIT + generous margin. */
const JAX_WORKER_TIMEOUT_PER_DAY_MS = 800;
const JAX_WORKER_TIMEOUT_MIN_MS = 30_000;
const JAX_WORKER_TIMEOUT_MAX_MS = 300_000;

function jaxWorkerTimeout(ndays) {
  return Math.min(
    JAX_WORKER_TIMEOUT_MAX_MS,
    Math.max(JAX_WORKER_TIMEOUT_MIN_MS, ndays * JAX_WORKER_TIMEOUT_PER_DAY_MS),
  );
}

const status = document.getElementById("status");
const runBtn = document.getElementById("run-btn");
const referencePeriod = document.getElementById("reference-period");
const referenceSource = document.getElementById("reference-source");
const sabCheck = document.getElementById("sab-check");
const debugLog = document.getElementById("debug-log");
const summaryPanels = document.getElementById("summary-panels");
const chartRows = document.getElementById("chart-rows");
const placeholder = document.getElementById("placeholder");

ENGINE_META.webr.checkbox = document.getElementById("engine-webr");
ENGINE_META.webr.timing = document.getElementById("engine-webr-timing");
ENGINE_META.jax.checkbox = document.getElementById("engine-jax");
ENGINE_META.jax.timing = document.getElementById("engine-jax-timing");
ENGINE_META.jax.settings = document.getElementById("engine-jax-settings");

const jaxBackendSelect = document.getElementById("engine-jax-backend");
const jaxJitCheckbox = document.getElementById("engine-jax-jit");

const charts = [];
let referenceData = null;
let referenceReady = false;
let webRInstance = null;
let webRInitPromise = null;
const debugLogLines = [];

window.addEventListener("error", (event) => {
  debugJax("window error", {
    message: event.message,
    filename: event.filename,
    lineno: event.lineno,
    colno: event.colno,
  }, "error");
});

window.addEventListener("unhandledrejection", (event) => {
  debugJax("unhandled rejection", {
    reason: stringifyDebugValue(event.reason),
  }, "error");
});

debugJax("module loaded");

updateSharedArrayBufferStatus();
bindControlEvents();
void initializeDemo();

async function initializeDemo() {
  try {
    setStatus("loading", "Loading Qvidja reference inputs...");
    referenceData = await loadReferenceData();
    applyReferenceDefaults(referenceData);
    referenceReady = true;
    setStatus("ok", "Ready - reference inputs loaded. Select one or both engines and run the replay.");
  } catch (error) {
    setStatus("err", `Failed to initialize demo: ${toErrorMessage(error)}`);
    console.error(error);
  }
  updateRunButtonState();
}

function bindControlEvents() {
  for (const engine of Object.values(ENGINE_META)) {
    engine.checkbox.addEventListener("change", updateRunButtonState);
  }
  document.getElementById("obs_lai").addEventListener("change", updateRunButtonState);
  jaxBackendSelect.addEventListener("change", updateRunButtonState);
  jaxJitCheckbox.addEventListener("change", updateRunButtonState);
}

function updateSharedArrayBufferStatus() {
  if (typeof SharedArrayBuffer !== "undefined") {
    sabCheck.textContent = "SharedArrayBuffer available - WebR can run in this page.";
    sabCheck.style.color = "var(--ok)";
    return;
  }

  sabCheck.textContent = "SharedArrayBuffer unavailable - WebR is disabled on this page.";
  sabCheck.style.color = "#dc3545";
  ENGINE_META.webr.checkbox.checked = false;
  ENGINE_META.webr.checkbox.disabled = true;
  setEngineTiming("webr", "unavailable", "error");
}

function updateRunButtonState() {
  const selected = getSelectedEngines();
  const unsupported = getUnsupportedEngineSelections();
  ENGINE_META.jax.settings.classList.toggle("is-disabled", !ENGINE_META.jax.checkbox.checked);
  jaxBackendSelect.disabled = !ENGINE_META.jax.checkbox.checked;
  jaxJitCheckbox.disabled = !ENGINE_META.jax.checkbox.checked;
  runBtn.disabled = !referenceReady || selected.length === 0 || unsupported.length > 0;
}

function getSelectedEngines() {
  return Object.entries(ENGINE_META)
    .filter(([, meta]) => meta.checkbox.checked && !meta.checkbox.disabled)
    .map(([engine]) => engine);
}

function getUnsupportedEngineSelections() {
  const issues = [];
  if (ENGINE_META.jax.checkbox.checked && val("obs_lai") === 0) {
    issues.push("jax-js-nonconsuming currently supports only the observed-LAI replay path.");
  }
  return issues;
}

runBtn.addEventListener("click", async () => {
  debugJax("run button clicked", {
    selectedEngines: getSelectedEngines(),
    jaxBackend: jaxBackendSelect.value,
    jaxUseJit: jaxJitCheckbox.checked,
    requestedDays: val("ndays"),
    obsLai: val("obs_lai"),
  });

  const unsupported = getUnsupportedEngineSelections();
  if (unsupported.length > 0) {
    debugJax("run blocked by unsupported selections", { unsupported });
    setStatus("err", unsupported.join(" "));
    updateRunButtonState();
    return;
  }

  const selectedEngines = getSelectedEngines();
  if (selectedEngines.length === 0) {
    setStatus("err", "Select at least one engine before running the demo.");
    updateRunButtonState();
    return;
  }

  runBtn.disabled = true;
  destroyCharts();
  summaryPanels.innerHTML = "";
  chartRows.innerHTML = "";
  summaryPanels.style.display = "none";
  chartRows.style.display = "none";
  placeholder.style.display = "";

  const requestedDays = val("ndays");
  const slice = sliceReferenceData(referenceData, requestedDays);
  document.getElementById("ndays").value = String(slice.ndays);

  const results = new Map();
  const failures = [];

  for (const engine of selectedEngines) {
    debugJax("starting selected engine", { engine });
    setEngineTiming(engine, "running...", "running");

    try {
      const result = engine === "webr"
        ? await runWebRSimulation(slice)
        : await runJaxSimulation(slice);
      results.set(engine, result);
      setEngineTiming(engine, formatRunLabel(result), "done");
      debugJax("engine completed", {
        engine,
        elapsedMs: result.elapsedMs,
        executionLabel: result.executionLabel,
      });
    } catch (error) {
      const message = toErrorMessage(error);
      failures.push(`${ENGINE_META[engine].label}: ${message}`);
      setEngineTiming(engine, "failed", "error");
      debugJax("engine failed", { engine, message, error });
      console.error(error);
    }
  }

  if (results.size > 0) {
    renderResults(slice, results);
    const timingSummary = Array.from(results.entries())
      .map(([engine, result]) => `${ENGINE_META[engine].label} ${formatTiming(result.elapsedMs)}`)
      .join("; ");

    if (failures.length > 0) {
      setStatus(
        "err",
        `Rendered ${results.size}/${selectedEngines.length} engines for ${slice.ndays} days (${slice.daily.dates[0]} to ${slice.daily.dates.at(-1)}). ${timingSummary}. ${failures.join(" ")}`,
      );
    } else {
      setStatus(
        "ok",
        `Rendered ${results.size} engine${results.size === 1 ? "" : "s"} for ${slice.ndays} days (${slice.daily.dates[0]} to ${slice.daily.dates.at(-1)}). ${timingSummary}.`,
      );
    }
  } else {
    setStatus("err", failures.join(" ") || "No engine completed successfully.");
  }

  updateRunButtonState();
});

function setStatus(kind, message) {
  status.className = kind;
  status.textContent = message;
}

function setEngineTiming(engine, text, state) {
  ENGINE_META[engine].timing.textContent = text;
  ENGINE_META[engine].timing.dataset.state = state;
}

function formatRunLabel(result) {
  if (result.executionLabel) {
    return `${formatTiming(result.elapsedMs)} · ${result.executionLabel}`;
  }
  return formatTiming(result.elapsedMs);
}

function formatTiming(elapsedMs) {
  if (!Number.isFinite(elapsedMs)) return "n/a";
  if (elapsedMs >= 1000) return `${(elapsedMs / 1000).toFixed(2)} s`;
  return `${elapsedMs.toFixed(0)} ms`;
}

function val(id) {
  const element = document.getElementById(id);
  return element.tagName === "SELECT" ? parseInt(element.value, 10) : parseFloat(element.value);
}

function repoUrl() {
  const loc = window.location;
  if (loc.hostname === "localhost" || loc.hostname === "127.0.0.1") {
    return `${loc.origin}/`;
  }
  return loc.origin + loc.pathname.replace(/\/?$/, "/") + "repo/";
}

async function loadReferenceData() {
  const response = await fetch(new URL("./qvidja-v1-reference.json", window.location.href));
  if (!response.ok) {
    throw new Error(`Failed to load Qvidja reference data (${response.status})`);
  }
  return await response.json();
}

function applyReferenceDefaults(reference) {
  const { defaults, site } = reference;
  document.getElementById("ndays").max = String(site.ndays);
  document.getElementById("ndays").value = String(site.ndays);
  document.getElementById("pft_type_code").value = String(defaults.pft_type_code);
  document.getElementById("obs_lai").value = defaults.obs_lai ? "1" : "0";

  document.getElementById("conductivity").value = String(defaults.conductivity);
  document.getElementById("psi50").value = String(defaults.psi50);
  document.getElementById("b_param").value = String(defaults.b);
  document.getElementById("alpha").value = String(defaults.alpha);
  document.getElementById("gamma").value = String(defaults.gamma);
  document.getElementById("rdark").value = String(defaults.rdark);

  document.getElementById("soil_depth").value = String(defaults.soil_depth);
  document.getElementById("max_poros").value = String(defaults.max_poros);
  document.getElementById("fc").value = String(defaults.fc);
  document.getElementById("wp").value = String(defaults.wp);
  document.getElementById("ksat").value = String(defaults.ksat);

  document.getElementById("cratio_resp").value = String(defaults.cratio_resp);
  document.getElementById("cratio_leaf").value = String(defaults.cratio_leaf);
  document.getElementById("cratio_root").value = String(defaults.cratio_root);
  document.getElementById("cratio_biomass").value = String(defaults.cratio_biomass);
  document.getElementById("harvest_index").value = String(defaults.harvest_index);
  document.getElementById("turnover_cleaf").value = String(defaults.turnover_cleaf);
  document.getElementById("turnover_croot").value = String(defaults.turnover_croot);
  document.getElementById("sla").value = String(defaults.sla);
  document.getElementById("q10").value = String(defaults.q10);

  document.getElementById("yasso_totc").value = String(defaults.yasso_totc);
  document.getElementById("yasso_cn_input").value = String(defaults.yasso_cn_input);
  document.getElementById("yasso_fract_root").value = String(defaults.yasso_fract_root);
  document.getElementById("yasso_fract_legacy").value = String(defaults.yasso_fract_legacy);

  referencePeriod.textContent = `Reference replay window: ${site.start_date} to ${site.end_date_exclusive} (${site.ndays} days, ${site.nhours} hourly steps)`;
  referenceSource.textContent = "Source: vendored Qvidja NetCDF inputs under vendor/SVMC/data/input, matching the maintained SVMC v1.0.0 test configuration.";
}

async function ensureWebRReady() {
  if (webRInstance) return webRInstance;
  if (webRInitPromise) return webRInitPromise;

  webRInitPromise = (async () => {
    setStatus("loading", "Initializing WebR runtime...");
    const instance = new WebR();
    await instance.init();
    setStatus("loading", "Installing SVMCwebr into WebR...");
    await instance.installPackages(["SVMCwebr"], {
      repos: [repoUrl(), "https://repo.r-wasm.org/"],
    });
    await instance.evalRVoid("library(SVMCwebr)");
    webRInstance = instance;
    return instance;
  })().catch((error) => {
    webRInitPromise = null;
    throw error;
  });

  return webRInitPromise;
}

async function runWebRSimulation(slice) {
  const instance = await ensureWebRReady();
  const startedAt = performance.now();

  setStatus("loading", `Binding ${slice.ndays}-day replay inputs into WebR...`);
  const env = instance.objs.globalEnv;
  await bindRVector(env, instance, ".__temp_hr", slice.hourly.temp_hr);
  await bindRVector(env, instance, ".__rg_hr", slice.hourly.rg_hr);
  await bindRVector(env, instance, ".__prec_hr", slice.hourly.prec_hr);
  await bindRVector(env, instance, ".__vpd_hr", slice.hourly.vpd_hr);
  await bindRVector(env, instance, ".__pres_hr", slice.hourly.pres_hr);
  await bindRVector(env, instance, ".__co2_hr", slice.hourly.co2_hr);
  await bindRVector(env, instance, ".__wind_hr", slice.hourly.wind_hr);
  await bindRVector(env, instance, ".__lai_day", slice.daily.lai_day);
  await bindRVector(env, instance, ".__snow_day", slice.daily.snowdepth_day);
  await bindRVector(env, instance, ".__sm_day", slice.daily.soilmoist_day);
  await bindRVector(env, instance, ".__manage_type", slice.daily.manage_type);
  await bindRVector(env, instance, ".__manage_c_in", slice.daily.manage_c_in);
  await bindRVector(env, instance, ".__manage_c_out", slice.daily.manage_c_out);

  setStatus("loading", `Running WebR reference simulation for ${slice.ndays} days...`);
  const rObject = await instance.evalR(buildWebRRunCode(slice));

  try {
    setStatus("loading", "Extracting WebR daily diagnostics...");
    const [gppDay, neeDay, heteroResp, autoResp, cleaf, croot, cstem, soilC, laiAlloc] = await Promise.all([
      getRVector(rObject, "gpp_day"),
      getRVector(rObject, "nee_day"),
      getRVector(rObject, "hetero_resp"),
      getRVector(rObject, "auto_resp"),
      getRVector(rObject, "cleaf"),
      getRVector(rObject, "croot"),
      getRVector(rObject, "cstem"),
      getRVector(rObject, "soil_c"),
      getRVector(rObject, "lai_alloc"),
    ]);

    return {
      elapsedMs: performance.now() - startedAt,
      executionLabel: "webr",
      daily: {
        gpp: gppDay,
        nee: neeDay,
        heteroResp,
        autoResp,
        cleaf,
        croot,
        cstem,
        socTotal: soilC,
        laiAlloc,
      },
      summary: buildSummary(slice, {
        gpp: gppDay,
        nee: neeDay,
        cleaf,
        socTotal: soilC,
        laiAlloc,
      }),
    };
  } finally {
    rObject.destroy?.();
  }
}

async function runJaxSimulation(slice) {
  const startedAt = performance.now();
  const device = jaxBackendSelect.value;
  const useJit = jaxJitCheckbox.checked;

  debugJax("runJaxSimulation start", {
    device,
    useJit,
    ndays: slice.ndays,
    nhours: slice.nhours,
    dailyPoints: slice.daily.dates.length,
  });

  debugJax("before buildJaxInputs", {
    ndays: slice.ndays,
    firstDate: slice.daily.dates[0],
    lastDate: slice.daily.dates.at(-1),
  });
  const inputLike = buildJaxInputs(slice, collectJaxControls());
  debugJax("after buildJaxInputs", summarizeInputLike(inputLike));

  setStatus(
    "loading",
    `Running jax-js-nonconsuming integration in a worker for ${slice.ndays} days on ${device}${useJit ? " with JIT" : " without JIT"}...`,
  );
  const timeoutMs = jaxWorkerTimeout(slice.ndays);
  await flushDebugUi("before worker invocation");
  debugJax("before worker invocation", {
    mode: useJit ? "jit" : "eager",
    device,
    timeoutMs,
  });

  const workerResult = await runJaxSimulationInWorker({
    device,
    useJit,
    inputLike,
  }, timeoutMs);

  debugJax("after worker invocation", {
    elapsedMs: workerResult.elapsedMs,
    executionLabel: workerResult.executionLabel,
    gppLength: workerResult.daily.gpp.length,
    lastSoc: workerResult.daily.socTotal.at(-1),
  });

  return {
    elapsedMs: performance.now() - startedAt,
    executionLabel: workerResult.executionLabel,
    daily: workerResult.daily,
    summary: buildSummary(slice, workerResult.daily),
  };
}

function runJaxSimulationInWorker(payload, timeoutMs) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./qvidja-jax-worker.js", import.meta.url), {
      type: "module",
    });
    let settled = false;

    const cleanup = () => {
      clearTimeout(timeoutId);
      worker.removeEventListener("message", onMessage);
      worker.removeEventListener("error", onError);
      worker.terminate();
    };

    const finish = (callback) => {
      if (settled) return;
      settled = true;
      cleanup();
      callback();
    };

    const onMessage = (event) => {
      const message = event.data;
      if (!message || typeof message !== "object") return;

      if (message.type === "debug") {
        debugJax(message.message, message.details, message.level);
        return;
      }

      if (message.type === "result") {
        finish(() => resolve(message.result));
        return;
      }

      if (message.type === "error") {
        finish(() => reject(new Error(message.message)));
      }
    };

    const onError = (event) => {
      finish(() => reject(new Error(event.message || "jax-js-nonconsuming worker failed.")));
    };

    const timeoutId = setTimeout(() => {
      debugJax("worker timeout", {
        timeoutMs,
        device: payload.device,
        mode: payload.useJit ? "jit" : "eager",
      }, "error");
      finish(() => reject(new Error(`jax-js-nonconsuming worker timed out after ${Math.round(timeoutMs / 1000)} s.`)));
    }, timeoutMs);

    worker.addEventListener("message", onMessage);
    worker.addEventListener("error", onError);
    worker.postMessage({ type: "run", payload });
  });
}

function collectJaxControls() {
  return {
    ...getDefaultWebsiteControls(referenceData.defaults),
    pftTypeCode: val("pft_type_code"),
    conductivity: val("conductivity"),
    psi50: val("psi50"),
    bParam: val("b_param"),
    alpha: val("alpha"),
    gamma: val("gamma"),
    rdark: val("rdark"),
    soilDepth: val("soil_depth"),
    maxPoros: val("max_poros"),
    fc: val("fc"),
    wp: val("wp"),
    ksat: val("ksat"),
    cratioResp: val("cratio_resp"),
    cratioLeaf: val("cratio_leaf"),
    cratioRoot: val("cratio_root"),
    cratioBiomass: val("cratio_biomass"),
    harvestIndex: val("harvest_index"),
    turnoverCleaf: val("turnover_cleaf"),
    turnoverCroot: val("turnover_croot"),
    sla: val("sla"),
    q10: val("q10"),
    yassoTotc: val("yasso_totc"),
    yassoCnInput: val("yasso_cn_input"),
    yassoFractRoot: val("yasso_fract_root"),
    yassoFractLegacy: val("yasso_fract_legacy"),
  };
}

function buildSummary(slice, daily) {
  const years = slice.ndays / 365.25;
  return {
    annualGpp: sum(daily.gpp) * 86400 * 1000 / years,
    annualNee: sum(daily.nee) * 86400 * 1000 / years,
    finalSoc: daily.socTotal.at(-1) ?? NaN,
    peakLai: val("obs_lai") === 1 ? maxOf(slice.daily.lai_day) : maxOf(daily.laiAlloc),
    peakCleaf: maxOf(daily.cleaf) * 1000,
  };
}

function renderResults(slice, results) {
  placeholder.style.display = "none";
  summaryPanels.style.display = "grid";
  chartRows.style.display = "grid";

  for (const [engine, result] of results.entries()) {
    summaryPanels.appendChild(createSummaryPanel(engine, result));
  }

  const labels = decimateLabels(slice.daily.dates);
  for (const def of CHART_DEFS) {
    const row = document.createElement("section");
    row.className = "chart-row";
    const head = document.createElement("div");
    head.className = "chart-row-head";
    const title = document.createElement("h3");
    title.textContent = def.title;
    const subtitle = document.createElement("span");
    subtitle.textContent = def.subtitle;
    head.append(title, subtitle);
    row.appendChild(head);

    const grid = document.createElement("div");
    grid.className = "chart-row-grid";
    grid.style.gridTemplateColumns = `repeat(${results.size}, minmax(0, 1fr))`;

    for (const [engine, result] of results.entries()) {
      const box = document.createElement("div");
      box.className = `chart-box ${ENGINE_META[engine].className}`;

      const boxHead = document.createElement("div");
      boxHead.className = "chart-box-head";
      const label = document.createElement("h4");
      label.textContent = ENGINE_META[engine].label;
      const timing = document.createElement("span");
      timing.textContent = result.executionLabel
        ? `${formatTiming(result.elapsedMs)} · ${result.executionLabel}`
        : formatTiming(result.elapsedMs);
      boxHead.append(label, timing);
      box.appendChild(boxHead);

      const canvas = document.createElement("canvas");
      box.appendChild(canvas);
      grid.appendChild(box);

      const chart = new Chart(canvas, {
        type: "line",
        data: {
          labels,
          datasets: def.buildDatasets(decimateDailyResult(result)),
        },
        options: buildChartOptions(def.yLabel, def.y2Label),
      });
      charts.push(chart);
    }

    row.appendChild(grid);
    chartRows.appendChild(row);
  }
}

function createSummaryPanel(engine, result) {
  const panel = document.createElement("section");
  panel.className = `engine-panel ${ENGINE_META[engine].className}`;

  const head = document.createElement("div");
  head.className = "engine-panel-head";
  const title = document.createElement("h3");
  title.textContent = ENGINE_META[engine].label;
  const timing = document.createElement("span");
  timing.textContent = result.executionLabel
    ? `Run time ${formatTiming(result.elapsedMs)} · ${result.executionLabel}`
    : `Run time ${formatTiming(result.elapsedMs)}`;
  head.append(title, timing);
  panel.appendChild(head);

  const grid = document.createElement("div");
  grid.className = "engine-summary-grid";

  for (const field of SUMMARY_FIELDS) {
    const stat = document.createElement("div");
    stat.className = "stat";

    const value = document.createElement("div");
    value.className = "val";
    value.textContent = formatNumber(result.summary[field.key], field.digits);

    const label = document.createElement("div");
    label.className = "lbl";
    label.textContent = field.label;

    stat.append(value, label);
    grid.appendChild(stat);
  }

  panel.appendChild(grid);
  return panel;
}

function buildChartOptions(yLabel, y2Label) {
  const scales = {
    x: {
      ticks: { maxTicksLimit: 10, maxRotation: 0 },
    },
    y: {
      title: { display: true, text: yLabel },
      position: "left",
    },
  };

  if (y2Label) {
    scales.y2 = {
      title: { display: true, text: y2Label },
      position: "right",
      grid: { drawOnChartArea: false },
    };
  }

  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    scales,
    plugins: {
      legend: {
        labels: {
          boxWidth: 12,
          font: { size: 11 },
        },
      },
    },
  };
}

function lineDataset(label, data, borderColor, yAxisID = "y") {
  return {
    label,
    data,
    borderColor,
    borderWidth: 1.5,
    pointRadius: 0,
    tension: 0.1,
    yAxisID,
  };
}

function decimateDailyResult(result) {
  return {
    daily: {
      gpp: decimateSeries(result.daily.gpp),
      nee: decimateSeries(result.daily.nee),
      heteroResp: decimateSeries(result.daily.heteroResp),
      autoResp: decimateSeries(result.daily.autoResp),
      cleaf: decimateSeries(result.daily.cleaf),
      croot: decimateSeries(result.daily.croot),
      cstem: decimateSeries(result.daily.cstem),
      socTotal: decimateSeries(result.daily.socTotal),
      laiAlloc: decimateSeries(result.daily.laiAlloc),
    },
  };
}

function decimateLabels(labels) {
  const step = dailyDecimationStep(labels.length);
  return labels.filter((_, index) => index % step === 0);
}

function decimateSeries(values) {
  const step = dailyDecimationStep(values.length);
  return values.filter((_, index) => index % step === 0);
}

function dailyDecimationStep(length) {
  const maxPoints = 500;
  return Math.max(1, Math.floor(length / maxPoints));
}

function scaleSeries(values, factor) {
  return values.map((value) => value * factor);
}

function chunkHourly(values, ndays) {
  return Array.from({ length: ndays }, (_, day) => values.slice(day * 24, (day + 1) * 24));
}

async function bindRVector(env, instance, name, values) {
  const vector = await new instance.RDouble(Array.from(values));
  await env.bind(name, vector);
}

function buildWebRRunCode(slice) {
  return `
    result <- svmc_run(
      nhours = ${slice.nhours}L, ndays = ${slice.ndays}L,
      temp_hr = .__temp_hr,
      rg_hr = .__rg_hr,
      prec_hr = .__prec_hr,
      vpd_hr = .__vpd_hr,
      pres_hr = .__pres_hr,
      co2_hr = .__co2_hr,
      wind_hr = .__wind_hr,
      lai_day = .__lai_day,
      snowdepth_day = .__snow_day,
      soilmoist_day = .__sm_day,
      obs_lai = ${val("obs_lai") === 1 ? "TRUE" : "FALSE"},
      obs_soilmoist = FALSE,
      obs_snowdepth = FALSE,
      conductivity = ${val("conductivity")},
      psi50 = ${val("psi50")},
      b = ${val("b_param")},
      alpha = ${val("alpha")},
      gamma = ${val("gamma")},
      rdark = ${val("rdark")},
      pft_type_code = ${val("pft_type_code")}L,
      opt_hypothesis = "${slice.defaults.opt_hypothesis}",
      soil_depth = ${val("soil_depth")},
      max_poros = ${val("max_poros")},
      fc = ${val("fc")},
      wp = ${val("wp")},
      ksat = ${val("ksat")},
      cratio_resp = ${val("cratio_resp")},
      cratio_leaf = ${val("cratio_leaf")},
      cratio_root = ${val("cratio_root")},
      cratio_biomass = ${val("cratio_biomass")},
      harvest_index = ${val("harvest_index")},
      turnover_cleaf = ${val("turnover_cleaf")},
      turnover_croot = ${val("turnover_croot")},
      sla = ${val("sla")},
      q10 = ${val("q10")},
      yasso_totc = ${val("yasso_totc")},
      yasso_cn_input = ${val("yasso_cn_input")},
      yasso_fract_root = ${val("yasso_fract_root")},
      yasso_fract_legacy = ${val("yasso_fract_legacy")},
      manage_type = .__manage_type,
      manage_c_in = .__manage_c_in,
      manage_c_out = .__manage_c_out
    )
    rm(.__temp_hr, .__rg_hr, .__prec_hr, .__vpd_hr, .__pres_hr,
       .__co2_hr, .__wind_hr, .__lai_day, .__snow_day, .__sm_day,
       .__manage_type, .__manage_c_in, .__manage_c_out)
    result
  `;
}

async function getRVector(rObject, name) {
  const vector = await rObject.get(name);
  try {
    return Array.from(await vector.toArray());
  } finally {
    vector.destroy?.();
  }
}

function destroyCharts() {
  while (charts.length > 0) {
    charts.pop().destroy();
  }
}

function yieldToBrowser() {
  return new Promise((resolve) => {
    requestAnimationFrame(() => resolve());
  });
}

function formatNumber(value, digits) {
  return Number.isFinite(value) ? value.toFixed(digits) : "-";
}

function sum(values) {
  return values.reduce((total, value) => total + value, 0);
}

function maxOf(values) {
  return values.reduce((maxValue, value) => Math.max(maxValue, value), -Infinity);
}

function toErrorMessage(error) {
  if (error instanceof Error) return error.message;
  return stringifyDebugValue(error);
}

function summarizeInputLike(inputLike) {
  return {
    hourlyDays: inputLike.hourly_temp.length,
    hourlyStepsPerDay: inputLike.hourly_temp[0]?.length ?? 0,
    dailyLaiLength: inputLike.daily_lai.length,
    manageTypeLength: inputLike.daily_manage_type.length,
    firstHourlyTemp: inputLike.hourly_temp[0]?.[0],
    firstDailyLai: inputLike.daily_lai[0],
    conductivity: inputLike.conductivity,
    psi50: inputLike.psi50,
    invertOption: inputLike.invert_option,
  };
}

async function flushDebugUi(reason) {
  debugJax("flush UI", { reason });
  await new Promise((resolve) => {
    requestAnimationFrame(() => {
      setTimeout(resolve, 0);
    });
  });
}

function debugJax(message, details, level = "log") {
  const timestamp = new Date().toISOString().slice(11, 23);
  const entry = details === undefined
    ? `${timestamp} ${JAX_DEBUG_PREFIX} ${message}`
    : `${timestamp} ${JAX_DEBUG_PREFIX} ${message} ${stringifyDebugValue(details)}`;

  debugLogLines.push(entry);
  if (debugLogLines.length > 250) {
    debugLogLines.splice(0, debugLogLines.length - 250);
  }
  if (debugLog) {
    debugLog.textContent = debugLogLines.join("\n");
    debugLog.scrollTop = debugLog.scrollHeight;
  }

  if (level === "error") {
    console.error(entry);
    return;
  }
  console.log(entry);
}

function stringifyDebugValue(value) {
  const seen = new WeakSet();

  const replacer = (_key, innerValue) => {
    if (typeof innerValue === "object" && innerValue !== null) {
      if (seen.has(innerValue)) return "[circular]";
      seen.add(innerValue);

      if (
        Array.isArray(innerValue.shape)
        && typeof innerValue.dtype === "string"
        && typeof innerValue.js === "function"
      ) {
        return {
          kind: "np.Array",
          dtype: innerValue.dtype,
          shape: innerValue.shape,
        };
      }
    }
    return innerValue;
  };

  if (value instanceof Error) {
    return JSON.stringify({
      name: value.name,
      message: value.message,
      stack: value.stack,
    });
  }

  try {
    return JSON.stringify(value, replacer);
  } catch {
    try {
      return JSON.stringify({ fallback: Object.prototype.toString.call(value) });
    } catch {
      return "[unserializable]";
    }
  }
}
