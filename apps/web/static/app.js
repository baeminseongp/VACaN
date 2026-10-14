const app = {
  data: null,
  frame: 0,
  layer: "density",
  robot: null,
  zone: null,
  playing: false,
  timer: null,
  caviarZone: null,
  caviarObservation: 0,
  caviarCamera: 0,
  demoAtcZone: null,
  mediaAvailability: {},
};

const $ = (id) => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";
const stateNames = { F: "FIXED_COVERED", R: "ROBOT_RECOVERED", S: "STALE", U: "UNKNOWN" };
const levelNames = { L: "LOW", M: "MEDIUM", H: "HIGH", U: "UNKNOWN" };
const colors = { L: "#183b4d", M: "#bc8d2d", H: "#e75b58", U: "#39434e", F: "#52d6c9", R: "#a28bff", S: "#f2bd62" };

function node(tag, attrs = {}) {
  const element = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

function clock(timestamp) {
  return new Intl.DateTimeFormat("en-GB", {
    timeZone: "Asia/Tokyo", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  }).format(new Date(timestamp));
}

function frame() { return app.data.frames[app.frame]; }
function clamp(value, low, high) { return Math.max(low, Math.min(high, value)); }
function mapValue(value, a, b, c, d) { return a === b ? (c + d) / 2 : c + ((value - a) / (b - a)) * (d - c); }
function planningKey(timestamp, robot) { return `${timestamp}|${robot}`; }

function nearestPlanning() {
  const target = Date.parse(frame().t);
  return app.data.planningTimestamps.reduce((best, current) =>
    Math.abs(Date.parse(current) - target) < Math.abs(Date.parse(best) - target) ? current : best
  );
}

function routes() {
  return app.data.planning[planningKey(nearestPlanning(), app.robot)];
}

function projector() {
  const points = app.data.zones.flatMap((zone) => zone.polygon);
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const bounds = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  return (point) => [mapValue(point[0], bounds[0], bounds[1], 36, 864), mapValue(point[1], bounds[2], bounds[3], 534, 36)];
}

function renderMap() {
  const map = $("mapSvg");
  map.replaceChildren();
  map.setAttribute("viewBox", "0 0 900 570");
  map.setAttribute("aria-label", "ATC zone congestion and robot paths");
  const project = projector();
  const current = frame();

  app.data.zones.forEach((zone, index) => {
    const observation = current.o[index];
    const code = app.layer === "density" ? observation[1] : observation[2];
    const polygon = node("polygon", {
      points: zone.polygon.map(project).map((point) => point.join(",")).join(" "),
      fill: colors[code] || colors.U,
      class: `zone${zone.id === app.zone ? " selected" : ""}`,
      tabindex: "0",
      "aria-label": `${zone.id}: ${app.layer === "density" ? levelNames[code] : stateNames[code]}`,
    });
    const select = () => { app.zone = zone.id; renderMap(); renderInspector(); };
    polygon.addEventListener("click", select);
    polygon.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") select(); });
    map.append(polygon);
  });

  const currentRoutes = routes();
  const drawPath = (path, className) => {
    if (!path?.zones?.length) return;
    const points = path.zones.map((id) => app.data.zones.find((zone) => zone.id === id)?.centroid).filter(Boolean).map(project);
    map.append(node("polyline", { points: points.map((point) => point.join(",")).join(" "), class: `path ${className}` }));
    points.forEach((point, index) => map.append(node("circle", { cx: point[0], cy: point[1], r: index ? 2.6 : 5, fill: index ? "#edf3f5" : "#5bd5c5", class: "path-point" })));
  };
  if ($("baselineToggle").checked) drawPath(currentRoutes?.BASELINE, "baseline");
  if ($("awareToggle").checked) drawPath(currentRoutes?.CONGESTION_AWARE, "aware");

  const legend = app.layer === "density"
    ? [["Low", colors.L], ["Medium", colors.M], ["High", colors.H], ["Unknown", colors.U]]
    : [["Fixed", colors.F], ["Recovered", colors.R], ["Stale", colors.S], ["Unknown", colors.U]];
  $("mapLegend").innerHTML = legend.map(([label, color]) => `<span style="--c:${color}">${label}</span>`).join("");
}

function renderMetrics() {
  const current = frame();
  const [fixed, recovered, stale, unknown, total] = current.c;
  const currentRoutes = routes();
  const baseline = currentRoutes?.BASELINE;
  const aware = currentRoutes?.CONGESTION_AWARE;
  $("clockLabel").textContent = `${clock(current.t)} JST · ${app.frame + 1}/${app.data.frames.length}`;
  $("coverageMetric").textContent = `${(((fixed + recovered) / total) * 100).toFixed(1)}%`;
  $("coverageDelta").textContent = `+${recovered.toFixed(0)} m² mobile recovery`;
  $("unknownMetric").textContent = `${((unknown / total) * 100).toFixed(1)}%`;
  $("etaMetric").textContent = baseline && aware ? `${aware.actualEta - baseline.actualEta >= 0 ? "+" : ""}${(aware.actualEta - baseline.actualEta).toFixed(2)} s` : "—";
  $("exposureMetric").textContent = baseline && aware ? `${aware.exposure - baseline.exposure >= 0 ? "+" : ""}${(aware.exposure - baseline.exposure).toFixed(3)}` : "—";
  $("timeSlider").value = app.frame;
  $("playButton").textContent = app.playing ? "Ⅱ" : "▶";
}

function renderInspector() {
  if (!app.zone) return;
  const index = app.data.zoneIds.indexOf(app.zone);
  const zone = app.data.zones[index];
  const truth = frame().g[index];
  const [density, level, coverage, confidence, age, source] = frame().o[index];
  $("zoneEmpty").hidden = true;
  $("zoneDetails").hidden = false;
  $("zoneName").textContent = zone.id;
  $("zoneSwatch").style.background = colors[app.layer === "density" ? level : coverage] || colors.U;
  $("zoneDensity").textContent = density == null ? "not observed" : `${density.toFixed(3)} person/m²`;
  $("zoneLevel").textContent = levelNames[level];
  $("zoneState").textContent = stateNames[coverage];
  $("zoneAge").textContent = age == null ? "—" : `${age.toFixed(0)} s`;
  $("zoneConfidence").textContent = confidence == null ? "—" : confidence.toFixed(2);
  $("zoneSource").textContent = source || "—";
  $("observedText").textContent = density == null
    ? "This zone has no fresh observation at the selected minute."
    : `The dashboard exposes the ${stateNames[coverage].toLowerCase().replaceAll("_", " ")} estimate, not individual trajectories.`;
  $("inferenceText").textContent = `Held-out density proxy: ${truth.toFixed(3)} person/m². Relative levels use within-day empirical quantiles and are not universal safety thresholds.`;
  renderDemoCctv(zone.id);
}

function renderEvents() {
  const time = Date.parse(frame().t);
  const nearby = app.data.events
    .map((event) => ({ ...event, gap: Math.abs(Date.parse(event.t) - time) }))
    .filter((event) => event.gap <= 15 * 60 * 1000)
    .sort((a, b) => a.gap - b.gap)
    .slice(0, 8);
  $("eventCount").textContent = `${nearby.length}`;
  $("eventList").replaceChildren(...nearby.map((event) => {
    const button = document.createElement("button");
    button.className = "event-item";
    button.innerHTML = `<b>${event.robot} · ${event.type}</b><span>${clock(event.t)} · ${event.zone || "route"} · ΔETA ${event.etaDelta.toFixed(2)} s</span>`;
    button.addEventListener("click", () => {
      app.robot = event.robot;
      app.zone = event.zone || event.newPath[0];
      app.frame = app.data.frames.reduce((best, item, index) => Math.abs(Date.parse(item.t) - Date.parse(event.t)) < Math.abs(Date.parse(app.data.frames[best].t) - Date.parse(event.t)) ? index : best, 0);
      $("robotSelect").value = app.robot;
      render();
    });
    return button;
  }));
}

function renderTimeline() {
  const chart = $("timelineSvg");
  chart.replaceChildren();
  chart.setAttribute("viewBox", "0 0 900 180");
  const plot = { left: 56, right: 876, top: 16, bottom: 150 };
  const means = app.data.frames.map((item) => item.d[0]);
  const p90s = app.data.frames.map((item) => item.d[1]);
  const thresholds = app.data.meta.thresholds;
  const yMax = Math.max(...p90s, thresholds[1]) * 1.08;
  const xAt = (index) => mapValue(index, 0, app.data.frames.length - 1, plot.left, plot.right);
  const yAt = (value) => mapValue(value, 0, yMax, plot.bottom, plot.top);
  const label = (text, x, y, attrs = {}) => {
    const element = node("text", { x, y, fill: "#85939c", "font-size": "9", "font-family": "ui-monospace, monospace", ...attrs });
    element.textContent = text;
    chart.append(element);
  };

  const bands = [
    [0, thresholds[0], "#183b4d", "LOW"],
    [thresholds[0], thresholds[1], "#bc8d2d", "MEDIUM"],
    [thresholds[1], yMax, "#e75b58", "HIGH"],
  ];
  bands.forEach(([low, high, color, name]) => {
    chart.append(node("rect", { x: plot.left, y: yAt(high), width: plot.right - plot.left, height: yAt(low) - yAt(high), fill: color, opacity: ".09" }));
    label(name, plot.right - 5, yAt(high) + 11, { "text-anchor": "end", fill: color });
  });
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = (yMax / 4) * tick;
    const y = yAt(value);
    chart.append(node("line", { x1: plot.left, x2: plot.right, y1: y, y2: y, stroke: "#283239", "stroke-width": "1" }));
    label(value.toFixed(3), plot.left - 8, y + 3, { "text-anchor": "end" });
  }
  label("person/m²", 8, 12);

  app.data.events.forEach((event) => {
    const start = Date.parse(app.data.frames[0].t);
    const end = Date.parse(app.data.frames.at(-1).t);
    const x = mapValue(Date.parse(event.t), start, end, plot.left, plot.right);
    chart.append(node("line", { x1: x, x2: x, y1: plot.top, y2: plot.bottom, stroke: "#ff6f61", "stroke-width": ".55", opacity: ".18" }));
  });
  const meanArea = `${xAt(0)},${plot.bottom} ${means.map((value, index) => `${xAt(index)},${yAt(value)}`).join(" ")} ${xAt(means.length - 1)},${plot.bottom}`;
  chart.append(node("polygon", { points: meanArea, fill: "#5bd5c5", opacity: ".12" }));
  chart.append(node("polyline", { points: means.map((value, index) => `${xAt(index)},${yAt(value)}`).join(" "), fill: "none", stroke: "#5bd5c5", "stroke-width": "2" }));
  chart.append(node("polyline", { points: p90s.map((value, index) => `${xAt(index)},${yAt(value)}`).join(" "), fill: "none", stroke: "#ffe083", "stroke-width": "2.2" }));
  const x = xAt(app.frame);
  chart.append(node("line", { x1: x, x2: x, y1: plot.top - 4, y2: plot.bottom + 4, stroke: "#fff", "stroke-width": "1.5" }));
  chart.append(node("circle", { cx: x, cy: yAt(means[app.frame]), r: "4", fill: "#5bd5c5", stroke: "#081311" }));
  chart.append(node("circle", { cx: x, cy: yAt(p90s[app.frame]), r: "4", fill: "#ffe083", stroke: "#081311" }));
  $("timelineMean").textContent = `${means[app.frame].toFixed(3)} person/m²`;
  $("timelineP90").textContent = `${p90s[app.frame].toFixed(3)} person/m²`;
  $("timelineStart").textContent = `${clock(app.data.frames[0].t)} JST`;
  $("timelineEnd").textContent = `${clock(app.data.frames.at(-1).t)} JST`;
  chart.addEventListener("click", (event) => {
    const bounds = chart.getBoundingClientRect();
    const svgX = ((event.clientX - bounds.left) / bounds.width) * 900;
    app.frame = Math.round(clamp((svgX - plot.left) / (plot.right - plot.left), 0, 1) * (app.data.frames.length - 1));
    render();
  }, { once: true });
}

function renderComparison() {
  const current = routes();
  if (!current) return;
  const baseline = current.BASELINE;
  const aware = current.CONGESTION_AWARE;
  $("planningTime").textContent = `${clock(nearestPlanning())} JST · ${app.robot}`;
  $("baselineDistance").textContent = `${baseline.distance.toFixed(1)} m`;
  $("awareDistance").textContent = `${aware.distance.toFixed(1)} m`;
  $("baselineEta").textContent = `${baseline.actualEta.toFixed(2)} s`;
  $("awareEta").textContent = `${aware.actualEta.toFixed(2)} s`;
  $("baselineExposure").textContent = baseline.exposure.toFixed(3);
  $("awareExposure").textContent = aware.exposure.toFixed(3);
  $("baselineUncertainty").textContent = `${baseline.staleDistance.toFixed(1)} / ${baseline.unknownDistance.toFixed(1)} m`;
  $("awareUncertainty").textContent = `${aware.staleDistance.toFixed(1)} / ${aware.unknownDistance.toFixed(1)} m`;
  $("disclaimer").textContent = app.data.meta.disclaimer;
}

function caviarObservation() {
  const zone = app.data.caviar.zones.find((candidate) => candidate.id === app.caviarZone);
  if (!zone) return null;
  if (!zone.observations.length) return { zone, evidence: [zone.bestFrame, ...zone.bestCounts] };
  return { zone, evidence: zone.observations[app.caviarObservation] };
}

function renderDemoCctv(atcZoneId) {
  const mapping = app.data.caviar.demoMappings[atcZoneId];
  if (!mapping) return;
  if (app.demoAtcZone !== atcZoneId) {
    const zone = app.data.caviar.zones.find((candidate) => candidate.id === mapping.caviarZone);
    app.demoAtcZone = atcZoneId;
    app.caviarZone = mapping.caviarZone;
    app.caviarCamera = mapping.camera;
    app.caviarObservation = Math.max(0, zone.observations.findIndex((item) => item[0] === mapping.frame));
    document.querySelectorAll("[data-camera]").forEach((button) => {
      button.classList.toggle("active", Number(button.dataset.camera) === app.caviarCamera);
    });
  }
  $("demoAtcZone").textContent = atcZoneId;
  $("demoCaviarZone").textContent = mapping.caviarZone;
  seekCaviarVideo();
}

async function seekCaviarVideo() {
  const current = caviarObservation();
  if (!current) return;
  const [frameIndex, corridorCount, frontCount] = current.evidence;
  const seconds = frameIndex / app.data.caviar.fps;
  $("caviarFrameLabel").textContent = `Frame ${frameIndex.toLocaleString()}`;
  $("caviarTimeLabel").textContent = `${seconds.toFixed(2)} s`;
  $("corridorCount").textContent = corridorCount;
  $("frontCount").textContent = frontCount;
  const notice = $("caviarMediaNotice");
  const video = $("evidenceVideo");
  const source = app.data.caviar.cameras[app.caviarCamera].media;
  notice.hidden = true;
  video.hidden = true;
  if (app.mediaAvailability[source] == null) {
    try {
      const response = await fetch(source, { method: "HEAD", cache: "no-store" });
      app.mediaAvailability[source] = response.ok && (response.headers.get("content-type") || "").startsWith("video/");
    } catch (_error) {
      app.mediaAvailability[source] = false;
    }
  }
  if (app.data.caviar.cameras[app.caviarCamera].media !== source) return;
  if (!app.mediaAvailability[source]) {
    notice.hidden = false;
    notice.textContent = `${app.data.caviar.notice} Run “make web-serve” and open its printed URL to view it.`;
    return;
  }
  video.hidden = false;
  const seek = () => { video.currentTime = seconds; };
  if (!video.src.endsWith(source)) {
    video.src = source;
    video.addEventListener("loadedmetadata", seek, { once: true });
  } else if (video.readyState >= 1) {
    seek();
  } else {
    video.addEventListener("loadedmetadata", seek, { once: true });
  }
}

function stepCaviarEvidence(direction) {
  const zone = app.data.caviar.zones.find((candidate) => candidate.id === app.caviarZone);
  if (!zone?.observations.length) return;
  app.caviarObservation = (app.caviarObservation + direction + zone.observations.length) % zone.observations.length;
  seekCaviarVideo();
}

function render() {
  renderMetrics(); renderMap(); renderInspector(); renderEvents(); renderTimeline(); renderComparison();
}

function bind() {
  $("timeSlider").max = app.data.frames.length - 1;
  $("timeSlider").addEventListener("input", (event) => { app.frame = Number(event.target.value); render(); });
  $("playButton").addEventListener("click", () => {
    app.playing = !app.playing;
    clearInterval(app.timer);
    if (app.playing) app.timer = setInterval(() => { app.frame = (app.frame + 1) % app.data.frames.length; render(); }, 500);
    renderMetrics();
  });
  document.querySelectorAll("[data-layer]").forEach((button) => button.addEventListener("click", () => {
    app.layer = button.dataset.layer;
    document.querySelectorAll("[data-layer]").forEach((candidate) => candidate.classList.toggle("active", candidate === button));
    renderMap(); renderInspector();
  }));
  $("robotSelect").addEventListener("change", (event) => { app.robot = event.target.value; render(); });
  $("baselineToggle").addEventListener("change", renderMap);
  $("awareToggle").addEventListener("change", renderMap);
  $("caviarPrevious").addEventListener("click", () => stepCaviarEvidence(-1));
  $("caviarNext").addEventListener("click", () => stepCaviarEvidence(1));
  document.querySelectorAll("[data-camera]").forEach((button) => button.addEventListener("click", () => {
    app.caviarCamera = Number(button.dataset.camera);
    document.querySelectorAll("[data-camera]").forEach((candidate) => candidate.classList.toggle("active", candidate === button));
    seekCaviarVideo();
  }));
}

async function start() {
  try {
    const version = encodeURIComponent(window.VACAN_BUILD_VERSION || "development");
    const response = await fetch(`data/dashboard.json?v=${version}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    app.data = await response.json();
    app.robot = app.data.robots[0];
    app.zone = app.data.zoneIds[0];
    app.data.robots.forEach((robot) => $("robotSelect").append(new Option(robot, robot)));
    bind(); render();
  } catch (error) {
    $("clockLabel").textContent = `Data load failed: ${error.message}`;
  }
}

start();
