const form = document.querySelector("#resolve-form");
const health = document.querySelector("#health");
const emptyState = document.querySelector("#empty-state");
const resolved = document.querySelector("#resolved");
const cardName = document.querySelector("#card-name");
const cardMeta = document.querySelector("#card-meta");
const priceLabel = document.querySelector("#price-label");
const priceValue = document.querySelector("#price-value");
const priceMeta = document.querySelector("#price-meta");
const diagnostics = document.querySelector("#diagnostics");
const debugPanel = document.querySelector("#debug-panel");
const debugImage = document.querySelector("#debug-image");
const bboxStage = document.querySelector("#bbox-stage");
const bboxImage = document.querySelector("#bbox-image");
const bboxLayer = document.querySelector("#bbox-layer");
const bboxNote = document.querySelector("#bbox-note");
const imageInput = form.querySelector('input[name="image"]');
const previewWrap = document.querySelector("#preview-wrap");
const previewImage = document.querySelector("#preview-image");
const clearPhoto = document.querySelector("#clear-photo");
const photoTitle = document.querySelector("#photo-title");
const photoMeta = document.querySelector("#photo-meta");
const submitButton = document.querySelector("#submit-button");
const formStatus = document.querySelector("#form-status");
const cardArtWrap = document.querySelector("#card-art-wrap");
const cardArt = document.querySelector("#card-art");
const priceSource = document.querySelector("#price-source");
const matchList = document.querySelector("#match-list");
const startCamera = document.querySelector("#start-camera");
const stopCamera = document.querySelector("#stop-camera");
const captureFrame = document.querySelector("#capture-frame");
const liveView = document.querySelector("#live-view");
const liveVideo = document.querySelector("#live-video");
const liveCanvas = document.querySelector("#live-canvas");
const liveLayer = document.querySelector("#live-layer");
const liveStatus = document.querySelector("#live-status");
const liveBadge = document.querySelector("#live-badge");

let previewUrl = null;
let lastDetectorDebug = null;
let liveStream = null;
let liveTimer = null;
let liveDetecting = false;
let liveStableReadings = 0;
let liveLastBox = null;
let liveLastDetection = null;
let liveAutoScanLocked = false;

async function loadHealth() {
  const response = await fetch("/api/health");
  const data = await response.json();
  health.textContent = data.visual_index_available
    ? `Visual index ready · ${data.embedding_model}`
    : "Visual index missing, build embeddings first";
}

function renderResult(data) {
  renderDebug(data.debug);
  setStatus("");
  if (!data.best_match) {
    emptyState.classList.remove("hidden");
    resolved.classList.add("hidden");
    emptyState.querySelector("h2").textContent = "No visual match";
    emptyState.querySelector("p").textContent =
      data.warnings?.join(" ") || "Build the vector index or try a clearer photo.";
    return;
  }

  const best = data.best_match;
  const card = best.card;
  emptyState.classList.add("hidden");
  resolved.classList.remove("hidden");
  cardName.textContent = card.name;
  cardMeta.textContent = `${card.set_code} ${card.printed_number} · ${card.set_name}`;
  renderCardArt(card);
  if (data.price) {
    renderPrice(data.price);
  } else {
    renderPriceLoading();
    loadPrice(card);
  }
  renderMatches(data.matches || []);

  const confidence = Math.round(best.score * 100);
  const latency = data.latency_ms || {};
  const warningText = data.warnings?.length ? ` · ${data.warnings.join(" ")}` : "";
  diagnostics.textContent = `Top match ${confidence}% · total ${latency.total ?? "?"} ms · embedding ${
    latency.embedding ?? "?"
  } ms · search ${latency.vector_search ?? "?"} ms${warningText}`;
}

function renderCardArt(card) {
  const imageUrl = card.image_url || (card.local_image_path ? `/${card.local_image_path}` : "");
  if (!imageUrl) {
    cardArtWrap.classList.add("hidden");
    cardArt.removeAttribute("src");
    cardArt.alt = "";
    return;
  }
  cardArtWrap.classList.remove("hidden");
  cardArt.src = imageUrl;
  cardArt.alt = `${card.name} card image`;
}

function renderPrice(price) {
  if (!price) {
    priceLabel.textContent = "Price";
    priceValue.textContent = "Unavailable";
    priceMeta.textContent = "No pricing provider result";
    priceSource.classList.add("hidden");
    return;
  }
  priceLabel.textContent = `${price.mode === "trend" ? "Trend" : "Minimum"} price`;
  priceValue.textContent =
    price.amount === null ? "Unavailable" : `${price.amount.toFixed(2)} ${price.currency}`;
  priceMeta.textContent = price.message
    ? `${price.source} · ${price.message}`
    : `${price.source} · ${price.filters.language} · ${price.filters.seller_country}`;
  const url = price?.filters?.url;
  if (!url) {
    priceSource.classList.add("hidden");
    priceSource.removeAttribute("href");
    return;
  }
  priceSource.href = url;
  priceSource.classList.remove("hidden");
}

function renderPriceLoading() {
  priceLabel.textContent = "Price";
  priceValue.textContent = "Loading...";
  priceMeta.textContent = "Fetching provider price";
  priceSource.classList.add("hidden");
  priceSource.removeAttribute("href");
}

async function loadPrice(card) {
  try {
    const response = await fetch("/api/price", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        card_id: card.card_id,
        language: form.elements.language.value,
        seller_country: form.elements.seller_country.value,
        price_mode: form.elements.price_mode.value,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Price unavailable");
    renderPrice(data.price);
  } catch (error) {
    renderPrice(null);
    priceMeta.textContent = "Price provider unavailable";
  }
}

function renderMatches(matches) {
  matchList.replaceChildren();
  for (const match of matches) {
    const row = document.createElement("div");
    row.className = "match-row";
    const title = document.createElement("strong");
    const meta = document.createElement("span");
    title.textContent = `${match.rank}. ${match.card.name}`;
    meta.textContent = `${match.card.set_code} ${match.card.printed_number} · score ${Math.round(
      match.score * 100,
    )}% · distance ${match.distance.toFixed(4)}`;
    row.append(title, meta);
    matchList.append(row);
  }
}

function renderDebug(debug) {
  if (!debug?.preprocessed_image) {
    clearDebug();
    return;
  }
  debugPanel.classList.remove("hidden");
  debugImage.src = debug.preprocessed_image;
  lastDetectorDebug = debug.detector || null;
  renderYoloOverlay();
}

function renderYoloOverlay() {
  bboxLayer.replaceChildren();
  if (!previewUrl) {
    bboxImage.removeAttribute("src");
    bboxNote.textContent = "No original preview available.";
    return;
  }
  bboxImage.src = previewUrl;
  const detections = lastDetectorDebug?.detections || [];
  if (!detections.length) {
    bboxNote.textContent = "YOLO returned no card box; OpenCV/full-image fallback was used.";
    return;
  }
  for (const detection of detections) {
    const box = document.createElement("div");
    const label = document.createElement("span");
    const left = (detection.x / lastDetectorDebug.image_width) * 100;
    const top = (detection.y / lastDetectorDebug.image_height) * 100;
    const width = (detection.width / lastDetectorDebug.image_width) * 100;
    const height = (detection.height / lastDetectorDebug.image_height) * 100;
    box.className = `bbox ${detection.selected ? "selected" : ""}`;
    box.style.left = `${clampPercent(left)}%`;
    box.style.top = `${clampPercent(top)}%`;
    box.style.width = `${clampPercent(width)}%`;
    box.style.height = `${clampPercent(height)}%`;
    label.textContent = `${detection.label} ${Math.round(detection.confidence * 100)}%`;
    box.append(label);
    bboxLayer.append(box);
  }
  const selected = detections.find((item) => item.selected) || detections[0];
  bboxNote.textContent = `YOLO ${lastDetectorDebug.used ? "used" : "not used"} · ${Math.round(
    selected.confidence * 100,
  )}% · ${Math.round(selected.width)} x ${Math.round(selected.height)} px`;
}

function clampPercent(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, value));
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = imageInput.files[0];
  if (!file) {
    if (liveStream && liveVideo.readyState >= 2) {
      const blob = await captureLiveBlob(1280, 0.9);
      if (blob) {
        await scanImageBlob(blob, "live-submit.jpg");
        return;
      }
    }
    emptyState.classList.remove("hidden");
    resolved.classList.add("hidden");
    emptyState.querySelector("h2").textContent = "Photo required";
    emptyState.querySelector("p").textContent = "Choose a photo or start the live camera before searching.";
    return;
  }

  setBusy(true, "Searching visual index...");
  try {
    const response = await fetch("/api/scan-image", {
      method: "POST",
      body: new FormData(form),
    });
    const data = await response.json();
    if (!response.ok) {
      renderDebug(data.debug);
      emptyState.classList.remove("hidden");
      resolved.classList.add("hidden");
      emptyState.querySelector("h2").textContent = "Search failed";
      emptyState.querySelector("p").textContent = data.error || "Check the input and try again.";
      setStatus("Search failed.");
      return;
    }
    renderResult(data);
  } catch (error) {
    emptyState.classList.remove("hidden");
    resolved.classList.add("hidden");
    emptyState.querySelector("h2").textContent = "API unavailable";
    emptyState.querySelector("p").textContent = "Start the scanner server and try again.";
    setStatus("API unavailable.");
  } finally {
    setBusy(false);
  }
});

startCamera.addEventListener("click", async () => {
  if (!navigator.mediaDevices?.getUserMedia) {
    liveStatus.textContent = "Camera needs HTTPS on mobile browsers.";
    return;
  }
  try {
    liveStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });
    liveVideo.srcObject = liveStream;
    liveView.classList.remove("hidden");
    stopCamera.classList.remove("hidden");
    captureFrame.classList.remove("hidden");
    startCamera.disabled = true;
    liveStatus.textContent = "Looking for a full card border.";
    liveBadge.textContent = "Looking for card";
    liveAutoScanLocked = false;
    liveStableReadings = 0;
    liveLastBox = null;
    liveLastDetection = null;
    liveTimer = window.setInterval(detectLiveFrame, 450);
  } catch (error) {
    liveStatus.textContent = "Camera permission failed. Use HTTPS and allow camera access.";
  }
});

stopCamera.addEventListener("click", stopLiveCamera);

captureFrame.addEventListener("click", async () => {
  const blob = await captureLiveBlob();
  if (!blob) return;
  await scanImageBlob(blob, "manual-frame.jpg");
});

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) {
    clearPreview();
    return;
  }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewWrap.classList.remove("hidden");
  photoTitle.textContent = file.name;
  photoMeta.textContent = `${formatBytes(file.size)} selected. Submit to search by visual fingerprint.`;
  submitButton.textContent = "Search image";
  clearDebug();
});

clearPhoto.addEventListener("click", () => {
  imageInput.value = "";
  clearPreview();
});

function clearPreview() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  previewImage.removeAttribute("src");
  clearDebug();
  previewWrap.classList.add("hidden");
  photoTitle.textContent = "Choose or take a photo";
  photoMeta.textContent = "Use the full card face, with the lower-left collector line visible.";
  submitButton.textContent = "Search image";
}

function clearDebug() {
  debugPanel.classList.add("hidden");
  debugImage.removeAttribute("src");
  bboxImage.removeAttribute("src");
  bboxLayer.replaceChildren();
  bboxNote.textContent = "";
  lastDetectorDebug = null;
}

async function detectLiveFrame() {
  if (liveDetecting || liveAutoScanLocked || !liveStream || liveVideo.readyState < 2) return;
  const blob = await captureLiveBlob();
  if (!blob) return;

  liveDetecting = true;
  try {
    const body = new FormData();
    body.append("image", blob, "frame.jpg");
    const response = await fetch("/api/detect-region", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Detection failed");
    renderLiveDetections(data.detections || []);
    const selected = (data.detections || []).find((item) => item.selected) || data.detections?.[0];
    updateLiveStability(selected, data.latency_ms);
    if (liveStableReadings >= 3 && !liveAutoScanLocked) {
      liveAutoScanLocked = true;
      liveBadge.textContent = "Captured";
      liveStatus.textContent = "Card stable. Running scanner...";
      const scanBlob = await captureLiveBlob(1280, 0.9);
      if (scanBlob) await scanImageBlob(scanBlob, "live-card.jpg", liveLastDetection);
      stopLiveCamera();
    }
  } catch (error) {
    liveStatus.textContent = "Live detection unavailable.";
  } finally {
    liveDetecting = false;
  }
}

async function scanImageBlob(blob, filename, cropDetection = liveLastDetection) {
  setCapturedPreview(blob, filename);
  setBusy(true, "Scanning captured card...");
  try {
    const response = await fetch("/api/scan-image", {
      method: "POST",
      body: buildScanForm(blob, filename, cropDetection),
    });
    const data = await response.json();
    if (!response.ok) {
      renderDebug(data.debug);
      emptyState.classList.remove("hidden");
      resolved.classList.add("hidden");
      emptyState.querySelector("h2").textContent = "Search failed";
      emptyState.querySelector("p").textContent = data.error || "Check the input and try again.";
      return;
    }
    renderResult(data);
  } catch (error) {
    emptyState.classList.remove("hidden");
    resolved.classList.add("hidden");
    emptyState.querySelector("h2").textContent = "API unavailable";
    emptyState.querySelector("p").textContent = "Start the scanner server and try again.";
  } finally {
    setBusy(false);
  }
}

function setCapturedPreview(blob, filename) {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(blob);
  previewImage.src = previewUrl;
  previewWrap.classList.remove("hidden");
  photoTitle.textContent = filename;
  photoMeta.textContent = `${formatBytes(blob.size)} captured from live camera.`;
}

function buildScanForm(blob, filename, cropDetection = null) {
  const body = new FormData();
  body.append("image", blob, filename);
  body.append("language", form.elements.language.value);
  body.append("seller_country", form.elements.seller_country.value);
  body.append("top_k", form.elements.top_k.value);
  body.append("price_mode", form.elements.price_mode.value);
  if (cropDetection?.image_width && cropDetection?.image_height) {
    body.append("bbox_x", String(cropDetection.x));
    body.append("bbox_y", String(cropDetection.y));
    body.append("bbox_width", String(cropDetection.width));
    body.append("bbox_height", String(cropDetection.height));
    body.append("bbox_image_width", String(cropDetection.image_width));
    body.append("bbox_image_height", String(cropDetection.image_height));
  }
  return body;
}

async function captureLiveBlob(maxWidth = 1280, quality = 0.82) {
  if (!liveVideo.videoWidth || !liveVideo.videoHeight) return null;
  const scale = Math.min(1, maxWidth / liveVideo.videoWidth);
  liveCanvas.width = Math.round(liveVideo.videoWidth * scale);
  liveCanvas.height = Math.round(liveVideo.videoHeight * scale);
  const context = liveCanvas.getContext("2d");
  context.drawImage(liveVideo, 0, 0, liveCanvas.width, liveCanvas.height);
  return new Promise((resolve) => liveCanvas.toBlob(resolve, "image/jpeg", quality));
}

function renderLiveDetections(detections) {
  liveLayer.replaceChildren();
  if (!liveCanvas.width || !liveCanvas.height) return;
  for (const detection of detections) {
    const box = document.createElement("div");
    const label = document.createElement("span");
    box.className = `bbox ${detection.selected ? "selected" : ""}`;
    box.style.left = `${clampPercent((detection.x / liveCanvas.width) * 100)}%`;
    box.style.top = `${clampPercent((detection.y / liveCanvas.height) * 100)}%`;
    box.style.width = `${clampPercent((detection.width / liveCanvas.width) * 100)}%`;
    box.style.height = `${clampPercent((detection.height / liveCanvas.height) * 100)}%`;
    label.textContent = `${detection.label} ${Math.round(detection.confidence * 100)}%`;
    box.append(label);
    liveLayer.append(box);
  }
}

function updateLiveStability(detection, latencyMs) {
  if (!detection || detection.label !== "card" || detection.confidence < 0.78) {
    liveStableReadings = 0;
    liveLastBox = null;
    liveLastDetection = null;
    liveBadge.textContent = "Looking for card";
    liveStatus.textContent = latencyMs ? `No stable card · ${latencyMs} ms` : "No stable card";
    return;
  }
  const current = normalizeBox(detection, liveCanvas.width, liveCanvas.height);
  const area = current.width * current.height;
  const stable =
    liveLastBox &&
    Math.abs(current.cx - liveLastBox.cx) < 0.035 &&
    Math.abs(current.cy - liveLastBox.cy) < 0.035 &&
    Math.abs(current.width - liveLastBox.width) < 0.05 &&
    Math.abs(current.height - liveLastBox.height) < 0.05;
  liveStableReadings = stable && area > 0.12 ? liveStableReadings + 1 : 1;
  liveLastBox = current;
  liveLastDetection = { ...detection, image_width: liveCanvas.width, image_height: liveCanvas.height };
  liveBadge.textContent = liveStableReadings >= 2 ? "Hold still" : "Card found";
  liveStatus.textContent = `${Math.round(detection.confidence * 100)}% · stable ${liveStableReadings}/3 · ${
    latencyMs ?? "?"
  } ms`;
}

function normalizeBox(detection, width, height) {
  const boxWidth = detection.width / width;
  const boxHeight = detection.height / height;
  return {
    cx: (detection.x + detection.width / 2) / width,
    cy: (detection.y + detection.height / 2) / height,
    width: boxWidth,
    height: boxHeight,
  };
}

function stopLiveCamera() {
  if (liveTimer) window.clearInterval(liveTimer);
  liveTimer = null;
  liveDetecting = false;
  liveStableReadings = 0;
  liveLastBox = null;
  liveLastDetection = null;
  liveLayer.replaceChildren();
  if (liveStream) {
    for (const track of liveStream.getTracks()) track.stop();
  }
  liveStream = null;
  liveVideo.srcObject = null;
  liveView.classList.add("hidden");
  stopCamera.classList.add("hidden");
  captureFrame.classList.add("hidden");
  startCamera.disabled = false;
}

function setBusy(isBusy, message = "") {
  submitButton.disabled = isBusy;
  submitButton.textContent = isBusy ? message : "Search image";
  setStatus(message);
}

function setStatus(message) {
  formStatus.textContent = message;
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

loadHealth().catch(() => {
  health.textContent = "API unavailable";
});
