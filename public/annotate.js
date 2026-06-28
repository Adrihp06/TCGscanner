const imageSelect = document.querySelector("#image-select");
const canvas = document.querySelector("#annotation-canvas");
const ctx = canvas.getContext("2d");
const saveButton = document.querySelector("#save-button");
const deleteButton = document.querySelector("#delete-button");
const list = document.querySelector("#annotation-list");
const uploadInput = document.querySelector("#upload-input");

const colors = {
  card: "#168a56",
};

let images = [];
let currentImage = null;
let bitmap = null;
let boxes = [];
let selectedIndex = -1;
let drawing = null;

function selectedLabel() {
  return document.querySelector('input[name="label"]:checked').value;
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * canvas.width,
    y: ((event.clientY - rect.top) / rect.height) * canvas.height,
  };
}

async function loadDataset() {
  const response = await fetch("/api/dataset/images");
  const data = await response.json();
  images = sortImages(data.images);
  renderImageSelect();
  if (images.length) {
    await loadImage(images[0].id);
  }
}

function sortImages(items) {
  return [...items].sort((a, b) => {
    const aReal = a.path.includes("/user_samples/") ? 0 : 1;
    const bReal = b.path.includes("/user_samples/") ? 0 : 1;
    return aReal - bReal || a.path.localeCompare(b.path);
  });
}

function renderImageSelect() {
  imageSelect.replaceChildren();
  for (const image of images) {
    const option = document.createElement("option");
    option.value = image.id;
    option.textContent = `${image.annotated ? "✓ " : ""}${image.path}`;
    imageSelect.append(option);
  }
}

async function loadImage(imageId) {
  currentImage = images.find((image) => image.id === imageId);
  const image = new Image();
  image.src = currentImage.url;
  await image.decode();
  bitmap = image;
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const displayWidth = Math.min(980, Math.max(560, image.naturalWidth));
  canvas.style.width = `${displayWidth}px`;
  canvas.style.height = "auto";

  const response = await fetch(`/api/annotations/${encodeURIComponent(imageId)}`);
  const annotation = await response.json();
  boxes = annotation.boxes || [];
  selectedIndex = -1;
  draw();
  renderList();
}

function draw() {
  if (!bitmap) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(bitmap, 0, 0);
  boxes.forEach((box, index) => drawBox(box, index === selectedIndex));
  if (drawing) {
    drawBox(drawing, true);
  }
}

function drawBox(box, selected) {
  ctx.save();
  ctx.strokeStyle = colors[box.label] || "#222";
  ctx.lineWidth = selected ? 3 : 2;
  ctx.strokeRect(box.x, box.y, box.width, box.height);
  ctx.fillStyle = colors[box.label] || "#222";
  ctx.font = "13px system-ui";
  const label = box.label;
  const textWidth = ctx.measureText(label).width + 10;
  ctx.fillRect(box.x, Math.max(0, box.y - 20), textWidth, 20);
  ctx.fillStyle = "white";
  ctx.fillText(label, box.x + 5, Math.max(14, box.y - 6));
  ctx.restore();
}

function renderList() {
  list.replaceChildren();
  boxes.forEach((box, index) => {
    const item = document.createElement("div");
    item.className = `annotation-item${index === selectedIndex ? " selected" : ""}`;
    item.textContent = `${box.label}: ${Math.round(box.x)}, ${Math.round(box.y)}, ${Math.round(box.width)} x ${Math.round(box.height)}`;
    item.addEventListener("click", () => {
      selectedIndex = index;
      draw();
      renderList();
    });
    list.append(item);
  });
}

canvas.addEventListener("mousedown", (event) => {
  const point = canvasPoint(event);
  drawing = { label: selectedLabel(), x: point.x, y: point.y, width: 0, height: 0 };
});

canvas.addEventListener("mousemove", (event) => {
  if (!drawing) return;
  const point = canvasPoint(event);
  drawing.width = point.x - drawing.x;
  drawing.height = point.y - drawing.y;
  draw();
});

canvas.addEventListener("mouseup", () => {
  if (!drawing) return;
  const normalized = {
    label: drawing.label,
    x: drawing.width < 0 ? drawing.x + drawing.width : drawing.x,
    y: drawing.height < 0 ? drawing.y + drawing.height : drawing.y,
    width: Math.abs(drawing.width),
    height: Math.abs(drawing.height),
  };
  drawing = null;
  if (normalized.width >= 4 && normalized.height >= 4) {
    boxes.push(normalized);
    selectedIndex = boxes.length - 1;
  }
  draw();
  renderList();
});

deleteButton.addEventListener("click", () => {
  if (selectedIndex < 0) return;
  boxes.splice(selectedIndex, 1);
  selectedIndex = -1;
  draw();
  renderList();
});

saveButton.addEventListener("click", async () => {
  if (!currentImage) return;
  await fetch(`/api/annotations/${encodeURIComponent(currentImage.id)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ boxes }),
  });
  currentImage.annotated = true;
  const option = imageSelect.selectedOptions[0];
  if (option && !option.textContent.startsWith("✓ ")) {
    option.textContent = `✓ ${option.textContent}`;
  }
});

uploadInput.addEventListener("change", async () => {
  const file = uploadInput.files[0];
  if (!file) return;
  const form = new FormData();
  form.append("image", file);
  const response = await fetch("/api/dataset/upload", {
    method: "POST",
    body: form,
  });
  const data = await response.json();
  if (!response.ok) {
    alert(data.error || "Upload failed");
    return;
  }
  images = sortImages([data.image, ...images.filter((image) => image.id !== data.image.id)]);
  renderImageSelect();
  imageSelect.value = data.image.id;
  uploadInput.value = "";
  await loadImage(data.image.id);
});

imageSelect.addEventListener("change", () => loadImage(imageSelect.value));

loadDataset();
