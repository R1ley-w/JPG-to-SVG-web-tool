"use strict";

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const imagePreview = document.getElementById("image-preview");
const previewImg = document.getElementById("preview-img");
const filenameEl = document.getElementById("filename");
const placeholder = document.getElementById("placeholder");
const result = document.getElementById("result");
const svgPreview = document.getElementById("svg-preview");
const downloadBtn = document.getElementById("download-btn");
const loading = document.getElementById("loading");
const errorEl = document.getElementById("error");

const MAX_BYTES = 10 * 1024 * 1024;
const ACCEPTED = ["image/png", "image/jpeg", "image/webp"];

let currentSvg = null;
let currentName = "logo";

function showError(message) {
  errorEl.textContent = message;
  errorEl.hidden = false;
}

function clearError() {
  errorEl.textContent = "";
  errorEl.hidden = true;
}

function setLoading(on) {
  loading.hidden = !on;
  dropzone.classList.toggle("disabled", on);
}

function setPreview(file) {
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  imagePreview.hidden = false;
  filenameEl.textContent = file.name;
  currentName = file.name.replace(/\.[^.]+$/, "") || "logo";
}

async function convertFile(file) {
  if (!ACCEPTED.includes(file.type)) {
    showError("Please drop a PNG, JPEG, or WebP image.");
    return;
  }
  if (file.size > MAX_BYTES) {
    showError("That file is too large (max 10 MB).");
    return;
  }

  clearError();
  placeholder.hidden = true;
  result.hidden = true;
  downloadBtn.disabled = true;
  setPreview(file);
  setLoading(true);

  const form = new FormData();
  form.append("file", file);

  try {
    const resp = await fetch("/api/convert", { method: "POST", body: form });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || `Server error (${resp.status})`);
    }
    const data = await resp.json();
    currentSvg = data.svg;
    svgPreview.src = "data:image/svg+xml;utf8," + encodeURIComponent(data.svg);
    result.hidden = false;
    downloadBtn.disabled = false;
  } catch (err) {
    showError(err.message || "Something went wrong while converting.");
  } finally {
    setLoading(false);
  }
}

function downloadSvg() {
  if (!currentSvg) return;
  const blob = new Blob([currentSvg], { type: "image/svg+xml" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${currentName}.svg`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  const file = event.dataTransfer.files && event.dataTransfer.files[0];
  if (file) convertFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files[0]) {
    convertFile(fileInput.files[0]);
  }
  fileInput.value = "";
});

downloadBtn.addEventListener("click", downloadSvg);
