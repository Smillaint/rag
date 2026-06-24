const statusText = document.querySelector("#statusText");
const collectionCount = document.querySelector("#collectionCount");
const chunkCount = document.querySelector("#chunkCount");
const vectorstoreState = document.querySelector("#vectorstoreState");
const codeFileCount = document.querySelector("#codeFileCount");
const collectionList = document.querySelector("#collectionList");
const fileList = document.querySelector("#fileList");
const milestoneList = document.querySelector("#milestoneList");
const projectName = document.querySelector("#projectName");
const projectSummary = document.querySelector("#projectSummary");
const branchName = document.querySelector("#branchName");
const codeTitle = document.querySelector("#codeTitle");
const codeSummary = document.querySelector("#codeSummary");
const codePath = document.querySelector("#codePath");
const codeBlock = document.querySelector("#codeBlock");
const refreshButton = document.querySelector("#refreshButton");

let activeFileKey = null;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function requestJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return response.json();
}

function renderProgress(data) {
  projectName.textContent = data.name || "Local PDF RAG QA";
  projectSummary.textContent = data.summary || "";
  branchName.textContent = data.branch || "-";
  statusText.textContent = data.status || "-";
  collectionCount.textContent = data.metrics?.collections ?? "-";
  chunkCount.textContent = data.metrics?.chunks ?? "-";
  vectorstoreState.textContent = data.metrics?.vectorstore ?? "-";
  codeFileCount.textContent = data.metrics?.code_files ?? "-";

  collectionList.innerHTML = (data.collections || [])
    .map((item) => `
      <div class="collection-item">
        <div>
          <div class="collection-name">${escapeHtml(item.name)}</div>
          <div class="muted">${item.pdf_count} PDF(s)</div>
        </div>
        <span class="tag ${item.loaded ? "done" : ""}">${item.loaded ? "loaded" : "local"}</span>
      </div>
    `)
    .join("");

  milestoneList.innerHTML = (data.milestones || [])
    .map((item) => `
      <article class="milestone">
        <div class="milestone-head">
          <h4>${escapeHtml(item.name)}</h4>
          <span class="tag ${escapeHtml(item.status)}">${escapeHtml(item.status)}</span>
        </div>
        <p>${escapeHtml(item.detail)}</p>
      </article>
    `)
    .join("");
}

function renderCodeFileList(files) {
  fileList.innerHTML = files
    .map((file) => `
      <button class="module-button" data-key="${escapeHtml(file.key)}">
        <strong>${escapeHtml(file.title)}</strong>
        <span>${escapeHtml(file.group)} · ${escapeHtml(file.summary)}</span>
      </button>
    `)
    .join("");

  fileList.querySelectorAll(".module-button").forEach((button) => {
    button.addEventListener("click", () => loadCode(button.dataset.key));
  });

  if (!activeFileKey && files.length) {
    loadCode(files[0].key);
  }
}

function renderCode(data) {
  activeFileKey = data.key;
  codeTitle.textContent = data.title;
  codeSummary.textContent = data.summary;
  codePath.textContent = `${data.path} · ${data.line_count} lines`;

  const lines = String(data.content || "").split("\n");
  codeBlock.innerHTML = lines
    .map((line, index) => `
      <span class="line">
        <span class="line-number">${index + 1}</span>
        <span class="line-code">${escapeHtml(line)}</span>
      </span>
    `)
    .join("");

  fileList.querySelectorAll(".module-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.key === activeFileKey);
  });
}

async function loadCode(key) {
  codeBlock.innerHTML = "<code>Loading...</code>";
  try {
    const data = await requestJson(`/project/code/${encodeURIComponent(key)}`);
    renderCode(data);
  } catch (error) {
    codeBlock.innerHTML = `<code class="error">${escapeHtml(error.message)}</code>`;
  }
}

async function loadDashboard() {
  const [progress, codeFiles] = await Promise.all([
    requestJson("/project/progress"),
    requestJson("/project/code-files"),
  ]);
  renderProgress(progress);
  renderCodeFileList(codeFiles.files || []);
}

refreshButton.addEventListener("click", () => {
  loadDashboard().catch((error) => {
    statusText.textContent = "error";
    projectSummary.textContent = error.message;
  });
});

loadDashboard().catch((error) => {
  statusText.textContent = "error";
  projectSummary.textContent = error.message;
});
