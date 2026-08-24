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

const MODULES = Object.freeze([
  {
    key: "worker",
    title: "Cloudflare Worker",
    group: "Edge",
    summary: "托管根域页面并提供经过清洗的健康状态。",
    content: [
      "smillick.org",
      "  ├─ static assets",
      "  ├─ security headers",
      "  └─ /api/status",
    ].join("\n"),
  },
  {
    key: "tunnel",
    title: "Cloudflare Tunnel",
    group: "Network",
    summary: "通过出站隧道连接本地 TraceRAG 服务。",
    content: [
      "rag.smillick.org",
      "  → Cloudflare Tunnel",
      "  → 127.0.0.1:8765",
      "  → TraceRAG API",
    ].join("\n"),
  },
  {
    key: "desktop",
    title: "TaffySearchTool",
    group: "Desktop",
    summary: "使用受保护 API 完成知识库检索与引用展示。",
    content: [
      "TaffySearchTool",
      "  → authenticated request",
      "  → hybrid retrieval",
      "  → traceable answer",
    ].join("\n"),
  },
]);

let activeModuleKey = null;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderModule(module) {
  activeModuleKey = module.key;
  codeTitle.textContent = module.title;
  codeSummary.textContent = module.summary;
  codePath.textContent = module.group;
  codeBlock.innerHTML = escapeHtml(module.content);

  fileList.querySelectorAll(".module-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.key === activeModuleKey);
  });
}

function renderStaticContent() {
  projectName.textContent = "TraceRAG";
  projectSummary.textContent =
    "面向 TaffySearchTool 的可追踪知识检索服务。根域由 Worker 托管，RAG API 通过安全隧道连接本地模型与向量库。";
  branchName.textContent = "smillick.org";
  collectionCount.textContent = "Protected";
  chunkCount.textContent = "Private";
  codeFileCount.textContent = String(MODULES.length);

  collectionList.innerHTML = `
    <div class="collection-item">
      <div>
        <div class="collection-name">TraceRAG API</div>
        <div class="muted">rag.smillick.org</div>
      </div>
      <span id="ragCollectionState" class="tag">checking</span>
    </div>
  `;

  milestoneList.innerHTML = [
    ["Worker 边缘入口", "done", "根域静态资源、状态接口与安全响应头由 Cloudflare Worker 提供。"],
    ["Tunnel 私有源站", "done", "RAG 后端保持在本机，通过出站隧道提供受保护的 HTTPS 服务。"],
    ["桌面端集成", "planned", "TaffySearchTool 使用独立 API Key 连接知识服务，不在网页暴露共享密钥。"],
  ]
    .map(
      ([name, status, detail]) => `
        <article class="milestone">
          <div class="milestone-head">
            <h4>${escapeHtml(name)}</h4>
            <span class="tag ${escapeHtml(status)}">${escapeHtml(status)}</span>
          </div>
          <p>${escapeHtml(detail)}</p>
        </article>
      `,
    )
    .join("");

  fileList.innerHTML = MODULES.map(
    (module) => `
      <button class="module-button" data-key="${escapeHtml(module.key)}">
        <strong>${escapeHtml(module.title)}</strong>
        <span>${escapeHtml(module.group)} · ${escapeHtml(module.summary)}</span>
      </button>
    `,
  ).join("");

  fileList.querySelectorAll(".module-button").forEach((button) => {
    button.addEventListener("click", () => {
      const module = MODULES.find((item) => item.key === button.dataset.key);
      if (module) {
        renderModule(module);
      }
    });
  });

  renderModule(MODULES[0]);
}

async function loadStatus() {
  statusText.textContent = "Checking";
  vectorstoreState.textContent = "Checking";
  refreshButton.disabled = true;

  try {
    const response = await fetch("/api/status", {
      headers: {
        Accept: "application/json",
      },
    });
    if (!response.ok) {
      throw new Error(`Status request failed: ${response.status}`);
    }

    const payload = await response.json();
    const isReady = payload?.rag?.available === true;
    statusText.textContent = isReady ? "Online" : "Degraded";
    vectorstoreState.textContent = isReady ? "Ready" : "Unavailable";

    const collectionState = document.querySelector("#ragCollectionState");
    collectionState.textContent = isReady ? "ready" : "offline";
    collectionState.classList.toggle("done", isReady);
  } catch {
    statusText.textContent = "Unavailable";
    vectorstoreState.textContent = "Unknown";
  } finally {
    refreshButton.disabled = false;
  }
}

refreshButton.addEventListener("click", loadStatus);
renderStaticContent();
loadStatus();
