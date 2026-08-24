const RAG_HEALTH_URL = "https://rag.smillick.org/health";
const RELEASE_URL = "https://github.com/Smillaint/Ace-Taffy-Wiki-Tool/releases/latest";
const STATUS_TIMEOUT_MS = 5000;

const SECURITY_HEADERS = Object.freeze({
  "Content-Security-Policy": [
    "default-src 'self'",
    "base-uri 'none'",
    "form-action 'self'",
    "frame-ancestors 'none'",
    "img-src 'self' data:",
    "object-src 'none'",
    "script-src 'self'",
    "style-src 'self'",
  ].join("; "),
  "Cross-Origin-Opener-Policy": "same-origin",
  "Referrer-Policy": "strict-origin-when-cross-origin",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
});

function withSecurityHeaders(response) {
  const headers = new Headers(response.headers);
  for (const [name, value] of Object.entries(SECURITY_HEADERS)) {
    headers.set(name, value);
  }

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

function jsonResponse(payload, init = {}) {
  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json; charset=utf-8");
  headers.set("Cache-Control", "no-store");

  return withSecurityHeaders(
    new Response(JSON.stringify(payload), {
      ...init,
      headers,
    }),
  );
}

async function fetchRagStatus() {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), STATUS_TIMEOUT_MS);

  try {
    const response = await fetch(RAG_HEALTH_URL, {
      headers: {
        Accept: "application/json",
      },
      signal: controller.signal,
    });

    if (!response.ok) {
      return {
        available: false,
        status: "unavailable",
      };
    }

    const payload = await response.json();
    return {
      available: Boolean(payload.ready),
      status: payload.ready ? "ready" : "loading",
    };
  } catch {
    return {
      available: false,
      status: "unavailable",
    };
  } finally {
    clearTimeout(timeoutId);
  }
}

async function handleStatusRequest(request) {
  if (request.method !== "GET" && request.method !== "HEAD") {
    return jsonResponse(
      { error: "method_not_allowed" },
      {
        status: 405,
        headers: {
          Allow: "GET, HEAD",
        },
      },
    );
  }

  const rag = await fetchRagStatus();
  return jsonResponse({
    service: "smillick.org",
    status: "ok",
    rag,
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/api/status" || url.pathname === "/health") {
      return handleStatusRequest(request);
    }

    if (url.pathname === "/download") {
      return Response.redirect(RELEASE_URL, 302);
    }

    if (url.pathname.startsWith("/api/")) {
      return jsonResponse({ error: "not_found" }, { status: 404 });
    }

    const response = await env.ASSETS.fetch(request);
    return withSecurityHeaders(response);
  },
};
