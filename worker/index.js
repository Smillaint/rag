const RAG_HEALTH_URL = "https://rag.smillick.org/health";
const RELEASE_URL = "https://github.com/Smillaint/Ace-Taffy-Wiki-Tool/releases/latest";
const STATUS_TIMEOUT_MS = 5000;
const RAG_ASK_MAX_BODY_BYTES = 65536;

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

function ensureRequestId(request) {
  const existing = request.headers.get("X-Request-ID");
  if (existing) {
    const trimmed = existing.trim();
    if (trimmed.length > 0) {
      return trimmed;
    }
  }
  return crypto.randomUUID();
}

async function constantTimeEquals(candidate, secret) {
  if (
    typeof candidate !== "string" ||
    typeof secret !== "string" ||
    candidate.length === 0 ||
    secret.length === 0
  ) {
    return false;
  }

  const enc = new TextEncoder();
  const [candidateDigest, secretDigest] = await Promise.all([
    crypto.subtle.digest("SHA-256", enc.encode(candidate)),
    crypto.subtle.digest("SHA-256", enc.encode(secret)),
  ]);

  return crypto.subtle.timingSafeEqual(candidateDigest, secretDigest);
}

function extractClientToken(request) {
  const authHeader = request.headers.get("Authorization");
  if (authHeader) {
    const trimmed = authHeader.trim();
    const space = trimmed.indexOf(" ");
    if (space > 0) {
      const scheme = trimmed.slice(0, space).toLowerCase();
      if (scheme === "bearer") {
        const token = trimmed.slice(space + 1).trim();
        if (token.length > 0) {
          return token;
        }
      }
    }
  }

  const apiKeyHeader = request.headers.get("X-API-Key");
  if (apiKeyHeader) {
    const token = apiKeyHeader.trim();
    if (token.length > 0) {
      return token;
    }
  }

  return null;
}

async function authenticateGateway(request, env) {
  const gatewayKey = env.RAG_GATEWAY_API_KEY;
  const clientToken = extractClientToken(request);
  if (!clientToken || !gatewayKey) {
    return false;
  }

  return constantTimeEquals(clientToken, gatewayKey);
}

async function applyRateLimiter(limiter, request) {
  if (!limiter || typeof limiter.limit !== "function") {
    return { success: true };
  }

  const key = request.headers.get("CF-Connecting-IP") || "shared";
  try {
    return await limiter.limit({ key });
  } catch {
    return { success: true };
  }
}

function originUrl(env, path) {
  const base = String(env.RAG_ORIGIN_URL || "").replace(/\/+$/, "");
  return `${base}${path}`;
}

function buildUpstreamHeaders(request, env, requestId, includeAuth) {
  const headers = new Headers();

  const accept = request.headers.get("Accept");
  if (accept) {
    headers.set("Accept", accept);
  }

  const contentType = request.headers.get("Content-Type");
  if (contentType) {
    headers.set("Content-Type", contentType);
  }

  if (includeAuth && env.RAG_ORIGIN_API_KEY) {
    headers.set("Authorization", `Bearer ${env.RAG_ORIGIN_API_KEY}`);
  }

  headers.set("X-Request-ID", requestId);
  return headers;
}

async function readBoundedBody(stream, maxBytes) {
  if (!stream) {
    return { ok: true, buffer: new ArrayBuffer(0) };
  }

  const reader = stream.getReader();
  const chunks = [];
  let total = 0;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      total += value.byteLength;
      if (total > maxBytes) {
        return { ok: false };
      }

      chunks.push(value);
    }
  } finally {
    try {
      await reader.cancel();
    } catch {}
  }

  const buffer = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.byteLength;
  }

  return { ok: true, buffer: buffer.buffer };
}

function errorResponse(requestId, error, status, extraHeaders = {}) {
  const headers = new Headers(extraHeaders);
  headers.set("X-Request-ID", requestId);
  return jsonResponse({ error, request_id: requestId }, { status, headers });
}

function streamUpstreamResponse(upstreamResponse, requestId) {
  const headers = new Headers(upstreamResponse.headers);
  for (const [name, value] of Object.entries(SECURITY_HEADERS)) {
    headers.set(name, value);
  }
  headers.set("X-Request-ID", requestId);

  return new Response(upstreamResponse.body, {
    status: upstreamResponse.status,
    statusText: upstreamResponse.statusText,
    headers,
  });
}

async function proxyToOrigin(request, env, path, options = {}) {
  const method = options.method || request.method;
  const requestId = options.requestId || ensureRequestId(request);
  const includeAuth = options.includeAuth !== undefined ? options.includeAuth : true;

  if (!env.RAG_ORIGIN_URL) {
    return errorResponse(requestId, "origin_not_configured", 502);
  }

  const headers = buildUpstreamHeaders(request, env, requestId, includeAuth);

  let body = null;
  if (method !== "GET" && method !== "HEAD") {
    if (options.maxBodyBytes !== undefined) {
      const contentLength = request.headers.get("Content-Length");
      if (contentLength) {
        const parsed = Number(contentLength);
        if (Number.isFinite(parsed) && parsed > options.maxBodyBytes) {
          return errorResponse(requestId, "request_too_large", 413);
        }
      }

      const result = await readBoundedBody(request.body, options.maxBodyBytes);
      if (!result.ok) {
        return errorResponse(requestId, "request_too_large", 413);
      }
      body = result.buffer;
    } else {
      body = request.body;
    }
  }

  let upstreamResponse;
  try {
    upstreamResponse = await fetch(originUrl(env, path), { method, headers, body });
  } catch {
    return errorResponse(requestId, "upstream_unavailable", 502);
  }

  return streamUpstreamResponse(upstreamResponse, requestId);
}

async function proxyProtected(request, env, path, rateLimiter, options = {}) {
  const requestId = ensureRequestId(request);

  const rateLimitResult = await applyRateLimiter(rateLimiter, request);
  if (rateLimitResult && rateLimitResult.success === false) {
    return errorResponse(requestId, "rate_limited", 429, { "Retry-After": "60" });
  }

  const authenticated = await authenticateGateway(request, env);
  if (!authenticated) {
    return errorResponse(requestId, "unauthorized", 401, {
      "WWW-Authenticate": 'Bearer realm="rag"',
    });
  }

  return proxyToOrigin(request, env, path, {
    method: options.method || request.method,
    includeAuth: true,
    requestId,
    maxBodyBytes: options.maxBodyBytes,
  });
}

async function handleRagHealth(request, env) {
  if (request.method !== "GET" && request.method !== "HEAD") {
    return jsonResponse(
      { error: "method_not_allowed" },
      { status: 405, headers: { Allow: "GET, HEAD" } },
    );
  }

  return proxyToOrigin(request, env, "/health", {
    method: request.method,
    includeAuth: false,
  });
}

async function handleRagCollections(request, env) {
  if (request.method !== "GET" && request.method !== "HEAD") {
    return jsonResponse(
      { error: "method_not_allowed" },
      { status: 405, headers: { Allow: "GET, HEAD" } },
    );
  }

  return proxyProtected(request, env, "/collections", env.RAG_API_RATE_LIMITER, {
    method: request.method,
  });
}

async function handleRagStats(request, env) {
  if (request.method !== "GET" && request.method !== "HEAD") {
    return jsonResponse(
      { error: "method_not_allowed" },
      { status: 405, headers: { Allow: "GET, HEAD" } },
    );
  }

  return proxyProtected(request, env, "/stats", env.RAG_API_RATE_LIMITER, {
    method: request.method,
  });
}

async function handleRagAsk(request, env) {
  if (request.method !== "POST") {
    return jsonResponse(
      { error: "method_not_allowed" },
      { status: 405, headers: { Allow: "POST" } },
    );
  }

  return proxyProtected(request, env, "/ask", env.RAG_ASK_RATE_LIMITER, {
    method: "POST",
    maxBodyBytes: RAG_ASK_MAX_BODY_BYTES,
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

    if (url.pathname === "/api/v1/rag/health") {
      return handleRagHealth(request, env);
    }

    if (url.pathname === "/api/v1/rag/collections") {
      return handleRagCollections(request, env);
    }

    if (url.pathname === "/api/v1/rag/stats") {
      return handleRagStats(request, env);
    }

    if (url.pathname === "/api/v1/rag/ask") {
      return handleRagAsk(request, env);
    }

    if (url.pathname.startsWith("/api/")) {
      return jsonResponse({ error: "not_found" }, { status: 404 });
    }

    const response = await env.ASSETS.fetch(request);
    return withSecurityHeaders(response);
  },
};
