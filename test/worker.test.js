import { describe, it, expect } from "vitest";
import { env } from "cloudflare:workers";
import { createExecutionContext, waitOnExecutionContext } from "cloudflare:test";
import { http, HttpResponse } from "msw";
import worker from "../worker/index.js";
import { network } from "./network.js";

const ORIGIN = "https://rag.smillick.org";
const GATEWAY_KEY = "test-gateway-key";
const ORIGIN_KEY = "test-origin-key";

function makeRequest(path, init = {}) {
  const headers = new Headers(init.headers);
  return new Request(`https://smillick.org${path}`, { ...init, headers });
}

async function callWorker(request) {
  const ctx = createExecutionContext();
  const response = await worker.fetch(request, env, ctx);
  await waitOnExecutionContext(ctx);
  return response;
}

describe("RAG gateway worker", () => {
  it("public health: proxies origin and includes request-id, security, no-store headers", async () => {
    network.use(
      http.get(`${ORIGIN}/health`, () => HttpResponse.json({ ready: true })),
    );

    const res = await callWorker(makeRequest("/api/v1/rag/health"));

    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ready: true });
    expect(res.headers.get("Cache-Control")).toBe("no-store");
    expect(res.headers.get("X-Content-Type-Options")).toBe("nosniff");
    expect(res.headers.get("X-Frame-Options")).toBe("DENY");
    expect(res.headers.get("Content-Security-Policy")).toBeTruthy();
    expect(res.headers.get("X-Request-ID")).toMatch(/^[A-Za-z0-9_-]+$/);
  });

  it("protected collections without auth returns 401 and does not call origin", async () => {
    let originCalls = 0;
    network.use(
      http.get(`${ORIGIN}/collections`, () => {
        originCalls++;
        return HttpResponse.json([]);
      }),
    );

    const res = await callWorker(makeRequest("/api/v1/rag/collections"));

    expect(res.status).toBe(401);
    expect(originCalls).toBe(0);
    const body = await res.json();
    expect(body.error).toBe("unauthorized");
    expect(body.request_id).toBeTruthy();
    expect(res.headers.get("Cache-Control")).toBe("no-store");
  });

  it("valid Bearer auth proxies to /collections and replaces client auth with origin key", async () => {
    let capturedAuth = null;
    network.use(
      http.get(`${ORIGIN}/collections`, ({ request }) => {
        capturedAuth = request.headers.get("Authorization");
        return HttpResponse.json([{ id: "col1", name: "Test" }]);
      }),
    );

    const res = await callWorker(
      makeRequest("/api/v1/rag/collections", {
        headers: { Authorization: `Bearer ${GATEWAY_KEY}` },
      }),
    );

    expect(res.status).toBe(200);
    expect(capturedAuth).toBe(`Bearer ${ORIGIN_KEY}`);
    expect(await res.json()).toEqual([{ id: "col1", name: "Test" }]);
  });

  it("wrong method returns 405 with Allow header and request_id", async () => {
    const res = await callWorker(
      makeRequest("/api/v1/rag/collections", { method: "POST" }),
    );

    expect(res.status).toBe(405);
    expect(res.headers.get("Allow")).toBe("GET");
    const body = await res.json();
    expect(body.error).toBe("method_not_allowed");
    expect(body.request_id).toBeTruthy();
    expect(res.headers.get("X-Request-ID")).toBe(body.request_id);
  });

  it("ask rejects non-JSON content type with 415", async () => {
    let originCalls = 0;
    network.use(
      http.post(`${ORIGIN}/ask`, () => {
        originCalls++;
        return HttpResponse.json({ answer: "x" });
      }),
    );

    const res = await callWorker(
      makeRequest("/api/v1/rag/ask", {
        method: "POST",
        headers: {
          "Content-Type": "text/plain",
          Authorization: `Bearer ${GATEWAY_KEY}`,
        },
        body: "hello",
      }),
    );

    expect(res.status).toBe(415);
    expect(originCalls).toBe(0);
    const body = await res.json();
    expect(body.error).toBe("unsupported_media_type");
  });

  it("ask rejects body above 65536 bytes with 413 without calling origin", async () => {
    let originCalls = 0;
    network.use(
      http.post(`${ORIGIN}/ask`, () => {
        originCalls++;
        return HttpResponse.json({ answer: "x" });
      }),
    );

    const oversized = "x".repeat(65537);
    const res = await callWorker(
      makeRequest("/api/v1/rag/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${GATEWAY_KEY}`,
        },
        body: oversized,
      }),
    );

    expect(res.status).toBe(413);
    expect(originCalls).toBe(0);
    const body = await res.json();
    expect(body.error).toBe("request_too_large");
  });

  it("upstream non-2xx becomes safe 502 and does not leak upstream body", async () => {
    network.use(
      http.post(`${ORIGIN}/ask`, () =>
        HttpResponse.json({ secret: "SECRET_LEAK_VALUE" }, { status: 500 }),
      ),
    );

    const res = await callWorker(
      makeRequest("/api/v1/rag/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${GATEWAY_KEY}`,
        },
        body: JSON.stringify({ question: "test" }),
      }),
    );

    expect(res.status).toBe(502);
    const text = await res.text();
    expect(text).not.toContain("SECRET_LEAK_VALUE");
    const parsed = JSON.parse(text);
    expect(parsed.error).toBe("upstream_error");
    expect(parsed.request_id).toBeTruthy();
  });

  it("successful ask preserves response and includes no-store, security, request-id headers", async () => {
    network.use(
      http.post(`${ORIGIN}/ask`, () =>
        HttpResponse.json({ answer: "42", sources: [] }),
      ),
    );

    const res = await callWorker(
      makeRequest("/api/v1/rag/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${GATEWAY_KEY}`,
        },
        body: JSON.stringify({ question: "what is the answer?" }),
      }),
    );

    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ answer: "42", sources: [] });
    expect(res.headers.get("Cache-Control")).toBe("no-store");
    expect(res.headers.get("X-Content-Type-Options")).toBe("nosniff");
    expect(res.headers.get("X-Frame-Options")).toBe("DENY");
    expect(res.headers.get("X-Request-ID")).toMatch(/^[A-Za-z0-9_-]+$/);
  });

  it("malformed incoming X-Request-ID is replaced with a valid one", async () => {
    network.use(
      http.get(`${ORIGIN}/health`, () => HttpResponse.json({ ready: true })),
    );

    const res = await callWorker(
      makeRequest("/api/v1/rag/health", {
        headers: { "X-Request-ID": "!!! invalid !!!" },
      }),
    );

    expect(res.status).toBe(200);
    const requestId = res.headers.get("X-Request-ID");
    expect(requestId).not.toBe("!!! invalid !!!");
    expect(requestId).toMatch(/^[A-Za-z0-9_-]{1,128}$/);
  });

  it("HEAD /health sends GET to origin and returns empty body with same status/headers", async () => {
    let originMethod = null;
    network.use(
      http.get(`${ORIGIN}/health`, ({ request }) => {
        originMethod = request.method;
        return HttpResponse.json({ ready: true });
      }),
    );

    const res = await callWorker(
      makeRequest("/api/v1/rag/health", { method: "HEAD" }),
    );

    expect(originMethod).toBe("GET");
    expect(res.status).toBe(200);
    expect(await res.text()).toBe("");
    expect(res.headers.get("Cache-Control")).toBe("no-store");
    expect(res.headers.get("X-Content-Type-Options")).toBe("nosniff");
    expect(res.headers.get("X-Frame-Options")).toBe("DENY");
    expect(res.headers.get("X-Request-ID")).toMatch(/^[A-Za-z0-9_-]+$/);
  });
});
