/**
 * Tests for the OpenAI-compatible embedding provider. Mocks global fetch so
 * none of the cases need network. Coverage: success path, batching, dim
 * mismatch error, hard fail on 401, retry with backoff on 429 and 5xx,
 * Retry-After honored, malformed responses rejected, missing API key.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { EmbeddingConfig } from "../src/config.js";
import { OpenAICompatEmbeddingService } from "../src/embeddings-openai.js";

function makeConfig(overrides: Partial<EmbeddingConfig> = {}): EmbeddingConfig {
  return {
    provider: "openai-compat",
    dimensions: 1024,
    modelPath: "",
    openaiCompat: {
      model: "text-embedding-3-small",
      baseURL: "https://api.openai.com/v1",
      apiKeyEnv: "TEST_OPENAI_KEY",
    },
    ...overrides,
  };
}

function vec(dim: number, fill = 0.1): number[] {
  return new Array(dim).fill(fill);
}

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

function makeFetchMock() {
  return vi.fn() as unknown as typeof fetch & ReturnType<typeof vi.fn>;
}

describe("OpenAICompatEmbeddingService", () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };

  beforeEach(() => {
    process.env.TEST_OPENAI_KEY = "sk-test-1234";
    // Default fetch mock — replaced in individual tests.
    (globalThis as any).fetch = makeFetchMock();
  });

  afterEach(() => {
    (globalThis as any).fetch = originalFetch;
    process.env = { ...originalEnv };
    vi.useRealTimers();
  });

  it("providerId encodes provider, model, and dimensions", () => {
    const svc = new OpenAICompatEmbeddingService(makeConfig());
    expect(svc.providerId).toBe("openai-compat-text-embedding-3-small-1024d");
    expect(svc.dimensions).toBe(1024);
  });

  it("initialize fails fast when API key env var is empty", async () => {
    delete process.env.TEST_OPENAI_KEY;
    const svc = new OpenAICompatEmbeddingService(makeConfig());
    await expect(svc.initialize()).rejects.toThrow(/API key not set/);
  });

  it("embed posts a request with bearer auth and dimensions, returns the vector", async () => {
    const fetchMock = vi.fn(async (_url: any, _init: any) =>
      jsonResponse({ data: [{ index: 0, embedding: vec(1024, 0.5) }] }),
    );
    (globalThis as any).fetch = fetchMock;

    const svc = new OpenAICompatEmbeddingService(makeConfig());
    await svc.initialize();
    const result = await svc.embed("hello");

    expect(result).toHaveLength(1024);
    expect(result[0]).toBe(0.5);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("https://api.openai.com/v1/embeddings");
    expect((init as any).method).toBe("POST");
    expect((init as any).headers["Authorization"]).toBe("Bearer sk-test-1234");
    const body = JSON.parse((init as any).body);
    expect(body.model).toBe("text-embedding-3-small");
    expect(body.input).toEqual(["hello"]);
    expect(body.dimensions).toBe(1024);
    expect(body.encoding_format).toBe("float");
  });

  it("embedBatch chunks inputs at the per-request limit and concatenates results", async () => {
    // 200 inputs at maxBatchSize=96 means 3 calls (96 + 96 + 8).
    const inputs = Array.from({ length: 200 }, (_, i) => `text-${i}`);
    let callCount = 0;
    const fetchMock = vi.fn(async (_url: any, init: any) => {
      callCount++;
      const body = JSON.parse(init.body);
      const chunkSize = body.input.length;
      return jsonResponse({
        data: body.input.map((_: string, i: number) => ({ index: i, embedding: vec(1024, callCount * 0.1 + i * 0.001) })),
      });
    });
    (globalThis as any).fetch = fetchMock;

    const svc = new OpenAICompatEmbeddingService(makeConfig());
    await svc.initialize();
    const results = await svc.embedBatch(inputs);

    expect(results).toHaveLength(200);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    const firstCallBody = JSON.parse((fetchMock.mock.calls[0][1] as any).body);
    expect(firstCallBody.input).toHaveLength(96);
    const lastCallBody = JSON.parse((fetchMock.mock.calls[2][1] as any).body);
    expect(lastCallBody.input).toHaveLength(8);
  });

  it("hard-fails on 401 without retrying", async () => {
    const fetchMock = vi.fn(async () =>
      new Response("Unauthorized", { status: 401 }),
    );
    (globalThis as any).fetch = fetchMock;

    const svc = new OpenAICompatEmbeddingService(makeConfig());
    await svc.initialize();
    await expect(svc.embed("hello")).rejects.toThrow(/auth failed.*401/);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("hard-fails on 404 (wrong baseURL)", async () => {
    const fetchMock = vi.fn(async () =>
      new Response("Not Found", { status: 404 }),
    );
    (globalThis as any).fetch = fetchMock;

    const svc = new OpenAICompatEmbeddingService(makeConfig());
    await svc.initialize();
    await expect(svc.embed("hello")).rejects.toThrow(/endpoint not found/);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("retries on 429 and succeeds on the second attempt", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response("rate limited", { status: 429, headers: { "retry-after": "0" } }),
      )
      .mockResolvedValueOnce(
        jsonResponse({ data: [{ index: 0, embedding: vec(1024, 0.7) }] }),
      );
    (globalThis as any).fetch = fetchMock;

    const svc = new OpenAICompatEmbeddingService(makeConfig());
    await svc.initialize();
    const result = await svc.embed("hello");

    expect(result[0]).toBe(0.7);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("retries on 5xx errors", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response("upstream error", { status: 502, headers: { "retry-after": "0" } }))
      .mockResolvedValueOnce(jsonResponse({ data: [{ index: 0, embedding: vec(1024, 0.3) }] }));
    (globalThis as any).fetch = fetchMock;

    const svc = new OpenAICompatEmbeddingService(makeConfig());
    await svc.initialize();
    const result = await svc.embed("hello");

    expect(result[0]).toBe(0.3);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("rejects when server returns a different dimension than configured", async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({ data: [{ index: 0, embedding: vec(1536, 0.1) }] }),
    );
    (globalThis as any).fetch = fetchMock;

    const svc = new OpenAICompatEmbeddingService(makeConfig({ dimensions: 1024 }));
    await svc.initialize();
    await expect(svc.embed("hello")).rejects.toThrow(/1536-dim vectors but config requested 1024/);
  });

  it("rejects when server returns fewer vectors than inputs", async () => {
    const fetchMock = vi.fn(async (_url: any, init: any) => {
      const body = JSON.parse(init.body);
      const inputCount = body.input.length;
      // Return one fewer than asked — surfaces a server-side bug.
      return jsonResponse({
        data: body.input.slice(0, inputCount - 1).map((_: string, i: number) => ({ index: i, embedding: vec(1024) })),
      });
    });
    (globalThis as any).fetch = fetchMock;

    const svc = new OpenAICompatEmbeddingService(makeConfig());
    await svc.initialize();
    await expect(svc.embedBatch(["a", "b", "c"])).rejects.toThrow(/returned 2 vectors for 3 inputs/);
  });

  it("honors out-of-order index field in response", async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({
        data: [
          { index: 1, embedding: vec(1024, 0.2) },
          { index: 0, embedding: vec(1024, 0.1) },
          { index: 2, embedding: vec(1024, 0.3) },
        ],
      }),
    );
    (globalThis as any).fetch = fetchMock;

    const svc = new OpenAICompatEmbeddingService(makeConfig());
    await svc.initialize();
    const results = await svc.embedBatch(["a", "b", "c"]);

    expect(results[0][0]).toBe(0.1);
    expect(results[1][0]).toBe(0.2);
    expect(results[2][0]).toBe(0.3);
  });

  it("isAvailable() returns false before init, true after", async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({ data: [{ index: 0, embedding: vec(1024) }] }),
    );
    (globalThis as any).fetch = fetchMock;

    const svc = new OpenAICompatEmbeddingService(makeConfig());
    expect(svc.isAvailable()).toBe(false);
    await svc.initialize();
    expect(svc.isAvailable()).toBe(true);
  });

  it("trailing slashes on baseURL do not double-slash the endpoint", async () => {
    const fetchMock = vi.fn(async (url: any) => {
      expect(url).toBe("https://api.openai.com/v1/embeddings");
      return jsonResponse({ data: [{ index: 0, embedding: vec(1024) }] });
    });
    (globalThis as any).fetch = fetchMock;

    const svc = new OpenAICompatEmbeddingService(
      makeConfig({
        openaiCompat: { model: "x", baseURL: "https://api.openai.com/v1///", apiKeyEnv: "TEST_OPENAI_KEY" },
      }),
    );
    await svc.initialize();
    await svc.embed("hello");
    expect(fetchMock).toHaveBeenCalled();
  });
});
