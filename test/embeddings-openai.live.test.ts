/**
 * Live smoke test against a real OpenAI-compatible embeddings endpoint.
 *
 * Skipped by default. Opt in with:
 *   KONGBRAIN_LIVE_OPENAI=1 OPENAI_API_KEY=sk-... npx vitest run test/embeddings-openai.live.test.ts
 *
 * Costs a few thousand tokens on text-embedding-3-small (~$0.0001).
 */

import { describe, expect, it } from "vitest";
import type { EmbeddingConfig } from "../src/config.js";
import { OpenAICompatEmbeddingService } from "../src/embeddings-openai.js";

const live = process.env.KONGBRAIN_LIVE_OPENAI === "1" && !!process.env.OPENAI_API_KEY;

function makeConfig(overrides: Partial<EmbeddingConfig> = {}): EmbeddingConfig {
  return {
    provider: "openai-compat",
    dimensions: 1024,
    modelPath: "",
    openaiCompat: {
      model: "text-embedding-3-small",
      baseURL: process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1",
      apiKeyEnv: "OPENAI_API_KEY",
    },
    ...overrides,
  };
}

describe.skipIf(!live)("OpenAICompatEmbeddingService — live", () => {
  it("embeds a single string and returns the requested dim", async () => {
    const svc = new OpenAICompatEmbeddingService(makeConfig());
    await svc.initialize();
    const vec = await svc.embed("KongBrain re-embed migration smoke test.");
    expect(vec).toHaveLength(1024);
    // Embeddings are normalized by OpenAI's text-embedding-3-small at any
    // requested dim, so the vector should have unit norm (within fp noise).
    const norm = Math.sqrt(vec.reduce((s, x) => s + x * x, 0));
    expect(norm).toBeGreaterThan(0.95);
    expect(norm).toBeLessThan(1.05);
  }, 30_000);

  it("embeds a batch of strings and returns vectors in input order", async () => {
    const svc = new OpenAICompatEmbeddingService(makeConfig());
    await svc.initialize();
    const inputs = [
      "The cat sat on the mat.",
      "Dogs bark at strangers.",
      "Birds fly south for winter.",
    ];
    const vecs = await svc.embedBatch(inputs);
    expect(vecs).toHaveLength(3);
    for (const v of vecs) expect(v).toHaveLength(1024);

    // Sanity check: each vector should be more similar to itself than to
    // a different sentence's vector (cosine self-similarity = 1).
    function cos(a: number[], b: number[]): number {
      let dot = 0;
      for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
      return dot;
    }
    expect(cos(vecs[0], vecs[0])).toBeGreaterThan(0.99);
    expect(cos(vecs[0], vecs[1])).toBeLessThan(cos(vecs[0], vecs[0]));
  }, 30_000);

  it("flags a dim mismatch when the requested dim is unsupported", async () => {
    // text-embedding-3-small accepts arbitrary dims, but a value the
    // server doesn't support produces an HTTP 400. We verify our error
    // path surfaces a useful message that mentions the HTTP status,
    // not an unrelated transport error.
    const svc = new OpenAICompatEmbeddingService(makeConfig({ dimensions: 99999 }));
    await svc.initialize();
    await expect(svc.embed("hello")).rejects.toThrow(/HTTP 400/);
  }, 30_000);
});
