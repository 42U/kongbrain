/**
 * End-to-end migration test: real SurrealDB + real OpenAI-compat provider
 * + real reembedAll core. Skipped by default. Opt in with:
 *
 *   KONGBRAIN_LIVE_OPENAI=1 OPENAI_API_KEY=sk-... npx vitest run test/migrate-reembed.live.test.ts
 *
 * Costs a fraction of a cent (4 small inputs through text-embedding-3-small).
 *
 * Uses a throwaway test database and drops it at the end — never touches
 * the user's production data.
 */

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { OpenAICompatEmbeddingService } from "../src/embeddings-openai.js";
import { reembedAll } from "../src/migrate-reembed.js";
import { SurrealStore } from "../src/surreal.js";

const live = process.env.KONGBRAIN_LIVE_OPENAI === "1" && !!process.env.OPENAI_API_KEY;
const TEST_NS = "kong_test";
const TEST_DB = `reembed_e2e_${Date.now()}`;

let store: SurrealStore;
let embeddings: OpenAICompatEmbeddingService;
const FROM_PROVIDER = "fake-old-provider-e2e";

beforeAll(async () => {
  if (!live) return;
  const url = process.env.SURREAL_URL ?? "ws://127.0.0.1:8000/rpc";
  store = new SurrealStore({
    url,
    get httpUrl() { return url.replace("ws://", "http://").replace("wss://", "https://").replace("/rpc", ""); },
    user: process.env.SURREAL_USER ?? "root",
    pass: process.env.SURREAL_PASS ?? "root",
    ns: TEST_NS,
    db: TEST_DB,
  });
  try {
    await Promise.race([
      store.initialize(),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error("SurrealDB connection timed out after 10s")), 10_000),
      ),
    ]);
  } catch {
    store = undefined as any;
    return;
  }

  embeddings = new OpenAICompatEmbeddingService({
    provider: "openai-compat",
    dimensions: 1024,
    modelPath: "",
    openaiCompat: {
      model: "text-embedding-3-small",
      baseURL: "https://api.openai.com/v1",
      apiKeyEnv: "OPENAI_API_KEY",
    },
  });
  await embeddings.initialize();
  store.setActiveProvider(embeddings.providerId);

  // Seed: 4 memory rows tagged with the FAKE old provider so we can watch
  // them migrate. Embeddings are zero-filled placeholders (different
  // vector space than openai-compat would produce).
  const placeholderVec = new Array(1024).fill(0.001);
  const seedRows = [
    { text: "The cat sat on the mat.", category: "general" },
    { text: "Database migrations should be idempotent.", category: "engineering" },
    { text: "Embeddings live in vector spaces, never mix them.", category: "engineering" },
    { text: "Beavers build dams to slow water flow.", category: "general" },
  ];
  for (const row of seedRows) {
    await store.queryExec(
      `CREATE memory CONTENT $r`,
      {
        r: {
          ...row,
          importance: 5,
          source: row.category,
          embedding: placeholderVec,
          embedding_provider: FROM_PROVIDER,
        },
      },
    );
  }
}, 30_000);

afterAll(async () => {
  if (!store) return;
  try { await store.queryExec(`REMOVE DATABASE ${TEST_DB}`); } catch { /* ok */ }
  try { await store.close(); } catch { /* ok */ }
  try { await embeddings.dispose(); } catch { /* ok */ }
}, 15_000);

describe.skipIf(!live)("reembedAll — live (real DB + real OpenAI)", () => {
  it("dry-run reports the seeded rows without writing", async () => {
    if (!store) throw new Error("SurrealDB not available");
    const result = await reembedAll(store, embeddings, {
      fromProvider: FROM_PROVIDER,
      tables: ["memory"],
      dryRun: true,
    });
    expect(result.dryRun).toBe(true);
    expect(result.perTable.memory).toBe(4);
    expect(result.approxTokens).toBeGreaterThan(0);

    // Confirm the rows are still tagged with the OLD provider after dry-run.
    const rows = await store.queryFirst<{ embedding_provider: string }>(
      `SELECT embedding_provider FROM memory WHERE embedding_provider = $p`,
      { p: FROM_PROVIDER },
    );
    expect(rows).toHaveLength(4);
  }, 30_000);

  it("real run flips every row's tag and replaces the embedding with one in the new space", async () => {
    if (!store) throw new Error("SurrealDB not available");

    const result = await reembedAll(store, embeddings, {
      fromProvider: FROM_PROVIDER,
      tables: ["memory"],
    });

    expect(result.dryRun).toBe(false);
    expect(result.perTable.memory).toBe(4);

    // No rows should still be tagged with the old provider.
    const stillOld = await store.queryFirst<{ id: string }>(
      `SELECT id FROM memory WHERE embedding_provider = $p`,
      { p: FROM_PROVIDER },
    );
    expect(stillOld).toHaveLength(0);

    // All rows should be tagged with the OpenAI provider id, with a
    // 1024-dim embedding that is no longer the placeholder.
    const migrated = await store.queryFirst<{ embedding: number[]; embedding_provider: string }>(
      `SELECT embedding, embedding_provider FROM memory WHERE embedding_provider = $p`,
      { p: embeddings.providerId },
    );
    expect(migrated).toHaveLength(4);
    for (const row of migrated) {
      expect(row.embedding).toHaveLength(1024);
      // Placeholder was 0.001 everywhere — real embedding will not be.
      const allPlaceholder = row.embedding.every(v => Math.abs(v - 0.001) < 1e-6);
      expect(allPlaceholder).toBe(false);
    }
  }, 60_000);

  it("re-running on a clean DB is a no-op", async () => {
    if (!store) throw new Error("SurrealDB not available");

    const result = await reembedAll(store, embeddings, {
      fromProvider: FROM_PROVIDER,
      tables: ["memory"],
    });
    expect(result.total).toBe(0);
  }, 30_000);
});
