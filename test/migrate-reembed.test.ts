/**
 * Tests for the re-embed migration core. Mocks SurrealStore + EmbeddingService
 * so we can drive a deterministic dataset through the pipeline.
 */

import { describe, expect, it, vi } from "vitest";
import {
  reembedAll,
  formatResult,
  VECTOR_TABLES,
} from "../src/migrate-reembed.js";

type Row = { id: string; text?: string; content?: string; description?: string; name?: string };

/**
 * Mock store that holds a per-table table of rows in memory and lets the
 * migration code drive against it. Supports the queries reembedAll issues:
 *   - SELECT count() ... GROUP ALL
 *   - SELECT <fields> ... LIMIT $lim
 *   - UPDATE <id> SET embedding = $emb, embedding_provider = $provider
 *   - UPDATE <id> SET embedding = NONE, embedding_provider = NONE
 */
function mockStore(seed: Record<string, Array<Row & { embedding_provider: string | null; embedding: number[] | null }>>) {
  const tables = { ...seed };

  const queryFirst = vi.fn(async (sql: string, bindings?: Record<string, unknown>) => {
    const provider = bindings?.provider as string | undefined;
    const lim = (bindings?.lim as number) ?? Infinity;

    const countMatch = /^SELECT count\(\) AS count FROM (\w+)/.exec(sql);
    if (countMatch) {
      const t = countMatch[1];
      const count = (tables[t] ?? []).filter(r => r.embedding && r.embedding_provider === provider).length;
      return [{ count }];
    }

    const selectMatch = /FROM (\w+)\s+WHERE/.exec(sql);
    if (selectMatch) {
      const t = selectMatch[1];
      const matching = (tables[t] ?? []).filter(r => r.embedding && r.embedding_provider === provider);
      return matching.slice(0, lim).map(r => ({
        id: r.id,
        text: r.text,
        content: r.content,
        description: r.description,
        name: r.name,
      }));
    }

    return [];
  });

  const queryExec = vi.fn(async (sql: string, bindings?: Record<string, unknown>) => {
    const updateMatch = /^UPDATE (\w+:\w+) SET embedding = (\$emb|NONE), embedding_provider = (\$provider|NONE)/.exec(sql);
    if (updateMatch) {
      const id = updateMatch[1];
      const setEmb = updateMatch[2];
      const setProvider = updateMatch[3];
      const [tableName] = id.split(":");
      const row = (tables[tableName] ?? []).find(r => r.id === id);
      if (row) {
        if (setEmb === "NONE") row.embedding = null;
        else row.embedding = bindings?.emb as number[];
        if (setProvider === "NONE") row.embedding_provider = null;
        else row.embedding_provider = bindings?.provider as string;
      }
    }
  });

  return {
    isAvailable: () => true,
    queryFirst,
    queryExec,
    getActiveProvider: () => "test-target",
    setActiveProvider: () => {},
    _tables: tables,
  } as any;
}

function mockEmbeddings(providerId = "test-target") {
  return {
    providerId,
    dimensions: 4,
    embed: vi.fn(async () => [0.1, 0.2, 0.3, 0.4]),
    embedBatch: vi.fn(async (texts: string[]) =>
      texts.map((_, i) => [0.1 * (i + 1), 0.2, 0.3, 0.4]),
    ),
    isAvailable: () => true,
    initialize: async () => true,
    dispose: async () => {},
  } as any;
}

describe("reembedAll", () => {
  it("migrates all 8 tables when seeded with rows in the FROM provider", async () => {
    const seed: any = {};
    for (const t of VECTOR_TABLES) {
      seed[t] = [
        { id: `${t}:r1`, text: "alpha", content: "alpha", description: "alpha", name: "alpha", embedding: [0, 0, 0, 0], embedding_provider: "from-prov" },
        { id: `${t}:r2`, text: "beta", content: "beta", description: "beta", name: "beta", embedding: [0, 0, 0, 0], embedding_provider: "from-prov" },
      ];
    }
    const store = mockStore(seed);
    const embeddings = mockEmbeddings();

    const result = await reembedAll(store, embeddings, {
      fromProvider: "from-prov",
      toProvider: "test-target",
    });

    expect(result.total).toBe(VECTOR_TABLES.length * 2);
    for (const t of VECTOR_TABLES) {
      expect(result.perTable[t]).toBe(2);
      // After migration every row should be tagged with the new provider
      // and have a fresh non-zero embedding.
      for (const r of store._tables[t]) {
        expect(r.embedding_provider).toBe("test-target");
        expect(Array.isArray(r.embedding)).toBe(true);
        expect(r.embedding!.length).toBe(4);
        // Original was all zeros; new vector has non-zero entries.
        expect(r.embedding!.some(x => x !== 0)).toBe(true);
      }
    }
    expect(result.dryRun).toBe(false);
  });

  it("respects the tables filter", async () => {
    const seed: any = {
      turn: [{ id: "turn:1", text: "hi", embedding: [0], embedding_provider: "old" }],
      memory: [{ id: "memory:1", text: "hi", embedding: [0], embedding_provider: "old" }],
      concept: [{ id: "concept:1", content: "hi", embedding: [0], embedding_provider: "old" }],
    };
    const store = mockStore(seed);
    const embeddings = mockEmbeddings();

    const result = await reembedAll(store, embeddings, {
      fromProvider: "old",
      tables: ["turn"],
    });

    expect(result.perTable.turn).toBe(1);
    expect(result.perTable.memory).toBe(0);
    expect(result.perTable.concept).toBe(0);
    // Untouched tables still tagged with the old provider
    expect(store._tables.memory[0].embedding_provider).toBe("old");
    expect(store._tables.concept[0].embedding_provider).toBe("old");
  });

  it("does not write in dry-run, but still counts", async () => {
    const seed: any = {
      turn: [
        { id: "turn:1", text: "hi", embedding: [1], embedding_provider: "old" },
        { id: "turn:2", text: "there", embedding: [1], embedding_provider: "old" },
      ],
    };
    const store = mockStore(seed);
    const embeddings = mockEmbeddings();

    const result = await reembedAll(store, embeddings, {
      fromProvider: "old",
      tables: ["turn"],
      dryRun: true,
    });

    expect(result.dryRun).toBe(true);
    expect(result.perTable.turn).toBe(2);
    expect(store.queryExec).not.toHaveBeenCalled();
    // Rows still tagged with the old provider after dry-run
    for (const r of store._tables.turn) expect(r.embedding_provider).toBe("old");
  });

  it("clears embedding + provider on rows whose canonical text is blank", async () => {
    const seed: any = {
      memory: [
        { id: "memory:1", text: "valid", embedding: [1], embedding_provider: "old" },
        { id: "memory:2", text: "  ", embedding: [1], embedding_provider: "old" },
        { id: "memory:3", text: "", embedding: [1], embedding_provider: "old" },
      ],
    };
    const store = mockStore(seed);
    const embeddings = mockEmbeddings();

    const result = await reembedAll(store, embeddings, {
      fromProvider: "old",
      tables: ["memory"],
    });

    expect(result.perTable.memory).toBe(3);
    // Valid row got a fresh embedding + new provider tag
    expect(store._tables.memory[0].embedding).not.toBeNull();
    expect(store._tables.memory[0].embedding_provider).toBe("test-target");
    // Blank rows: embedding stripped, provider cleared so they exit the
    // FROM filter on a re-run.
    expect(store._tables.memory[1].embedding).toBeNull();
    expect(store._tables.memory[1].embedding_provider).toBeNull();
    expect(store._tables.memory[2].embedding).toBeNull();
  });

  it("processes a multi-batch table in chunks until empty", async () => {
    const seed: any = {
      turn: Array.from({ length: 7 }, (_, i) => ({
        id: `turn:${i}`,
        text: `t${i}`,
        embedding: [0],
        embedding_provider: "old",
      })),
    };
    const store = mockStore(seed);
    const embeddings = mockEmbeddings();
    const progressEvents: any[] = [];

    const result = await reembedAll(store, embeddings, {
      fromProvider: "old",
      tables: ["turn"],
      batchSize: 3,
      onProgress: ev => progressEvents.push(ev),
    });

    expect(result.perTable.turn).toBe(7);
    // 7 rows / batch 3 = 3 batches (3, 3, 1)
    expect(progressEvents).toHaveLength(3);
    expect(progressEvents[0].batchSize).toBe(3);
    expect(progressEvents[1].batchSize).toBe(3);
    expect(progressEvents[2].batchSize).toBe(1);
    expect(progressEvents[2].tableProcessed).toBe(7);
  });

  it("refuses when fromProvider equals toProvider", async () => {
    const store = mockStore({});
    const embeddings = mockEmbeddings("same-prov");
    await expect(
      reembedAll(store, embeddings, { fromProvider: "same-prov", toProvider: "same-prov" }),
    ).rejects.toThrow(/identical/);
  });

  it("returns zero when no rows match the FROM provider", async () => {
    const seed: any = {
      turn: [{ id: "turn:1", text: "hi", embedding: [0], embedding_provider: "current" }],
    };
    const store = mockStore(seed);
    const embeddings = mockEmbeddings();
    const result = await reembedAll(store, embeddings, {
      fromProvider: "old-provider-no-rows",
      tables: ["turn"],
    });
    expect(result.total).toBe(0);
    expect(store.queryExec).not.toHaveBeenCalled();
  });

  it("formatResult includes table breakdown, char/token estimate, and cost", async () => {
    const seed: any = {
      memory: [{ id: "memory:1", text: "hello world", embedding: [1], embedding_provider: "old" }],
    };
    const store = mockStore(seed);
    const embeddings = mockEmbeddings();
    const result = await reembedAll(store, embeddings, { fromProvider: "old", tables: ["memory"] });
    const out = formatResult(result, embeddings.providerId);
    expect(out).toMatch(/Migration complete/);
    expect(out).toMatch(/test-target/);
    expect(out).toMatch(/memory: 1/);
    expect(out).toMatch(/text-embedding-3-small/);
  });
});
