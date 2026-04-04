/**
 * ACAN — Attentive Cross-Attention Network tests.
 *
 * Tests the learned memory scorer: linear algebra primitives, weight validation,
 * inference scoring, and readiness checks.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { writeFileSync, mkdirSync, rmSync, existsSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  initACAN,
  isACANActive,
  scoreWithACAN,
  checkACANReadiness,
  type ACANWeights,
  type ACANCandidate,
} from "../src/acan.js";

// ── Constants (must match src/acan.ts) ──

const ATTN_DIM = 64;
const EMBED_DIM = 1024;
const FEATURE_COUNT = 7;

// ── Helpers ──

function makeTmpDir(): string {
  const dir = join(tmpdir(), `kongbrain-acan-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(dir, { recursive: true });
  return dir;
}

function makeValidWeights(): ACANWeights {
  const scale = Math.sqrt(2 / (EMBED_DIM + ATTN_DIM));
  return {
    W_q: Array.from({ length: EMBED_DIM }, () =>
      Array.from({ length: ATTN_DIM }, () => (Math.random() * 2 - 1) * scale),
    ),
    W_k: Array.from({ length: EMBED_DIM }, () =>
      Array.from({ length: ATTN_DIM }, () => (Math.random() * 2 - 1) * scale),
    ),
    W_final: Array.from({ length: FEATURE_COUNT }, () => Math.random() * 0.5),
    bias: 0.1,
    version: 1,
    trainedAt: Date.now(),
    trainedOnSamples: 6000,
  };
}

function makeCandidate(overrides: Partial<ACANCandidate> = {}): ACANCandidate {
  return {
    embedding: Array.from({ length: EMBED_DIM }, () => Math.random() * 0.1),
    recency: 0.9,
    importance: 0.7,
    access: 0.3,
    neighborBonus: 0,
    provenUtility: 0.5,
    reflectionBoost: 0,
    ...overrides,
  };
}

// ── Weight loading & activation ──

describe("ACAN weight loading", () => {
  let dir: string;

  beforeEach(() => {
    dir = makeTmpDir();
  });

  afterEach(() => {
    if (existsSync(dir)) rmSync(dir, { recursive: true, force: true });
  });

  it("initACAN returns false when no weights file exists", () => {
    const result = initACAN(dir);
    expect(result).toBe(false);
    expect(isACANActive()).toBe(false);
  });

  it("initACAN loads valid weights and activates", () => {
    const weights = makeValidWeights();
    writeFileSync(join(dir, "acan_weights.json"), JSON.stringify(weights));

    const result = initACAN(dir);
    expect(result).toBe(true);
    expect(isACANActive()).toBe(true);
  });

  it("rejects weights with wrong version", () => {
    const weights = makeValidWeights();
    weights.version = 99;
    writeFileSync(join(dir, "acan_weights.json"), JSON.stringify(weights));

    const result = initACAN(dir);
    expect(result).toBe(false);
  });

  it("rejects weights with wrong W_q dimensions", () => {
    const weights = makeValidWeights();
    weights.W_q = weights.W_q.slice(0, 10); // wrong outer dim
    writeFileSync(join(dir, "acan_weights.json"), JSON.stringify(weights));

    const result = initACAN(dir);
    expect(result).toBe(false);
  });

  it("rejects weights with wrong inner attention dimensions", () => {
    const weights = makeValidWeights();
    weights.W_q[0] = [1, 2, 3]; // wrong inner dim (should be ATTN_DIM=64)
    writeFileSync(join(dir, "acan_weights.json"), JSON.stringify(weights));

    const result = initACAN(dir);
    expect(result).toBe(false);
  });

  it("rejects weights with wrong W_final length", () => {
    const weights = makeValidWeights();
    weights.W_final = [1, 2]; // should be FEATURE_COUNT=7
    writeFileSync(join(dir, "acan_weights.json"), JSON.stringify(weights));

    const result = initACAN(dir);
    expect(result).toBe(false);
  });

  it("rejects weights with missing bias", () => {
    const weights = makeValidWeights();
    delete (weights as any).bias;
    writeFileSync(join(dir, "acan_weights.json"), JSON.stringify(weights));

    const result = initACAN(dir);
    expect(result).toBe(false);
  });

  it("handles corrupted JSON gracefully", () => {
    writeFileSync(join(dir, "acan_weights.json"), "not json{{{");

    const result = initACAN(dir);
    expect(result).toBe(false);
  });
});

// ── Inference (scoreWithACAN) ──

describe("scoreWithACAN", () => {
  let dir: string;

  beforeEach(() => {
    dir = makeTmpDir();
    const weights = makeValidWeights();
    writeFileSync(join(dir, "acan_weights.json"), JSON.stringify(weights));
    initACAN(dir);
  });

  afterEach(() => {
    if (existsSync(dir)) rmSync(dir, { recursive: true, force: true });
  });

  it("returns empty array for empty candidates", () => {
    const query = Array.from({ length: EMBED_DIM }, () => Math.random());
    expect(scoreWithACAN(query, [])).toEqual([]);
  });

  it("returns one score per candidate", () => {
    const query = Array.from({ length: EMBED_DIM }, () => Math.random());
    const candidates = [makeCandidate(), makeCandidate(), makeCandidate()];

    const scores = scoreWithACAN(query, candidates);
    expect(scores).toHaveLength(3);
    scores.forEach(s => expect(typeof s).toBe("number"));
  });

  it("scores are finite numbers", () => {
    const query = Array.from({ length: EMBED_DIM }, () => Math.random() * 0.1);
    const candidates = [makeCandidate()];

    const scores = scoreWithACAN(query, candidates);
    expect(scores).toHaveLength(1);
    expect(isFinite(scores[0])).toBe(true);
  });

  it("higher importance candidates tend to score higher (with identical embeddings)", () => {
    // Use deterministic weights for this test
    const weights = makeValidWeights();
    // Set W_final so importance (index 2) has high positive weight
    weights.W_final = [0.1, 0.1, 2.0, 0.1, 0.1, 0.1, 0.1];
    weights.bias = 0;
    writeFileSync(join(dir, "acan_weights.json"), JSON.stringify(weights));
    initACAN(dir);

    const query = Array.from({ length: EMBED_DIM }, () => 0.01);
    const embedding = Array.from({ length: EMBED_DIM }, () => 0.01);

    const lowImportance = makeCandidate({ embedding, importance: 0.1 });
    const highImportance = makeCandidate({ embedding, importance: 0.9 });

    const scores = scoreWithACAN(query, [lowImportance, highImportance]);
    expect(scores[1]).toBeGreaterThan(scores[0]);
  });

  it("neighbor bonus affects score when W_final[4] is positive", () => {
    const weights = makeValidWeights();
    weights.W_final = [0, 0, 0, 0, 5.0, 0, 0]; // only neighborBonus matters
    weights.bias = 0;
    writeFileSync(join(dir, "acan_weights.json"), JSON.stringify(weights));
    initACAN(dir);

    const query = Array.from({ length: EMBED_DIM }, () => 0);
    const noNeighbor = makeCandidate({ embedding: query, neighborBonus: 0 });
    const withNeighbor = makeCandidate({ embedding: query, neighborBonus: 1.0 });

    const scores = scoreWithACAN(query, [noNeighbor, withNeighbor]);
    expect(scores[1]).toBeGreaterThan(scores[0]);
  });

  it("returns empty when weights not loaded", () => {
    // Re-init with empty dir (no weights)
    const emptyDir = makeTmpDir();
    initACAN(emptyDir);
    expect(isACANActive()).toBe(false);

    const query = Array.from({ length: EMBED_DIM }, () => 0.1);
    const scores = scoreWithACAN(query, [makeCandidate()]);
    expect(scores).toEqual([]);

    rmSync(emptyDir, { recursive: true, force: true });
  });
});

// ── Readiness check ──

describe("checkACANReadiness", () => {
  it("does nothing when store is undefined", async () => {
    await expect(checkACANReadiness(undefined)).resolves.toBeUndefined();
  });

  it("does nothing when store has fewer samples than threshold", async () => {
    const store = {
      isAvailable: () => true,
      queryFirst: vi.fn(async () => [{ count: 100 }]),
    } as any;

    await checkACANReadiness(store, 5000);
    // queryFirst called once (for count), not twice (no training data fetch)
    expect(store.queryFirst).toHaveBeenCalledTimes(1);
  });

  it("skips training when weights are fresh and data hasn't grown", async () => {
    // Preload weights
    const dir = makeTmpDir();
    const weights = makeValidWeights();
    weights.trainedOnSamples = 6000;
    weights.trainedAt = Date.now(); // just trained
    writeFileSync(join(dir, "acan_weights.json"), JSON.stringify(weights));
    initACAN(dir);

    const store = {
      isAvailable: () => true,
      queryFirst: vi.fn(async () => [{ count: 6100 }]), // only 100 new samples (< 50% growth)
    } as any;

    await checkACANReadiness(store, 5000);
    // Only count query, no training data fetch
    expect(store.queryFirst).toHaveBeenCalledTimes(1);

    rmSync(dir, { recursive: true, force: true });
  });
});

// ── Linear algebra primitives (tested via scoreWithACAN behavior) ──

describe("ACAN linear algebra", () => {
  let dir: string;

  beforeEach(() => {
    dir = makeTmpDir();
  });

  afterEach(() => {
    if (existsSync(dir)) rmSync(dir, { recursive: true, force: true });
  });

  it("zero query embedding produces scores driven only by features + bias", () => {
    const weights = makeValidWeights();
    // Zero out attention weights so only features matter
    weights.W_q = Array.from({ length: EMBED_DIM }, () => new Array(ATTN_DIM).fill(0));
    weights.W_final = [0, 0.5, 0.3, 0, 0, 0, 0]; // only recency and importance
    weights.bias = 0.1;
    writeFileSync(join(dir, "acan_weights.json"), JSON.stringify(weights));
    initACAN(dir);

    const zeroQuery = new Array(EMBED_DIM).fill(0);
    const cand = makeCandidate({ recency: 0.8, importance: 0.6 });

    const scores = scoreWithACAN(zeroQuery, [cand]);
    // Score should be: 0*W_final[0] + 0.8*0.5 + 0.6*0.3 + 0 + 0 + 0 + 0 + 0.1
    // = 0 + 0.4 + 0.18 + 0.1 = 0.68
    expect(scores[0]).toBeCloseTo(0.68, 1);
  });

  it("deterministic: same inputs produce same scores", () => {
    const weights = makeValidWeights();
    writeFileSync(join(dir, "acan_weights.json"), JSON.stringify(weights));
    initACAN(dir);

    const query = Array.from({ length: EMBED_DIM }, (_, i) => Math.sin(i) * 0.1);
    const cand = makeCandidate({ embedding: Array.from({ length: EMBED_DIM }, (_, i) => Math.cos(i) * 0.1) });

    const scores1 = scoreWithACAN(query, [cand]);
    const scores2 = scoreWithACAN(query, [cand]);
    expect(scores1[0]).toBe(scores2[0]);
  });
});
