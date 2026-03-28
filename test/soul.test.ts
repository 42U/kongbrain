import { describe, it, expect } from "vitest";
import { checkGraduation } from "../src/soul.js";

// Mock SurrealStore that returns configurable signal counts
function mockStore(signals: Partial<{
  sessions: number;
  reflections: number;
  causalChains: number;
  concepts: number;
  compactions: number;
  monologues: number;
  spanDays: number;
}> = {}) {
  const earliest = signals.spanDays
    ? new Date(Date.now() - signals.spanDays * 86400000).toISOString()
    : undefined;

  return {
    isAvailable: () => true,
    queryFirst: async (sql: string) => {
      if (sql.includes("FROM session GROUP ALL")) return [{ count: signals.sessions ?? 0 }];
      if (sql.includes("FROM reflection GROUP ALL")) return [{ count: signals.reflections ?? 0 }];
      if (sql.includes("FROM causal_chain GROUP ALL")) return [{ count: signals.causalChains ?? 0 }];
      if (sql.includes("FROM concept GROUP ALL")) return [{ count: signals.concepts ?? 0 }];
      if (sql.includes("FROM compaction_checkpoint")) return [{ count: signals.compactions ?? 0 }];
      if (sql.includes("FROM monologue GROUP ALL")) return [{ count: signals.monologues ?? 0 }];
      if (sql.includes("FROM session ORDER BY started_at")) return earliest ? [{ earliest }] : [];
      return [];
    },
  };
}

describe("checkGraduation", () => {
  it("not ready with zero signals", async () => {
    const result = await checkGraduation(mockStore() as any);
    expect(result.ready).toBe(false);
    expect(result.score).toBe(0);
    expect(result.unmet.length).toBe(7);
    expect(result.met.length).toBe(0);
  });

  it("not ready when unavailable", async () => {
    const store = { isAvailable: () => false, queryFirst: async () => [] };
    const result = await checkGraduation(store as any);
    expect(result.ready).toBe(false);
    expect(result.score).toBe(0);
  });

  it("ready when 5 of 7 thresholds met", async () => {
    const result = await checkGraduation(mockStore({
      sessions: 20,
      reflections: 15,
      causalChains: 10,
      concepts: 50,
      compactions: 8,
      monologues: 2,    // below threshold (5)
      spanDays: 1,      // below threshold (3)
    }) as any);

    expect(result.ready).toBe(true);
    expect(result.met.length).toBe(5);
    expect(result.unmet.length).toBe(2);
    expect(result.score).toBeCloseTo(5 / 7);
  });

  it("not ready when only 4 thresholds met", async () => {
    const result = await checkGraduation(mockStore({
      sessions: 20,
      reflections: 15,
      causalChains: 10,
      concepts: 50,
      compactions: 0,
      monologues: 0,
      spanDays: 0,
    }) as any);

    expect(result.ready).toBe(false);
    expect(result.met.length).toBe(4);
  });

  it("ready when all thresholds met", async () => {
    const result = await checkGraduation(mockStore({
      sessions: 20,
      reflections: 15,
      causalChains: 10,
      concepts: 50,
      compactions: 8,
      monologues: 10,
      spanDays: 7,
    }) as any);

    expect(result.ready).toBe(true);
    expect(result.score).toBe(1);
    expect(result.unmet.length).toBe(0);
  });

  it("reports exact threshold values in met/unmet strings", async () => {
    const result = await checkGraduation(mockStore({
      sessions: 5,
      concepts: 50,
    }) as any);

    expect(result.unmet.some(s => s.includes("sessions: 5/15"))).toBe(true);
    expect(result.met.some(s => s.includes("concepts: 50/30"))).toBe(true);
  });
});
