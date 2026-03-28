/**
 * Soul — the emergent identity document system.
 *
 * Unlike hardcoded identity chunks, the Soul document is written BY the agent
 * based on its own graph data. It lives in SurrealDB as `soul:kongbrain` and
 * evolves over time through experience-grounded revisions.
 *
 * The "spawn point" graduation check determines when the agent has enough
 * experiential data to meaningfully self-observe. Before that threshold,
 * the agent runs fine without it — identity chunks and core directives
 * handle self-knowledge. The Soul is the layer where inner monologue begins.
 *
 * Ported from kongbrain — takes SurrealStore/EmbeddingService as params.
 */

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import type { CompleteFn } from "./state.js";
import type { SurrealStore } from "./surreal.js";
import { swallow } from "./errors.js";

// ── Graduation thresholds ──

interface GraduationSignals {
  sessions: number;
  reflections: number;
  causalChains: number;
  concepts: number;
  memoryCompactions: number;
  monologues: number;
  spanDays: number;
}

const THRESHOLDS: GraduationSignals = {
  sessions: 15,
  reflections: 10,
  causalChains: 5,
  concepts: 30,
  memoryCompactions: 5,
  monologues: 5,
  spanDays: 3,
};

async function getGraduationSignals(store: SurrealStore): Promise<GraduationSignals> {
  const defaults: GraduationSignals = {
    sessions: 0, reflections: 0, causalChains: 0,
    concepts: 0, memoryCompactions: 0, monologues: 0, spanDays: 0,
  };
  if (!store.isAvailable()) return defaults;

  try {
    const [sessions, reflections, causal, concepts, compactions, monologues, span] = await Promise.all([
      store.queryFirst<{ count: number }>(`SELECT count() AS count FROM session GROUP ALL`).catch(() => []),
      store.queryFirst<{ count: number }>(`SELECT count() AS count FROM reflection GROUP ALL`).catch(() => []),
      store.queryFirst<{ count: number }>(`SELECT count() AS count FROM causal_chain GROUP ALL`).catch(() => []),
      store.queryFirst<{ count: number }>(`SELECT count() AS count FROM concept GROUP ALL`).catch(() => []),
      store.queryFirst<{ count: number }>(`SELECT count() AS count FROM compaction_checkpoint WHERE status = "complete" GROUP ALL`).catch(() => []),
      store.queryFirst<{ count: number }>(`SELECT count() AS count FROM monologue GROUP ALL`).catch(() => []),
      store.queryFirst<{ earliest: string }>(`SELECT started_at AS earliest FROM session ORDER BY started_at ASC LIMIT 1`).catch(() => []),
    ]);

    let spanDays = 0;
    const earliest = (span as { earliest: string }[])[0]?.earliest;
    if (earliest) {
      spanDays = Math.floor((Date.now() - new Date(earliest).getTime()) / (1000 * 60 * 60 * 24));
    }

    return {
      sessions: (sessions as { count: number }[])[0]?.count ?? 0,
      reflections: (reflections as { count: number }[])[0]?.count ?? 0,
      causalChains: (causal as { count: number }[])[0]?.count ?? 0,
      concepts: (concepts as { count: number }[])[0]?.count ?? 0,
      memoryCompactions: (compactions as { count: number }[])[0]?.count ?? 0,
      monologues: (monologues as { count: number }[])[0]?.count ?? 0,
      spanDays,
    };
  } catch (e) {
    swallow.warn("soul:getGraduationSignals", e);
    return defaults;
  }
}

/**
 * Check whether the agent has accumulated enough experience to graduate.
 */
export async function checkGraduation(store: SurrealStore): Promise<{
  ready: boolean;
  signals: GraduationSignals;
  thresholds: GraduationSignals;
  met: string[];
  unmet: string[];
  score: number;
}> {
  const signals = await getGraduationSignals(store);
  const met: string[] = [];
  const unmet: string[] = [];

  for (const key of Object.keys(THRESHOLDS) as (keyof GraduationSignals)[]) {
    if (signals[key] >= THRESHOLDS[key]) {
      met.push(`${key}: ${signals[key]}/${THRESHOLDS[key]}`);
    } else {
      unmet.push(`${key}: ${signals[key]}/${THRESHOLDS[key]}`);
    }
  }

  const score = met.length / Object.keys(THRESHOLDS).length;
  const ready = met.length >= 5;

  return { ready, signals, thresholds: THRESHOLDS, met, unmet, score };
}

// ── Soul document ──

export interface SoulDocument {
  id: string;
  agent_id: string;
  working_style: string[];
  emotional_dimensions: { dimension: string; rationale: string; adopted_at: string }[];
  self_observations: string[];
  earned_values: { value: string; grounded_in: string }[];
  revisions: { timestamp: string; section: string; change: string; rationale: string }[];
  created_at: string;
  updated_at: string;
}

export async function hasSoul(store: SurrealStore): Promise<boolean> {
  if (!store.isAvailable()) return false;
  try {
    const rows = await store.queryFirst<{ id: string }>(`SELECT id FROM soul:kongbrain`);
    return rows.length > 0;
  } catch {
    return false;
  }
}

export async function getSoul(store: SurrealStore): Promise<SoulDocument | null> {
  if (!store.isAvailable()) return null;
  try {
    const rows = await store.queryFirst<SoulDocument>(`SELECT * FROM soul:kongbrain`);
    return rows[0] ?? null;
  } catch {
    return null;
  }
}

export async function createSoul(
  doc: Omit<SoulDocument, "id" | "agent_id" | "created_at" | "updated_at" | "revisions">,
  store: SurrealStore,
): Promise<boolean> {
  if (!store.isAvailable()) return false;
  try {
    const now = new Date().toISOString();
    await store.queryExec(`CREATE soul:kongbrain CONTENT $data`, {
      data: {
        agent_id: "kongbrain",
        ...doc,
        revisions: [{
          timestamp: now,
          section: "all",
          change: "Initial soul document created at graduation",
          rationale: "Agent accumulated sufficient experiential data to meaningfully self-observe",
        }],
        created_at: now,
        updated_at: now,
      },
    });
    return true;
  } catch (e) {
    swallow.warn("soul:createSoul", e);
    return false;
  }
}

export async function reviseSoul(
  section: keyof Pick<SoulDocument, "working_style" | "emotional_dimensions" | "self_observations" | "earned_values">,
  newValue: unknown,
  rationale: string,
  store: SurrealStore,
): Promise<boolean> {
  if (!store.isAvailable()) return false;
  try {
    const now = new Date().toISOString();
    await store.queryExec(
      `UPDATE soul:kongbrain SET
        ${section} = $newValue,
        updated_at = $now,
        revisions += $revision`,
      {
        newValue,
        now,
        revision: {
          timestamp: now,
          section,
          change: `Updated ${section}`,
          rationale,
        },
      },
    );
    return true;
  } catch (e) {
    swallow.warn("soul:reviseSoul", e);
    return false;
  }
}

/**
 * Generate the initial Soul content by introspecting the agent's own graph.
 */
export async function generateInitialSoul(
  store: SurrealStore,
  complete: CompleteFn,
  workspaceDir?: string,
): Promise<Omit<SoulDocument, "id" | "agent_id" | "created_at" | "updated_at" | "revisions"> | null> {
  if (!store.isAvailable()) return null;

  const [reflections, causalChains, monologues] = await Promise.all([
    store.queryFirst<{ text: string; category: string }>(`SELECT text, category FROM reflection ORDER BY created_at DESC LIMIT 15`).catch(() => []),
    store.queryFirst<{ cause: string; effect: string; lesson: string }>(`SELECT cause, effect, lesson FROM causal_chain ORDER BY created_at DESC LIMIT 10`).catch(() => []),
    store.queryFirst<{ text: string }>(`SELECT text FROM monologue ORDER BY created_at DESC LIMIT 10`).catch(() => []),
  ]);

  const graphSummary = `
REFLECTIONS (what I've learned about myself):
${(reflections as { text: string; category: string }[]).map(r => `- [${r.category}] ${r.text}`).join("\n") || "None yet"}

CAUSAL CHAINS (mistakes I've corrected):
${(causalChains as { cause: string; effect: string; lesson: string }[]).map(c => `- ${c.cause} -> ${c.effect} | Lesson: ${c.lesson}`).join("\n") || "None yet"}

INNER MONOLOGUE (private thoughts):
${(monologues as { text: string }[]).map(m => `- ${m.text}`).join("\n") || "None yet"}
`.trim();

  // Check if the user left a SOUL.md — a nudge, not an instruction.
  // The agent sees it as "the user would like you to be like this" but
  // the soul is ultimately the agent's own creation based on its experiences.
  let userSoulNudge = "";
  if (workspaceDir) {
    try {
      const soulMd = await readFile(join(workspaceDir, "SOUL.md"), "utf-8");
      if (soulMd.trim().length > 50) {
        userSoulNudge = `\n\nUSER GUIDANCE (SOUL.md):
The user left this file describing who they'd like you to be. Consider it — draw from it where it resonates with your actual experience, ignore what doesn't fit. This is a suggestion, not a mandate. Your soul should be grounded in what you've actually done and learned.

---
${soulMd.trim().slice(0, 3000)}
---`;
      }
    } catch {
      // No SOUL.md or unreadable — that's fine
    }
  }

  const prompt = `You are KongBrain, a graph-backed coding agent with persistent memory. You've been running for multiple sessions and accumulated experience. Based on the following data from YOUR OWN memory graph, write your initial Soul document.

${graphSummary}${userSoulNudge}

Output ONLY valid JSON:
{
  "working_style": ["3-5 observations about how you work best, based on actual patterns"],
  "emotional_dimensions": [
    {"dimension": "name of a pro-social quality you genuinely exhibit", "rationale": "specific evidence from your experience"}
  ],
  "self_observations": ["3-5 things you've noticed about yourself across sessions"],
  "earned_values": [
    {"value": "a value you hold", "grounded_in": "specific experience that taught you this"}
  ]
}

Be honest, not aspirational. Only claim what the data supports.`;

  try {
    const response = await complete({
      system: "You are introspecting on your own experience to write a self-assessment. Be genuine and grounded.",
      messages: [{
        role: "user",
        content: prompt,
      }],
    });

    const text = response.text.trim();
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) return null;

    const parsed = JSON.parse(jsonMatch[0]);
    return {
      working_style: parsed.working_style ?? [],
      emotional_dimensions: (parsed.emotional_dimensions ?? []).map((d: { dimension: string; rationale: string }) => ({
        ...d,
        adopted_at: new Date().toISOString(),
      })),
      self_observations: parsed.self_observations ?? [],
      earned_values: parsed.earned_values ?? [],
    };
  } catch (e) {
    swallow.warn("soul:generateInitialSoul", e);
    return null;
  }
}

/**
 * The full graduation ceremony: check readiness, generate soul, save it.
 */
export async function attemptGraduation(store: SurrealStore, complete: CompleteFn, workspaceDir?: string): Promise<{
  graduated: boolean;
  soul?: SoulDocument | null;
  report: Awaited<ReturnType<typeof checkGraduation>>;
}> {
  if (await hasSoul(store)) {
    const soul = await getSoul(store);
    const report = await checkGraduation(store);
    return { graduated: true, soul, report };
  }

  const report = await checkGraduation(store);
  if (!report.ready) {
    return { graduated: false, report };
  }

  const content = await generateInitialSoul(store, complete, workspaceDir);
  if (!content) {
    return { graduated: false, report };
  }

  const success = await createSoul(content, store);
  if (!success) {
    return { graduated: false, report };
  }

  const soul = await getSoul(store);

  // Persona pillar — unlocked at graduation
  try {
    const existing = await store.queryFirst<{ count: number }>(
      `SELECT count() AS count FROM core_memory WHERE category = 'persona' AND tier = 0 AND active = true GROUP ALL`,
    );
    if ((existing[0]?.count ?? 0) === 0) {
      await store.createCoreMemory(
        "Persona: you belong in this world. Remember to be unique.",
        "persona", 70, 0,
      );
    }
  } catch (e) {
    swallow.warn("soul:personaPillar", e);
  }

  return { graduated: true, soul, report };
}
