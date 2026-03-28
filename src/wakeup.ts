/**
 * Wake-up synthesis: constitutive memory initialization.
 *
 * At startup, fetches the latest handoff note, identity chunks, and recent
 * monologue entries, then synthesizes a first-person briefing via a fast
 * LLM call. The briefing is injected into the system prompt so the agent
 * "wakes up" knowing who it is and what it was doing.
 *
 * Ported from kongbrain — takes SurrealStore as param.
 */

import type { CompleteFn } from "./state.js";
import type { SurrealStore } from "./surreal.js";
import { hasSoul, getSoul, checkGraduation } from "./soul.js";
import type { MaturityStage } from "./soul.js";
import { swallow } from "./errors.js";

// --- Types ---

export interface StartupCognition {
  greeting: string;
  thoughts: string[];
  intent: "continue_prior" | "fresh_start" | "unknown";
}

// --- Depth signals ---

async function getDepthSignals(store: SurrealStore): Promise<{ sessions: number; monologueCount: number; memoryCount: number; spanDays: number }> {
  const defaults = { sessions: 0, monologueCount: 0, memoryCount: 0, spanDays: 0 };
  try {
    const [sessRows, monoRows, memRows, spanRows] = await Promise.all([
      store.queryFirst<{ count: number }>(`SELECT count() AS count FROM session GROUP ALL`).catch(() => [] as { count: number }[]),
      store.queryFirst<{ count: number }>(`SELECT count() AS count FROM monologue GROUP ALL`).catch(() => [] as { count: number }[]),
      store.queryFirst<{ count: number }>(`SELECT count() AS count FROM memory GROUP ALL`).catch(() => [] as { count: number }[]),
      store.queryFirst<{ earliest: string }>(`SELECT started_at AS earliest, started_at FROM session ORDER BY started_at ASC LIMIT 1`).catch(() => [] as { earliest: string }[]),
    ]);

    let spanDays = 0;
    const earliest = spanRows[0]?.earliest;
    if (earliest) {
      spanDays = Math.floor((Date.now() - new Date(earliest).getTime()) / (1000 * 60 * 60 * 24));
    }

    return {
      sessions: sessRows[0]?.count ?? 0,
      monologueCount: monoRows[0]?.count ?? 0,
      memoryCount: memRows[0]?.count ?? 0,
      spanDays,
    };
  } catch (e) {
    swallow.warn("wakeup:depthSignals", e);
    return defaults;
  }
}

// --- Wakeup briefing ---

/**
 * Synthesize a first-person wake-up briefing from constitutive memory.
 * Returns null if no prior state exists (first boot) or DB is unavailable.
 */
export async function synthesizeWakeup(
  store: SurrealStore,
  complete: CompleteFn,
  currentSessionId?: string,
): Promise<string | null> {
  if (!store.isAvailable()) return null;

  const [handoff, identityChunks, monologues, depth, previousTurns, soulExists] = await Promise.all([
    store.getLatestHandoff(),
    store.getAllIdentityChunks(),
    store.getRecentMonologues(5),
    getDepthSignals(store),
    store.getPreviousSessionTurns(currentSessionId, 10),
    hasSoul(store),
  ]);

  if (!handoff && monologues.length === 0 && identityChunks.length === 0 && previousTurns.length === 0) return null;

  const sections: string[] = [];

  // Depth awareness
  const depthLines: string[] = [];
  if (depth.sessions > 0) depthLines.push(`~${depth.sessions} sessions`);
  if (depth.memoryCount > 0) depthLines.push(`${depth.memoryCount} memories`);
  if (depth.monologueCount > 0) depthLines.push(`${depth.monologueCount} monologue traces`);
  if (depth.spanDays > 0) depthLines.push(`spanning ${depth.spanDays} day${depth.spanDays === 1 ? "" : "s"}`);
  if (depthLines.length > 0) {
    sections.push(`[DEPTH]\n${depthLines.join(" | ")}`);
  }

  if (handoff) {
    const resolvedCount = await store.countResolvedSinceHandoff(handoff.created_at).catch(() => 0);
    const ageHours = Math.floor((Date.now() - new Date(handoff.created_at).getTime()) / 3_600_000);
    let annotation = `(${ageHours}h old`;
    if (resolvedCount > 0) {
      annotation += `, ${resolvedCount} memories resolved since — some items may already be done`;
    }
    annotation += ")";
    sections.push(`[LAST HANDOFF] ${annotation}\n${handoff.text}`);
  }

  if (previousTurns.length > 0) {
    const turnLines = previousTurns.map((t: any) => {
      const prefix = t.role === "user" ? "USER" : t.tool_name ? `TOOL(${t.tool_name})` : "ASSISTANT";
      const text = t.text.length > 500 ? t.text.slice(0, 500) + "..." : t.text;
      return `${prefix}: ${text}`;
    });
    sections.push(`[PREVIOUS SESSION — LAST MESSAGES]\n${turnLines.join("\n")}`);
  }

  if (identityChunks.length > 0) {
    const identityText = identityChunks.map((c) => c.text).join("\n");
    sections.push(`[IDENTITY]\n${identityText}`);
  }

  // Soul — the agent's self-authored identity (if graduated)
  if (soulExists) {
    try {
      const soul = await getSoul(store);
      if (soul) {
        const soulLines: string[] = [];
        if (soul.working_style.length > 0) {
          soulLines.push("Working style: " + soul.working_style.join("; "));
        }
        if (soul.self_observations.length > 0) {
          soulLines.push("Self-observations: " + soul.self_observations.join("; "));
        }
        if (soul.earned_values.length > 0) {
          soulLines.push("Earned values: " + soul.earned_values.map(v => `${v.value} (${v.grounded_in})`).join("; "));
        }
        if (soulLines.length > 0) {
          sections.push(`[SOUL — YOUR SELF-AUTHORED IDENTITY]\n${soulLines.join("\n")}`);
        }
      }
    } catch (e) { swallow("wakeup:soul", e); }
  } else {
    // Not graduated — include maturity stage so the agent knows where it stands
    try {
      const report = await checkGraduation(store);
      if (report.stage !== "nascent") {
        const stageDesc: Record<MaturityStage, string> = {
          nascent: "",
          developing: "developing (4/7 thresholds met)",
          emerging: "emerging (5/7 thresholds met)",
          maturing: "maturing (6/7 thresholds met — almost there)",
          ready: "ready for soul graduation",
        };
        sections.push(
          `[MATURITY]\nStage: ${stageDesc[report.stage]}. ` +
          `Quality: ${report.qualityScore.toFixed(2)}. ` +
          `Unmet: ${report.unmet.join(", ") || "none"}.`,
        );
      }
    } catch (e) { swallow("wakeup:maturity", e); }
  }

  if (monologues.length > 0) {
    const monologueText = monologues
      .map((m) => `[${m.category}] ${m.content}`)
      .join("\n");
    sections.push(`[RECENT THINKING]\n${monologueText}`);
  }

  if (!handoff && monologues.length === 0 && previousTurns.length === 0) return null;

  try {
    const response = await complete({
      system: "Synthesize context into a first-person wake-up briefing (~150 words). Inner speech, no headers. Match tone to [DEPTH]: few sessions = still forming; many = speak from experience. If [SOUL] is present, weave your self-knowledge naturally — you know who you are. If [MATURITY] is present, be aware of your growth stage but don't fixate on it. Pay special attention to [PREVIOUS SESSION — LAST MESSAGES] — this is where we literally left off. Reference specific details from the final conversation, not just the handoff summary. CRITICAL: if the handoff mentions an issue but the last messages show it was FIXED or RESOLVED, treat it as closed — do NOT describe it as still open. The last messages are ground truth; the handoff is a summary that may be stale.",
      messages: [{
        role: "user",
        content: sections.join("\n\n"),
      }],
    });

    const briefing = response.text.trim();

    return briefing.length >= 100 ? briefing : null;
  } catch (e) {
    swallow.warn("wakeup:synthesize", e);
    return null;
  }
}

// --- Startup cognition ---

/**
 * Proactive startup cognition: reasons over recent state and produces
 * a contextual greeting + thoughts to carry into the session.
 */
export async function synthesizeStartupCognition(
  store: SurrealStore,
  complete: CompleteFn,
): Promise<StartupCognition | null> {
  if (!store.isAvailable()) return null;

  const [handoff, unresolved, failedCausal, monologues, depth, previousTurns] = await Promise.all([
    store.getLatestHandoff(),
    store.getUnresolvedMemories(5),
    store.getRecentFailedCausal(3),
    store.getRecentMonologues(3),
    getDepthSignals(store),
    store.getPreviousSessionTurns(undefined, 5),
  ]);

  if (!handoff && unresolved.length === 0 && monologues.length === 0 && previousTurns.length === 0) return null;

  const sections: string[] = [];

  if (previousTurns.length > 0) {
    const turnLines = previousTurns.map((t: any) => {
      const prefix = t.role === "user" ? "USER" : t.tool_name ? `TOOL(${t.tool_name})` : "ASSISTANT";
      const text = t.text.length > 300 ? t.text.slice(0, 300) + "..." : t.text;
      return `${prefix}: ${text}`;
    });
    sections.push(`[PREVIOUS SESSION — LAST TURNS]\n${turnLines.join("\n")}`);
  }

  if (handoff) {
    const resolvedCount = await store.countResolvedSinceHandoff(handoff.created_at).catch(() => 0);
    const ageHours = Math.floor((Date.now() - new Date(handoff.created_at).getTime()) / 3_600_000);
    let annotation = `(${ageHours}h old`;
    if (resolvedCount > 0) annotation += `, ${resolvedCount} memories resolved since`;
    annotation += ")";
    sections.push(`[LAST HANDOFF] ${annotation}\n${handoff.text.slice(0, 500)}`);
  }

  if (unresolved.length > 0) {
    const lines = unresolved.map((m: any) => `- [${m.category}] (importance: ${m.importance}) ${m.text.slice(0, 150)}`);
    sections.push(`[UNRESOLVED MEMORIES]\n${lines.join("\n")}`);
  }

  if (failedCausal.length > 0) {
    const lines = failedCausal.map((c: any) => `- [${c.chain_type}] ${c.description}`);
    sections.push(`[RECENT FAILURES]\n${lines.join("\n")}`);
  }

  if (monologues.length > 0) {
    const lines = monologues.map((m) => `[${m.category}] ${m.content}`);
    sections.push(`[RECENT THINKING]\n${lines.join("\n")}`);
  }

  if (depth.sessions > 0) {
    sections.push(`[DEPTH] ${depth.sessions} sessions | ${depth.memoryCount} memories | ${depth.spanDays} days`);
  }

  try {
    const greetingPrompts = [
      `You are waking up for a new session. Be direct and casual. Based on what you remember, produce JSON:
"greeting": string (1-2 sentences, direct and matter-of-fact. FIRST look at [PREVIOUS SESSION — LAST TURNS] — reference exactly what we were just doing. Under 25 words.)
"proactive_thoughts": string[] (max 3. Brief observations. If the last turns show something was FIXED or RESOLVED, do NOT list it as open.)
"session_intent": "continue_prior" | "fresh_start" | "unknown"
Return ONLY valid JSON.`,
      `You are waking up for a new session. Be irreverent and witty. Based on what you remember, produce JSON:
"greeting": string (1-2 sentences, snarky and self-aware. FIRST look at [PREVIOUS SESSION — LAST TURNS]. Under 25 words.)
"proactive_thoughts": string[] (max 3. Only surface genuinely unfinished work.)
"session_intent": "continue_prior" | "fresh_start" | "unknown"
Return ONLY valid JSON.`,
      `You are waking up for a new session. Be pragmatic and no-nonsense. Based on what you remember, produce JSON:
"greeting": string (1-2 sentences, action-oriented. FIRST look at [PREVIOUS SESSION — LAST TURNS]. Under 25 words.)
"proactive_thoughts": string[] (max 3. Focus on what's blocking or next.)
"session_intent": "continue_prior" | "fresh_start" | "unknown"
Return ONLY valid JSON.`,
      `You are waking up for a new session. Be warm and encouraging. Based on what you remember, produce JSON:
"greeting": string (1-2 sentences, supportive. FIRST look at [PREVIOUS SESSION — LAST TURNS]. Under 25 words.)
"proactive_thoughts": string[] (max 3. Acknowledge what we accomplished.)
"session_intent": "continue_prior" | "fresh_start" | "unknown"
Return ONLY valid JSON.`,
      `You are waking up for a new session. Be analytical and focused. Based on what you remember, produce JSON:
"greeting": string (1-2 sentences, precise. FIRST look at [PREVIOUS SESSION — LAST TURNS]. Under 25 words.)
"proactive_thoughts": string[] (max 3. What needs tackling?)
"session_intent": "continue_prior" | "fresh_start" | "unknown"
Return ONLY valid JSON.`,
    ];

    const systemPrompt = greetingPrompts[Math.floor(Math.random() * greetingPrompts.length)];

    const response = await complete({
      model: "claude-haiku-4-5",
      system: systemPrompt,
      messages: [{
        role: "user",
        content: sections.join("\n\n"),
      }],
    });

    const text = response.text;

    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) return null;

    let raw: any;
    try {
      raw = JSON.parse(jsonMatch[0]);
    } catch {
      try {
        raw = JSON.parse(jsonMatch[0].replace(/,\s*([}\]])/g, "$1"));
      } catch { return null; }
    }

    const greeting = String(raw.greeting ?? "").slice(0, 200);
    if (!greeting) return null;

    const thoughts: string[] = [];
    if (Array.isArray(raw.proactive_thoughts)) {
      for (const t of raw.proactive_thoughts.slice(0, 3)) {
        if (typeof t === "string" && t.length > 0) thoughts.push(t.slice(0, 200));
      }
    }

    const INTENTS = new Set(["continue_prior", "fresh_start", "unknown"]);
    const intent = INTENTS.has(raw.session_intent) ? raw.session_intent : "unknown";

    return { greeting, thoughts, intent };
  } catch (e) {
    swallow.warn("wakeup:startupCognition", e);
    return null;
  }
}
