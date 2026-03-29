/**
 * Memory Daemon — Persistent worker thread for incremental extraction.
 *
 * Runs alongside the main conversation thread for the entire session.
 * Receives turn batches from the main thread, calls an LLM for incremental
 * extraction of 9 knowledge types, and writes results to SurrealDB.
 *
 * Extracts: causal chains, monologue traces, resolved memories,
 *           concepts, corrections, preferences, artifacts, decisions, skills.
 *
 * This file runs inside a Worker thread — it is NOT imported by the main thread.
 *
 * Ported from kongbrain — creates own SurrealStore/EmbeddingService instances.
 */
import { parentPort, workerData } from "node:worker_threads";
import type { DaemonMessage, DaemonResponse, DaemonWorkerData, PriorExtractions, TurnData } from "./daemon-types.js";
import { SurrealStore } from "./surreal.js";
import { EmbeddingService } from "./embeddings.js";
import { swallow } from "./errors.js";

if (!parentPort) {
  throw new Error("memory-daemon.ts must be run as a worker thread");
}

const config = workerData as DaemonWorkerData;

// Worker-local instances
let store: SurrealStore;
let embeddings: EmbeddingService;

// --- Cumulative extraction counts ---
const counts = {
  turns: 0, causal: 0, monologue: 0, resolved: 0, concept: 0,
  correction: 0, preference: 0, artifact: 0, decision: 0, skill: 0, errors: 0,
};

let processing = false;
let shuttingDown = false;
const batchQueue: DaemonMessage[] = [];

const priorState: PriorExtractions = {
  conceptNames: [], artifactPaths: [], skillNames: [],
};

// --- Initialization ---

async function init(): Promise<boolean> {
  try {
    store = new SurrealStore(config.surrealConfig);
    await store.initialize();

    embeddings = new EmbeddingService(config.embeddingConfig);
    await embeddings.initialize();

    return true;
  } catch (e) {
    swallow.warn("memory-daemon:init", e);
    return false;
  }
}

// --- Build the extraction prompt ---

function buildSystemPrompt(
  hasThinking: boolean,
  hasRetrievedMemories: boolean,
  prior: PriorExtractions,
): string {
  const dedup = prior.conceptNames.length > 0 || prior.artifactPaths.length > 0 || prior.skillNames.length > 0
    ? `\n\nALREADY EXTRACTED (do NOT repeat these):
- Concepts: ${prior.conceptNames.length > 0 ? prior.conceptNames.join(", ") : "none yet"}
- Artifacts: ${prior.artifactPaths.length > 0 ? prior.artifactPaths.join(", ") : "none yet"}
- Skills: ${prior.skillNames.length > 0 ? prior.skillNames.join(", ") : "none yet"}`
    : "";

  return `You are a memory extraction daemon. Analyze the conversation transcript and extract structured knowledge.
Return ONLY valid JSON with these fields (all arrays, use [] if none found for a field):
${dedup}

{
  "causal": [
    // Cause->effect chains from debugging, refactoring, fixing, or building.
    // Only when there's a clear trigger and outcome. Max 5.
    {"triggerText": "what caused it (max 200 chars)", "outcomeText": "what happened as a result", "chainType": "debug|refactor|feature|fix", "success": true/false, "confidence": 0.0-1.0, "description": "1-sentence summary"}
  ],
${hasThinking ? `  "monologue": [
    // Internal reasoning moments worth preserving: doubts, tradeoffs, insights, realizations.
    // Skip routine reasoning. Only novel/surprising thoughts. Max 5.
    {"category": "doubt|tradeoff|alternative|insight|realization", "content": "1-2 sentence description"}
  ],` : '  "monologue": [],'}
${hasRetrievedMemories ? `  "resolved": [
    // IDs from [RETRIEVED MEMORIES] that have been FULLY addressed/fixed/completed in this conversation.
    // Must be exact IDs like "memory:abc123". Empty [] if none resolved.
    "memory:example_id"
  ],` : '  "resolved": [],'}
  "concepts": [
    // Technical facts, knowledge, decisions, or findings worth remembering.
    // NOT conversation flow — only things that would be useful to recall later.
    // Categories: technical, architectural, behavioral, environmental, procedural
    // Max 8 per batch.
    {"name": "short identifier (3-6 words)", "content": "the actual knowledge (1-3 sentences)", "category": "technical|architectural|behavioral|environmental|procedural", "importance": 1-10}
  ],
  "corrections": [
    // Moments where the user corrects the assistant's understanding, approach, or output.
    // These are high-value signals about what NOT to do.
    {"original": "what the assistant said/did wrong", "correction": "what the user said the right answer/approach is", "context": "brief context of when this happened"}
  ],
  "preferences": [
    // User behavioral signals: communication style, workflow preferences, tool preferences.
    // Only extract NOVEL preferences not already obvious. Max 5.
    {"preference": "what the user prefers (1 sentence)", "evidence": "what they said/did that shows this"}
  ],
  "artifacts": [
    // Files that were created, modified, read, or discussed.
    // Extract from tool calls (bash, read, write, edit, grep commands).
    {"path": "/path/to/file", "action": "created|modified|read|discussed", "summary": "what was done to it (1 sentence)"}
  ],
  "decisions": [
    // Explicit choices made during the conversation with reasoning.
    // Architecture decisions, tool choices, approach selections. Max 3.
    {"decision": "what was decided", "rationale": "why", "alternatives_considered": "what else was considered (or 'none discussed')"}
  ],
  "skills": [
    // Reusable multi-step procedures that WORKED. Only extract when a procedure
    // was successfully completed and would be useful to repeat. Max 2.
    {"name": "short name", "steps": ["step 1", "step 2", "..."], "trigger_context": "when to use this skill"}
  ]
}

RULES:
- Return ONLY the JSON object. No markdown, no explanation.
- Every field must be present (use [] for empty).
- Quality over quantity — skip weak/uncertain extractions.
- Concepts should be self-contained — readable without the conversation.
- Corrections are the MOST important signal. Never miss one.
- For artifacts, extract file paths from bash/tool commands in the transcript.`;
}

function buildTranscript(turns: TurnData[]): string {
  return turns
    .map(t => {
      const prefix = t.tool_name ? `[tool:${t.tool_name}]` : `[${t.role}]`;
      let line = `${prefix} ${(t.text ?? "").slice(0, 1500)}`;
      if (t.tool_result) line += `\n  -> ${t.tool_result.slice(0, 500)}`;
      if (t.file_paths && t.file_paths.length > 0) line += `\n  files: ${t.file_paths.join(", ")}`;
      return line;
    })
    .join("\n");
}

// --- Main extraction logic ---

async function processExtraction(msg: DaemonMessage & { type: "turn_batch" }): Promise<void> {
  processing = true;
  try {
    const { turns, thinking, retrievedMemories, sessionId, priorExtractions } = msg;

    if (turns.length < 2) return;

    // Merge incoming prior state
    if (priorExtractions) {
      for (const name of priorExtractions.conceptNames) {
        if (!priorState.conceptNames.includes(name)) priorState.conceptNames.push(name);
      }
      for (const path of priorExtractions.artifactPaths) {
        if (!priorState.artifactPaths.includes(path)) priorState.artifactPaths.push(path);
      }
      for (const name of priorExtractions.skillNames) {
        if (!priorState.skillNames.includes(name)) priorState.skillNames.push(name);
      }
    }

    const transcript = buildTranscript(turns);
    const sections: string[] = [`[TRANSCRIPT]\n${transcript.slice(0, 60000)}`];

    if (thinking.length > 0) {
      sections.push(`[THINKING]\n${thinking.slice(-8).join("\n---\n").slice(0, 4000)}`);
    }

    if (retrievedMemories.length > 0) {
      const memList = retrievedMemories.map(m => `${m.id}: ${String(m.text).slice(0, 200)}`).join("\n");
      sections.push(`[RETRIEVED MEMORIES]\nMark any that have been fully addressed/fixed/completed.\n${memList}`);
    }

    const systemPrompt = buildSystemPrompt(thinking.length > 0, retrievedMemories.length > 0, priorState);

    const { completeSimple, getModel } = await import("@mariozechner/pi-ai");
    const provider = config.llmProvider;
    const modelId = config.llmModel;
    if (!provider || !modelId) {
      throw new Error("Memory daemon requires llmProvider and llmModel from host config");
    }
    // getModel is heavily typed for known providers; cast needed for runtime-configured values
    const model = (getModel as any)(provider, modelId);

    const response = await completeSimple(model, {
      systemPrompt,
      messages: [{
        role: "user",
        timestamp: Date.now(),
        content: sections.join("\n\n"),
      }],
    });

    const responseText = response.content
      .filter((c: any) => c.type === "text")
      .map((c: any) => c.text)
      .join("");

    const jsonMatch = responseText.match(/\{[\s\S]*\}/);
    if (!jsonMatch) return;

    let result: Record<string, any>;
    try {
      result = JSON.parse(jsonMatch[0]);
    } catch {
      try {
        result = JSON.parse(jsonMatch[0].replace(/,\s*([}\]])/g, "$1"));
      } catch {
        // Per-field fallback
        result = {};
        const fields = ["causal", "monologue", "resolved", "concepts", "corrections", "preferences", "artifacts", "decisions", "skills"];
        for (const field of fields) {
          const fieldMatch = jsonMatch[0].match(new RegExp(`"${field}"\\s*:\\s*(\\[[\\s\\S]*?\\])(?=\\s*[,}]\\s*"[a-z]|\\s*\\}$)`, "m"));
          if (fieldMatch) {
            try { result[field] = JSON.parse(fieldMatch[1]); } catch { /* skip */ }
          }
        }
        if (Object.keys(result).length === 0) return;
      }
    }

    // --- Write all results to DB ---
    const writeOps: Promise<void>[] = [];

    // 1. Causal chains
    if (Array.isArray(result.causal) && result.causal.length > 0) {
      const { linkCausalEdges } = await import("./causal.js");
      const validated = result.causal
        .filter((c: any) => c.triggerText && c.outcomeText && c.chainType && typeof c.success === "boolean")
        .slice(0, 5)
        .map((c: any) => ({
          triggerText: String(c.triggerText).slice(0, 200),
          outcomeText: String(c.outcomeText).slice(0, 200),
          chainType: (["debug", "refactor", "feature", "fix"].includes(c.chainType) ? c.chainType : "fix") as "debug" | "refactor" | "feature" | "fix",
          success: Boolean(c.success),
          confidence: Math.max(0, Math.min(1, Number(c.confidence) || 0.5)),
          description: String(c.description ?? "").slice(0, 150),
        }));
      if (validated.length > 0) {
        writeOps.push(linkCausalEdges(validated, sessionId, store, embeddings));
        counts.causal += validated.length;
      }
    }

    // 2. Monologue traces
    if (Array.isArray(result.monologue) && result.monologue.length > 0) {
      for (const entry of result.monologue.slice(0, 5)) {
        if (!entry.category || !entry.content) continue;
        counts.monologue++;
        writeOps.push((async () => {
          let emb: number[] | null = null;
          if (embeddings.isAvailable()) {
            try { emb = await embeddings.embed(entry.content); } catch (e) { swallow("daemon:embedMonologue", e); }
          }
          await store.createMonologue(sessionId, entry.category, entry.content, emb);
        })());
      }
    }

    // 3. Resolved memories
    if (Array.isArray(result.resolved) && result.resolved.length > 0) {
      const RECORD_ID_RE = /^memory:[a-zA-Z0-9_]+$/;
      writeOps.push((async () => {
        for (const memId of result.resolved!.slice(0, 20)) {
          if (typeof memId !== "string" || !RECORD_ID_RE.test(memId)) continue;
          counts.resolved++;
          await store.queryExec(
            `UPDATE ${memId} SET status = 'resolved', resolved_at = time::now(), resolved_by = $sid`,
            { sid: sessionId },
          ).catch(e => swallow.warn("daemon:resolveMemory", e));
        }
      })());
    }

    // 4. Concepts
    if (Array.isArray(result.concepts) && result.concepts.length > 0) {
      for (const c of result.concepts.slice(0, 11)) {
        if (!c.name || !c.content) continue;
        if (priorState.conceptNames.includes(c.name)) continue;
        counts.concept++;
        priorState.conceptNames.push(c.name);
        writeOps.push((async () => {
          let emb: number[] | null = null;
          if (embeddings.isAvailable()) {
            try { emb = await embeddings.embed(c.content); } catch (e) { swallow("daemon:embedConcept", e); }
          }
          await store.upsertConcept(c.content, emb, `daemon:${sessionId}`);
        })());
      }
    }

    // 5. Corrections — high-importance memories
    if (Array.isArray(result.corrections) && result.corrections.length > 0) {
      for (const c of result.corrections.slice(0, 5)) {
        if (!c.original || !c.correction) continue;
        counts.correction++;
        const text = `[CORRECTION] Original: "${String(c.original).slice(0, 200)}" -> Corrected: "${String(c.correction).slice(0, 200)}" (Context: ${String(c.context ?? "").slice(0, 100)})`;
        writeOps.push((async () => {
          let emb: number[] | null = null;
          if (embeddings.isAvailable()) {
            try { emb = await embeddings.embed(text); } catch (e) { swallow("daemon:embedCorrection", e); }
          }
          await store.createMemory(text, emb, 9, "correction", sessionId);
        })());
      }
    }

    // 6. User preferences
    if (Array.isArray(result.preferences) && result.preferences.length > 0) {
      for (const p of result.preferences.slice(0, 5)) {
        if (!p.preference) continue;
        counts.preference++;
        const text = `[USER PREFERENCE] ${String(p.preference).slice(0, 250)} (Evidence: ${String(p.evidence ?? "").slice(0, 150)})`;
        writeOps.push((async () => {
          let emb: number[] | null = null;
          if (embeddings.isAvailable()) {
            try { emb = await embeddings.embed(text); } catch (e) { swallow("daemon:embedPreference", e); }
          }
          await store.createMemory(text, emb, 7, "preference", sessionId);
        })());
      }
    }

    // 7. Artifacts
    if (Array.isArray(result.artifacts) && result.artifacts.length > 0) {
      for (const a of result.artifacts.slice(0, 10)) {
        if (!a.path) continue;
        if (priorState.artifactPaths.includes(a.path)) continue;
        counts.artifact++;
        priorState.artifactPaths.push(a.path);
        const desc = `${String(a.action ?? "modified")}: ${String(a.summary ?? "").slice(0, 200)}`;
        writeOps.push((async () => {
          let emb: number[] | null = null;
          if (embeddings.isAvailable()) {
            try { emb = await embeddings.embed(`${a.path} ${desc}`); } catch (e) { swallow("daemon:embedArtifact", e); }
          }
          await store.createArtifact(a.path, a.action ?? "modified", desc, emb);
        })());
      }
    }

    // 8. Decisions
    if (Array.isArray(result.decisions) && result.decisions.length > 0) {
      for (const d of result.decisions.slice(0, 6)) {
        if (!d.decision) continue;
        counts.decision++;
        const text = `[DECISION] ${String(d.decision).slice(0, 200)} — Rationale: ${String(d.rationale ?? "").slice(0, 200)} (Alternatives: ${String(d.alternatives_considered ?? "none").slice(0, 100)})`;
        writeOps.push((async () => {
          let emb: number[] | null = null;
          if (embeddings.isAvailable()) {
            try { emb = await embeddings.embed(text); } catch (e) { swallow("daemon:embedDecision", e); }
          }
          await store.createMemory(text, emb, 7, "decision", sessionId);
        })());
      }
    }

    // 9. Skills
    if (Array.isArray(result.skills) && result.skills.length > 0) {
      for (const s of result.skills.slice(0, 3)) {
        if (!s.name || !Array.isArray(s.steps) || s.steps.length === 0) continue;
        if (priorState.skillNames.includes(s.name)) continue;
        counts.skill++;
        priorState.skillNames.push(s.name);
        const content = `${s.name}\nTrigger: ${String(s.trigger_context ?? "").slice(0, 150)}\nSteps:\n${s.steps.map((st: string, i: number) => `${i + 1}. ${String(st).slice(0, 200)}`).join("\n")}`;
        writeOps.push((async () => {
          let emb: number[] | null = null;
          if (embeddings.isAvailable()) {
            try { emb = await embeddings.embed(content); } catch (e) { swallow("daemon:embedSkill", e); }
          }
          await store.queryExec(
            `CREATE skill CONTENT $record`,
            {
              record: {
                name: String(s.name).slice(0, 100),
                description: content,
                content,
                steps: s.steps.map((st: string) => String(st).slice(0, 200)),
                trigger_context: String(s.trigger_context ?? "").slice(0, 200),
                tags: ["auto-extracted"],
                session_id: sessionId,
                ...(emb ? { embedding: emb } : {}),
              },
            },
          ).catch(e => swallow.warn("daemon:createSkill", e));
        })());
      }
    }

    await Promise.allSettled(writeOps);

    counts.turns = turns.length;

    parentPort!.postMessage({
      type: "extraction_complete",
      extractedTurnCount: counts.turns,
      causalCount: counts.causal,
      monologueCount: counts.monologue,
      resolvedCount: counts.resolved,
      conceptCount: counts.concept,
      correctionCount: counts.correction,
      preferenceCount: counts.preference,
      artifactCount: counts.artifact,
      decisionCount: counts.decision,
      skillCount: counts.skill,
      extractedNames: { ...priorState },
    } satisfies DaemonResponse);
  } catch (e) {
    counts.errors++;
    swallow.warn("memory-daemon:extraction", e);
    parentPort!.postMessage({
      type: "error",
      message: String(e),
    } satisfies DaemonResponse);
  } finally {
    processing = false;
  }
}

// --- Batch Queue Processing ---

async function drainQueue(): Promise<void> {
  while (batchQueue.length > 0 && !shuttingDown) {
    const batch = batchQueue.shift()!;
    if (batch.type === "turn_batch") {
      await processExtraction(batch);
    }
  }
}

// --- Message Handler ---

async function handleMessage(msg: DaemonMessage): Promise<void> {
  switch (msg.type) {
    case "turn_batch": {
      batchQueue.length = 0;
      batchQueue.push(msg);
      if (!processing) {
        drainQueue().catch(e => swallow.warn("daemon:drainQueue", e));
      }
      break;
    }
    case "shutdown": {
      shuttingDown = true;
      if (processing) {
        await Promise.race([
          new Promise<void>(resolve => {
            const check = setInterval(() => {
              if (!processing) { clearInterval(check); resolve(); }
            }, 100);
          }),
          new Promise<void>(resolve => setTimeout(resolve, 45_000)),
        ]);
      }
      try {
        await Promise.allSettled([
          store.dispose(),
          embeddings.dispose(),
        ]);
      } catch (e) { swallow("daemon:cleanup", e); }

      parentPort!.postMessage({ type: "shutdown_complete" } satisfies DaemonResponse);
      break;
    }
    case "status_request": {
      parentPort!.postMessage({
        type: "status",
        extractedTurns: counts.turns,
        pendingBatches: batchQueue.length,
        errors: counts.errors,
      } satisfies DaemonResponse);
      break;
    }
  }
}

// --- Main ---

init().then(ok => {
  if (!ok) {
    parentPort!.postMessage({ type: "error", message: "Daemon initialization failed" } satisfies DaemonResponse);
    return;
  }
  parentPort!.on("message", (msg: DaemonMessage) => {
    handleMessage(msg).catch(e => swallow.warn("daemon:handleMessage", e));
  });
});
