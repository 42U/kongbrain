/**
 * Deferred Cleanup — extract knowledge from orphaned sessions.
 *
 * When the process dies abruptly (Ctrl+C×2), session cleanup never runs.
 * On next session start, this module finds orphaned sessions (started but
 * never marked cleanup_completed), loads their turns, runs daemon extraction,
 * generates a handoff note, and marks them complete.
 *
 * Turns are already persisted via afterTurn → ingest. This just processes them.
 */
import type { SurrealStore } from "./surreal.js";
import type { EmbeddingService } from "./embeddings.js";
import type { CompleteFn } from "./state.js";
import { buildSystemPrompt, buildTranscript, writeExtractionResults } from "./memory-daemon.js";
import type { PriorExtractions } from "./daemon-types.js";
import { swallow } from "./errors.js";

/**
 * Find and process orphaned sessions. Runs with a 30s total timeout.
 * Fire-and-forget from session_start — does not block the new session.
 */
export async function runDeferredCleanup(
  store: SurrealStore,
  embeddings: EmbeddingService,
  complete: CompleteFn,
): Promise<number> {
  if (!store.isAvailable()) return 0;

  const orphaned = await store.getOrphanedSessions(10).catch(() => []);
  if (orphaned.length === 0) return 0;

  let processed = 0;

  const cleanup = async () => {
    for (const session of orphaned) {
      try {
        await processOrphanedSession(session.id, store, embeddings, complete);
        processed++;
      } catch (e) {
        swallow.warn("deferredCleanup:session", e);
      }
    }
  };

  // 30s timeout — don't hold up the new session forever
  await Promise.race([
    cleanup(),
    new Promise<void>(resolve => setTimeout(resolve, 30_000)),
  ]);

  return processed;
}

async function processOrphanedSession(
  surrealSessionId: string,
  store: SurrealStore,
  embeddings: EmbeddingService,
  complete: CompleteFn,
): Promise<void> {
  // Load turns for extraction via part_of edges (turn->part_of->session)
  const turns = await store.queryFirst<{ role: string; text: string; tool_name?: string }>(
    `SELECT role, text, tool_name, created_at FROM turn
     WHERE id IN (SELECT VALUE in FROM part_of WHERE out = $sid)
     ORDER BY created_at ASC LIMIT 50`,
    { sid: surrealSessionId },
  ).catch(() => []);

  if (turns.length < 2) {
    // Nothing to extract, just mark complete
    await store.markSessionEnded(surrealSessionId).catch(e => swallow("deferred:markEmpty", e));
    return;
  }

  // Run daemon extraction
  const priorState: PriorExtractions = { conceptNames: [], artifactPaths: [], skillNames: [] };
  const turnData = turns.map(t => ({ role: t.role, text: t.text, tool_name: t.tool_name }));
  const transcript = buildTranscript(turnData);
  const systemPrompt = buildSystemPrompt(false, false, priorState);

  try {
    console.warn(`[deferred] extracting session ${surrealSessionId} (${turns.length} turns, transcript ${transcript.length} chars)`);
    const response = await complete({
      system: systemPrompt,
      messages: [{ role: "user", content: `[TRANSCRIPT]\n${transcript.slice(0, 60000)}` }],
    });

    const responseText = response.text;
    console.warn(`[deferred] extraction response: ${responseText.length} chars`);
    const jsonMatch = responseText.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      let result: Record<string, any>;
      try {
        result = JSON.parse(jsonMatch[0]);
      } catch {
        try {
          result = JSON.parse(jsonMatch[0].replace(/,\s*([}\]])/g, "$1"));
        } catch { result = {}; }
      }

      const keys = Object.keys(result);
      console.warn(`[deferred] parsed ${keys.length} keys: ${keys.join(", ")}`);
      if (keys.length > 0) {
        const sessionId = surrealSessionId;
        await writeExtractionResults(result, sessionId, store, embeddings, priorState);
        console.warn(`[deferred] wrote extraction results for ${surrealSessionId}`);
      }
    } else {
      console.warn(`[deferred] no JSON found in response`);
    }
  } catch (e) {
    swallow.warn("deferredCleanup:extraction", e);
  }

  // Generate handoff note
  try {
    const lastTurns = turns.slice(-15);
    const turnSummary = lastTurns
      .map(t => `[${t.role}] ${t.text.slice(0, 200)}`)
      .join("\n");

    const handoffResponse = await complete({
      system: "Summarize this session for handoff to your next self. What was worked on, what's unfinished, what to remember. 2-3 sentences. Write in first person.",
      messages: [{ role: "user", content: turnSummary }],
    });

    const handoffText = handoffResponse.text.trim();
    console.warn(`[deferred] handoff response: ${handoffText.length} chars`);
    if (handoffText.length > 20) {
      let emb: number[] | null = null;
      if (embeddings.isAvailable()) {
        try { emb = await embeddings.embed(handoffText); } catch { /* ok */ }
      }
      await store.createMemory(handoffText, emb, 8, "handoff", surrealSessionId);
    }
  } catch (e) {
    swallow.warn("deferredCleanup:handoff", e);
  }

  // Mark session as cleaned up
  await store.markSessionEnded(surrealSessionId).catch(e => swallow("deferred:markDone", e));
}
