/**
 * llm_output hook — token tracking, text length accumulation,
 * dynamic budget parsing, and cognitive check triggering.
 */

import type { GlobalPluginState } from "../state.js";
import { parseClassificationFromText } from "./before-tool-call.js";
import { swallow } from "../errors.js";

export function createLlmOutputHandler(state: GlobalPluginState) {
  return async (
    event: {
      runId: string;
      sessionId: string;
      provider: string;
      model: string;
      assistantTexts: string[];
      lastAssistant?: unknown;
      usage?: {
        input?: number;
        output?: number;
        cacheRead?: number;
        cacheWrite?: number;
        total?: number;
      };
    },
    ctx: { sessionKey?: string; sessionId?: string },
  ) => {
    const sessionKey = ctx.sessionKey ?? ctx.sessionId ?? "default";
    const session = state.getSession(sessionKey);
    if (!session) return;

    // Measure assistant text output (used for token estimation and planning gate)
    const textLen = event.assistantTexts.reduce((s, t) => s + t.length, 0);

    // Extract token counts — fall back to text-length estimate when provider
    // doesn't report usage (OpenClaw often passes 0 or undefined)
    let inputTokens = event.usage?.input ?? 0;
    let outputTokens = event.usage?.output ?? 0;
    if (inputTokens + outputTokens === 0 && textLen > 0) {
      outputTokens = Math.ceil(textLen / 4); // ~4 chars per token
    }

    // Always update session stats — turn_count must increment even without usage data
    if (session.surrealSessionId) {
      try {
        await state.store.updateSessionStats(
          session.surrealSessionId,
          inputTokens,
          outputTokens,
        );
      } catch (e) {
        swallow("hook:llmOutput:sessionStats", e);
      }
    }

    // Accumulate for daemon batching and mid-session cleanup
    session.newContentTokens += inputTokens + outputTokens;
    session.cumulativeTokens += inputTokens + outputTokens;

    // Track accumulated text output for planning gate
    session.turnTextLength += textLen;

    if (textLen > 50) {
      session.toolCallsSinceLastText = 0;
    }

    // Dynamic budget: parse LOOKUP/EDIT/REFACTOR from first assistant text
    if (session.toolCallCount <= 1 && event.assistantTexts.length > 0) {
      const fullText = event.assistantTexts.join("");
      const classLimit = parseClassificationFromText(fullText);
      if (classLimit !== null) {
        session.toolLimit = classLimit;
      }
    }

    // Capture thinking blocks for monologue extraction
    const lastAssistant = event.lastAssistant as any;
    if (lastAssistant?.content && Array.isArray(lastAssistant.content)) {
      for (const block of lastAssistant.content) {
        if (block.type === "thinking") {
          const thinking = block.thinking ?? block.text ?? "";
          if (thinking.length > 50) {
            session.pendingThinking.push(thinking);
            // Cap to prevent unbounded growth in long sessions
            const max = state.config.thresholds.maxPendingThinking;
            if (session.pendingThinking.length > max) {
              session.pendingThinking.splice(0, session.pendingThinking.length - max);
            }
          }
        }
      }
    }

    // Track lastAssistantText for downstream use (afterTurn, daemon batching).
    // Turn creation is handled by afterTurn() -> ingest() in context-engine.ts.
    if (event.assistantTexts.length > 0) {
      const text = event.assistantTexts.join("\n");
      if (text.length > 0) {
        session.lastAssistantText = text;
      }
    }
  };
}
