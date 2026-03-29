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

    // Extract token counts (0 if provider didn't report usage)
    const inputTokens = event.usage?.input ?? 0;
    const outputTokens = event.usage?.output ?? 0;

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

    // Accumulate for daemon batching (only when real tokens present)
    if (inputTokens + outputTokens > 0) {
      session.newContentTokens += inputTokens + outputTokens;
    }

    // Track accumulated text output for planning gate
    const textLen = event.assistantTexts.reduce((s, t) => s + t.length, 0);
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
