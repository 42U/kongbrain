/**
 * before_tool_call hook — planning gate + tool limit enforcement.
 *
 * - Planning gate: model must output text before its first tool call
 * - Tool limit: blocks when budget exceeded
 * - Soft interrupt: blocks when user pressed Ctrl+C
 */

import type { GlobalPluginState } from "../state.js";
import { recordToolCall } from "../orchestrator.js";

const DEFAULT_TOOL_LIMIT = 10;
const CLASSIFICATION_LIMITS: Record<string, number> = { LOOKUP: 3, EDIT: 4, REFACTOR: 8 };

export function createBeforeToolCallHandler(state: GlobalPluginState) {
  return async (
    event: {
      toolName: string;
      params: Record<string, unknown>;
      runId?: string;
      toolCallId?: string;
      assistantTextLengthSoFar?: number;
      toolCallIndexInTurn?: number;
    },
    ctx: { sessionKey?: string; sessionId?: string },
  ) => {
    const sessionKey = ctx.sessionKey ?? ctx.sessionId ?? "default";
    const session = state.getSession(sessionKey);
    if (!session) return;

    session.toolCallCount++;
    session.toolCallsSinceLastText++;

    // Record for steering analysis
    recordToolCall(session, event.toolName);

    // Use native fields when available, fall back to plugin-tracked state
    const textLengthSoFar = event.assistantTextLengthSoFar ?? session.turnTextLength;
    const toolIndex = event.toolCallIndexInTurn ?? (session.toolCallCount - 1);

    // Soft interrupt
    if (session.softInterrupted) {
      return {
        block: true,
        blockReason: "The user pressed Ctrl+C to interrupt you. Stop all tool calls immediately. Summarize what you've found so far, respond to the user with your current progress, and ask how to proceed.",
      };
    }

    // Tool limit
    if (session.toolCallCount > session.toolLimit) {
      return {
        block: true,
        blockReason: `Tool call limit reached (${session.toolLimit}). Stop calling tools. Continue exactly where you left off — deliver your answer from what you've gathered. Do NOT repeat anything you already said. State what's done and what remains.`,
      };
    }

    // Planning gate: model must output text before first tool call
    if (textLengthSoFar === 0 && toolIndex === 0) {
      return {
        block: true,
        blockReason:
          "PLANNING GATE — You must announce your plan before making tool calls.\n" +
          "1. Classify: LOOKUP (3 calls max), EDIT (4 max), REFACTOR (8 max)\n" +
          "2. STATE WHAT YOU ALREADY KNOW from injected memory/context — if you have prior knowledge about these files, say so\n" +
          "3. List each planned call and what SPECIFIC GAP it fills that memory doesn't cover\n" +
          "4. Every step still happens, but COMBINED. Edit + test in one bash call, not two.\n" +
          "If injected context already answers the question, you may need ZERO tool calls.\n" +
          "Speak your plan, then proceed.",
      };
    }

    return undefined;
  };
}

/**
 * Parse LOOKUP/EDIT/REFACTOR classification from planning gate response.
 * Called from llm_output to dynamically adjust tool limit.
 */
export function parseClassificationFromText(text: string): number | null {
  const match = text.match(/\b(LOOKUP|EDIT|REFACTOR)\b/);
  if (match && CLASSIFICATION_LIMITS[match[1]]) {
    return CLASSIFICATION_LIMITS[match[1]];
  }
  return null;
}
