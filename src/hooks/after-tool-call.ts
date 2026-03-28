/**
 * after_tool_call hook — artifact tracking + tool outcome recording.
 */

import type { GlobalPluginState } from "../state.js";
import { recordToolOutcome } from "../retrieval-quality.js";
import { swallow } from "../errors.js";

export function createAfterToolCallHandler(state: GlobalPluginState) {
  return async (
    event: {
      toolName: string;
      params: Record<string, unknown>;
      toolCallId?: string;
      result?: unknown;
      error?: string;
      durationMs?: number;
    },
    ctx: { sessionKey?: string; sessionId?: string },
  ) => {
    const sessionKey = ctx.sessionKey ?? ctx.sessionId ?? "default";
    const session = state.getSession(sessionKey);
    if (!session) return;

    const isError = !!event.error;
    recordToolOutcome(!isError);

    // Store tool result snippet
    const resultText = typeof event.result === "string"
      ? event.result.slice(0, 500)
      : JSON.stringify(event.result ?? "").slice(0, 500);

    try {
      await state.store.upsertTurn({
        session_id: session.sessionId,
        role: "tool",
        text: `[${event.toolName}] ${resultText}`,
        embedding: null,
      });
    } catch (e) {
      swallow("hook:afterToolCall:store", e);
    }

    // Auto-track file artifacts from write/edit tools
    if (!isError) {
      trackArtifact(event.toolName, event.params, session.taskId, state)
        .catch(e => swallow.warn("hook:afterToolCall:artifact", e));
    }

    // Clean up pending args
    if (event.toolCallId) {
      session.pendingToolArgs.delete(event.toolCallId);
    }
  };
}

async function trackArtifact(
  toolName: string,
  args: Record<string, unknown>,
  taskId: string,
  state: GlobalPluginState,
): Promise<void> {
  const ARTIFACT_TOOLS: Record<string, string> = {
    write: "created", edit: "edited", bash: "shell",
  };
  const action = ARTIFACT_TOOLS[toolName];
  if (!action) return;

  let description: string | null = null;

  if (toolName === "write" && args.path) {
    description = `File created: ${args.path}`;
  } else if (toolName === "edit" && args.path) {
    description = `File edited: ${args.path}`;
  } else if (toolName === "bash" && typeof args.command === "string") {
    const cmd = args.command;
    if (/\b(cp|mv|touch|mkdir|npm init|git init|tsc)\b/.test(cmd)) {
      description = `Shell: ${cmd.slice(0, 200)}`;
    } else {
      return;
    }
  }

  if (!description) return;

  let emb: number[] | null = null;
  if (state.embeddings.isAvailable()) {
    try { emb = await state.embeddings.embed(description); } catch { /* ok */ }
  }

  const ext = (args.path as string)?.split(".").pop() ?? "unknown";
  const artifactId = await state.store.createArtifact(
    (args.path as string) ?? "shell", ext, description, emb,
  );
  if (artifactId && taskId) {
    await state.store.relate(taskId, "produced", artifactId)
      .catch(e => swallow.warn("artifact:relate", e));
  }
}
