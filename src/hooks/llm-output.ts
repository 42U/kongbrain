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

    // Track token usage
    if (event.usage) {
      const inputTokens = event.usage.input ?? 0;
      const outputTokens = event.usage.output ?? 0;

      // Update session stats in SurrealDB
      try {
        await state.store.updateSessionStats(
          session.sessionId,
          inputTokens,
          outputTokens,
        );
      } catch (e) {
        swallow("hook:llmOutput:sessionStats", e);
      }

      // Accumulate for daemon batching
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

    // Store assistant turn with embedding
    if (event.assistantTexts.length > 0) {
      const text = event.assistantTexts.join("\n");
      if (text.length > 0) {
        try {
          const embedLimit = Math.round(8192 * 3.4 * 0.8);
          let embedding: number[] | null = null;
          if (hasSemantic(text) && state.embeddings.isAvailable()) {
            try {
              embedding = await state.embeddings.embed(text.slice(0, embedLimit));
            } catch (e) { swallow("hook:llmOutput:embed", e); }
          }

          const turnId = await state.store.upsertTurn({
            session_id: session.sessionId,
            role: "assistant",
            text,
            embedding,
          });

          if (turnId) {
            await state.store.relate(turnId, "part_of", session.sessionId)
              .catch(e => swallow("hook:llmOutput:relate", e));

            // Extract and link concepts
            if (hasSemantic(text)) {
              extractAndLinkConcepts(turnId, text, state)
                .catch(e => swallow.warn("hook:llmOutput:concepts", e));
            }
          }

          session.lastAssistantText = text;
        } catch (e) {
          swallow.warn("hook:llmOutput:storeTurn", e);
        }
      }
    }
  };
}

function hasSemantic(text: string): boolean {
  if (text.length < 15) return false;
  if (/^(ok|yes|no|sure|thanks|done|got it|hmm|hm|yep|nope|cool|nice|great)\s*[.!?]?\s*$/i.test(text)) {
    return false;
  }
  return text.split(/\s+/).filter(w => w.length > 2).length >= 3;
}

// --- Concept extraction ---

const CONCEPT_RE = /\b(?:(?:use|using|implement|create|add|configure|setup|install|import)\s+)([A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)?)/g;
const TECH_TERMS = /\b(api|database|schema|migration|endpoint|middleware|component|service|module|handler|controller|model|interface|type|class|function|method|hook|plugin|extension|config|cache|queue|worker|daemon)\b/gi;

async function extractAndLinkConcepts(
  turnId: string,
  text: string,
  state: GlobalPluginState,
): Promise<void> {
  const concepts = new Set<string>();

  // Named concepts (PascalCase after action verbs)
  let match: RegExpExecArray | null;
  const re1 = new RegExp(CONCEPT_RE.source, CONCEPT_RE.flags);
  while ((match = re1.exec(text)) !== null) {
    concepts.add(match[1].trim());
  }

  // Technical terms
  const re2 = new RegExp(TECH_TERMS.source, TECH_TERMS.flags);
  while ((match = re2.exec(text)) !== null) {
    concepts.add(match[1].toLowerCase());
  }

  if (concepts.size === 0) return;

  for (const conceptText of [...concepts].slice(0, 10)) {
    try {
      let embedding: number[] | null = null;
      if (state.embeddings.isAvailable()) {
        try { embedding = await state.embeddings.embed(conceptText); } catch { /* ok */ }
      }
      const conceptId = await state.store.upsertConcept(conceptText, embedding);
      if (conceptId) {
        await state.store.relate(turnId, "mentions", conceptId)
          .catch(e => swallow("concepts:relate", e));
      }
    } catch (e) {
      swallow("concepts:upsert", e);
    }
  }
}
