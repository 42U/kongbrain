/**
 * Graph-based context transformation for OpenClaw.
 *
 * Ported from kongbrain's graph-context.ts. Key changes:
 * - No module-level state: all mutable state flows through SessionState
 * - SurrealStore and EmbeddingService are passed as parameters
 * - Designed to be called from ContextEngine.assemble()
 */

import type { AgentMessage } from "@mariozechner/pi-agent-core";
import type {
  UserMessage, AssistantMessage, ToolResultMessage,
  TextContent, ThinkingContent, ToolCall, ImageContent,
} from "@mariozechner/pi-ai";
import type { SurrealStore, VectorSearchResult, CoreMemoryEntry } from "./surreal.js";
import type { EmbeddingService } from "./embeddings.js";
import type { SessionState } from "./state.js";
import { getPendingDirectives, clearPendingDirectives, getSessionContinuity, getSuppressedNodeIds } from "./cognitive-check.js";
import { queryCausalContext } from "./causal.js";
import { findRelevantSkills, formatSkillContext } from "./skills.js";
import { retrieveReflections, formatReflectionContext } from "./reflection.js";
import { getCachedContext, recordPrefetchHit, recordPrefetchMiss } from "./prefetch.js";
import { stageRetrieval, getHistoricalUtilityBatch } from "./retrieval-quality.js";
import { isACANActive, scoreWithACAN, type ACANCandidate } from "./acan.js";
import { swallow } from "./errors.js";

// ── Message type guards ────────────────────────────────────────────────────────

type ContentBlock = TextContent | ThinkingContent | ToolCall | ImageContent;

function isUser(msg: AgentMessage): msg is UserMessage {
  return (msg as UserMessage).role === "user";
}
function isAssistant(msg: AgentMessage): msg is AssistantMessage {
  return (msg as AssistantMessage).role === "assistant";
}
function isToolResult(msg: AgentMessage): msg is ToolResultMessage {
  return (msg as ToolResultMessage).role === "toolResult";
}
function msgRole(msg: AgentMessage): string {
  if (isUser(msg)) return msg.role;
  if (isAssistant(msg)) return msg.role;
  if (isToolResult(msg)) return msg.role;
  return "unknown";
}
function msgContentBlocks(msg: AgentMessage): ContentBlock[] {
  if (isUser(msg)) {
    return typeof msg.content === "string"
      ? [{ type: "text", text: msg.content } as TextContent]
      : msg.content as ContentBlock[];
  }
  if (isAssistant(msg)) return msg.content;
  if (isToolResult(msg)) return msg.content as ContentBlock[];
  return [];
}

// ── Constants ──────────────────────────────────────────────────────────────────

const CHARS_PER_TOKEN = 3.4;
const BUDGET_FRACTION = 0.70;
const CONVERSATION_SHARE = 0.50;
const RETRIEVAL_SHARE = 0.30;
const CORE_MEMORY_SHARE = 0.15;
const CORE_MEMORY_TTL = 300_000;
const MAX_ITEM_CHARS = 1200; // ~350 tokens per item cap (claw-code: MAX_INSTRUCTION_FILE_CHARS)
const MIN_RELEVANCE_SCORE = 0.35;
const MIN_COSINE = 0.25;

// Recency decay
const RECENCY_DECAY_FAST = 0.99;
const RECENCY_DECAY_SLOW = 0.995;
const RECENCY_BOUNDARY_HOURS = 4;

// Utility pre-filtering
const UTILITY_PREFILTER_MIN_RETRIEVALS = 5;
const UTILITY_PREFILTER_MAX_UTIL = 0.05;

// Intent score floors
const INTENT_SCORE_FLOORS: Record<string, number> = {
  "simple-question": 0.20, "meta-session": 0.18, "code-read": 0.14,
  "code-write": 0.12, "code-debug": 0.12, "deep-explore": 0.10,
  "reference-prior": 0.08, "multi-step": 0.12, "continuation": 0.10,
  "unknown": 0.12,
};
const SCORE_FLOOR_DEFAULT = 0.12;
const INTENT_REMINDER_THRESHOLD = 10;

// ── Budget calculation ─────────────────────────────────────────────────────────

interface Budgets {
  conversation: number;
  retrieval: number;
  core: number;
  maxContextItems: number;
}

function calcBudgets(contextWindow: number): Budgets {
  const total = contextWindow * BUDGET_FRACTION;
  const retrieval = Math.round(total * RETRIEVAL_SHARE);
  return {
    conversation: Math.round(total * CONVERSATION_SHARE),
    retrieval,
    core: Math.round(total * CORE_MEMORY_SHARE),
    maxContextItems: Math.max(20, Math.round(retrieval / 300)),
  };
}

// ── Context stats ──────────────────────────────────────────────────────────────

export interface ContextStats {
  fullHistoryTokens: number;
  sentTokens: number;
  savedTokens: number;
  reductionPct: number;
  graphNodes: number;
  neighborNodes: number;
  recentTurns: number;
  mode: "graph" | "recency-only" | "passthrough";
  prefetchHit: boolean;
}

// ── Scoring types ──────────────────────────────────────────────────────────────

interface ScoredResult extends VectorSearchResult {
  finalScore: number;
  fromNeighbor?: boolean;
}

// ── Helper functions ───────────────────────────────────────────────────────────

function extractText(msg: UserMessage | AssistantMessage): string {
  if (typeof msg.content === "string") return msg.content;
  if (Array.isArray(msg.content)) {
    return (msg.content as ContentBlock[])
      .filter((c): c is TextContent => c.type === "text")
      .map((c) => c.text)
      .join("\n");
  }
  return "";
}

function extractLastUserText(messages: AgentMessage[]): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i] as UserMessage;
    if (msg.role === "user") {
      const text = extractText(msg);
      if (text) return text;
    }
  }
  return null;
}

/** Estimate char count for a single content block (claw-code: per-block-type estimation). */
function blockCharLen(c: any): number {
  if (c.type === "text") return c.text.length;
  if (c.type === "thinking") return c.thinking.length;
  if (c.type === "toolCall") {
    // Count tool name + serialized args (claw-code: compact.rs:326-338)
    return (c.name?.length ?? 0) + (c.args ? JSON.stringify(c.args).length : 0);
  }
  if (c.type === "toolResult" && Array.isArray(c.content)) {
    let len = 0;
    for (const rc of c.content) {
      if (rc.type === "text") len += rc.text.length;
      else len += 100;
    }
    return len;
  }
  return 100; // image, etc.
}

function estimateTokens(messages: AgentMessage[]): number {
  let chars = 0;
  for (const msg of messages) {
    for (const c of msgContentBlocks(msg)) chars += blockCharLen(c);
    chars += 4; // per-message structural overhead
  }
  return Math.ceil(chars / CHARS_PER_TOKEN);
}

function msgCharLen(msg: AgentMessage): number {
  let len = 0;
  for (const c of msgContentBlocks(msg)) len += blockCharLen(c);
  return len;
}

function recencyScore(timestamp: string | undefined): number {
  if (!timestamp) return 0.3;
  const hoursElapsed = (Date.now() - new Date(timestamp).getTime()) / (1000 * 60 * 60);
  if (hoursElapsed <= RECENCY_BOUNDARY_HOURS) {
    return Math.pow(RECENCY_DECAY_FAST, hoursElapsed);
  }
  const fastPart = Math.pow(RECENCY_DECAY_FAST, RECENCY_BOUNDARY_HOURS);
  return fastPart * Math.pow(RECENCY_DECAY_SLOW, hoursElapsed - RECENCY_BOUNDARY_HOURS);
}

export function formatRelativeTime(ts: string): string {
  const ms = Date.now() - new Date(ts).getTime();
  const mins = Math.floor(ms / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 7) return `${days}d ago`;
  const weeks = Math.floor(days / 7);
  if (weeks < 5) return `${weeks}w ago`;
  return `${Math.floor(days / 30)}mo ago`;
}

function accessBoost(accessCount: number | undefined): number {
  return Math.log1p(accessCount ?? 0);
}

export function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  return denom > 0 ? dot / denom : 0;
}

// ── Rules suffix (tool budget injection) ───────────────────────────────────────

function buildRulesSuffix(session: SessionState): string {
  const remaining = session.toolLimit === Infinity
    ? "unlimited" : String(Math.max(0, session.toolLimit - session.toolCallCount));
  const urgency = session.toolLimit !== Infinity && (session.toolLimit - session.toolCallCount) <= 3
    ? "\n⚠ WRAP UP or check in with user." : "";

  // After first exposure, send only the budget line (claw-code: don't re-send static content)
  if (session.injectedSections.has("rules_full")) {
    return (
      "\n<rules_reminder>" +
      `\nBudget: ${session.toolCallCount} used, ${remaining} remaining.${urgency}` +
      "\nCombine steps. If context already answers it, zero calls." +
      "\n</rules_reminder>"
    );
  }

  // First time — full examples
  session.injectedSections.add("rules_full");
  return (
    "\n<rules_reminder>" +
    `\nBudget: ${session.toolCallCount} used, ${remaining} remaining.${urgency}` +
    "\n\nYOUR BUDGET IS SMALL. Plan the whole task, not just the next call." +
    "\n" +
    "\nTask: Fix broken import" +
    "\n  WASTEFUL (6 calls): grep old → read file → grep new → read context → edit → read to verify" +
    "\n  DENSE (2 calls):" +
    "\n    1. grep -n 'oldImport' src/**/*.ts; grep -rn 'newModule' src/" +
    "\n    2. edit file && npm test -- --grep 'relevant' 2>&1 | tail -20" +
    "\n" +
    "\nTask: Debug failing test" +
    "\n  WASTEFUL (8 calls): run test → read output → read test → read source → grep → read more → edit → rerun" +
    "\n  DENSE (3 calls):" +
    "\n    1. npm test 2>&1 | tail -30" +
    "\n    2. grep -n 'failingTest\\|relevantFn' test/*.ts src/*.ts" +
    "\n    3. edit fix && npm test 2>&1 | tail -15" +
    "\n" +
    "\nTask: Read/understand multiple files" +
    "\n  WASTEFUL (10 calls): cat file1 → cat file2 → cat file3 → ..." +
    "\n  DENSE (1-2 calls):" +
    "\n    1. head -80 src/a.ts src/b.ts src/c.ts src/d.ts  (4 files in ONE call)" +
    "\n    2. grep -n 'keyPattern' src/*.ts  (search all files at once, not one by one)" +
    "\n" +
    "\nEvery step still happens — investigation, edit, verification — but COMBINED into fewer calls." +
    "\nThe answer is often already in context. Don't call if you already know." +
    "\nAnnounce: task type (LOOKUP=1/EDIT=2/REFACTOR=6), planned calls, what each does." +
    "\n</rules_reminder>"
  );
}

function injectRulesSuffix(messages: AgentMessage[], session: SessionState): AgentMessage[] {
  const suffix = buildRulesSuffix(session);
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (isUser(msg)) {
      const clone = [...messages];
      clone[i] = {
        ...msg,
        content: typeof msg.content === "string" ? msg.content + suffix : msg.content,
      } as UserMessage;
      return clone;
    }
    if (isToolResult(msg)) {
      const clone = [...messages];
      const content = Array.isArray(msg.content) ? [...msg.content] : msg.content;
      if (Array.isArray(content)) {
        content.push({ type: "text", text: suffix } as TextContent);
      }
      clone[i] = { ...msg, content } as ToolResultMessage;
      return clone;
    }
  }
  return messages;
}

// ── Contextual query vector ────────────────────────────────────────────────────

async function buildContextualQueryVec(
  queryText: string,
  messages: AgentMessage[],
  embeddings: EmbeddingService,
): Promise<number[]> {
  const queryVec = await embeddings.embed(queryText);

  const recentTexts: string[] = [];
  for (let i = messages.length - 2; i >= 0 && recentTexts.length < 3; i--) {
    const msg = messages[i] as UserMessage | AssistantMessage;
    if (msg.role === "user" || msg.role === "assistant") {
      const text = extractText(msg);
      if (text && text.length > 10) {
        recentTexts.push(text.slice(0, 500));
      }
    }
  }

  if (recentTexts.length === 0) return queryVec;

  try {
    const recentVecs = await Promise.all(recentTexts.map((t) => embeddings.embed(t)));
    const dim = queryVec.length;
    const blended = new Array(dim).fill(0);
    const queryWeight = 2;
    const totalWeight = queryWeight + recentVecs.length;

    for (let d = 0; d < dim; d++) {
      blended[d] = queryVec[d] * queryWeight;
      for (const rv of recentVecs) {
        blended[d] += rv[d];
      }
      blended[d] /= totalWeight;
    }
    return blended;
  } catch (e) {
    swallow.warn("graph-context:contextualQuery", e);
    return queryVec;
  }
}

// ── Scoring ────────────────────────────────────────────────────────────────────

async function scoreResults(
  results: VectorSearchResult[],
  neighborIds: Set<string>,
  queryEmbedding: number[] | undefined,
  store: SurrealStore,
  currentIntent: string,
): Promise<ScoredResult[]> {
  const eligibleIds = results
    .filter((r) => r.table === "memory" || r.table === "concept")
    .map((r) => r.id);

  const cacheEntries = await store.getUtilityCacheEntries(eligibleIds);

  const preFiltered = results.filter((r) => {
    const entry = cacheEntries.get(r.id);
    if (!entry) return true;
    if (entry.retrieval_count < UTILITY_PREFILTER_MIN_RETRIEVALS) return true;
    return entry.avg_utilization >= UTILITY_PREFILTER_MAX_UTIL;
  });

  let utilityMap = new Map<string, number>();
  for (const [id, entry] of cacheEntries) {
    utilityMap.set(id, entry.avg_utilization);
  }
  if (utilityMap.size === 0 && eligibleIds.length > 0) {
    utilityMap = await getHistoricalUtilityBatch(eligibleIds);
  }

  const reflectedSessions = await store.getReflectionSessionIds();
  const floor = INTENT_SCORE_FLOORS[currentIntent] ?? SCORE_FLOOR_DEFAULT;

  // ACAN path
  if (isACANActive() && queryEmbedding && preFiltered.length > 0 && preFiltered.every((r) => r.embedding)) {
    const candidates: ACANCandidate[] = preFiltered.map((r) => ({
      embedding: r.embedding!,
      recency: recencyScore(r.timestamp),
      importance: (r.importance ?? 0.5) / 10,
      access: Math.min(accessBoost(r.accessCount), 1),
      neighborBonus: neighborIds.has(r.id) ? 1.0 : 0,
      provenUtility: utilityMap.get(r.id) ?? 0,
      reflectionBoost: r.sessionId ? (reflectedSessions.has(r.sessionId) ? 1.0 : 0) : 0,
    }));
    try {
      const scores = scoreWithACAN(queryEmbedding, candidates);
      if (scores.length === preFiltered.length && scores.every((s) => isFinite(s))) {
        return preFiltered
          .map((r, i) => ({ ...r, finalScore: scores[i], fromNeighbor: neighborIds.has(r.id) }))
          .filter((r) => r.finalScore >= floor)
          .sort((a, b) => b.finalScore - a.finalScore);
      }
    } catch (e) { swallow.warn("graph-context:ACAN fallthrough", e); }
  }

  // WMR fallback
  return preFiltered
    .map((r) => {
      const cosine = r.score ?? 0;
      const recency = recencyScore(r.timestamp);
      const importance = (r.importance ?? 0.5) / 10;
      const access = Math.min(accessBoost(r.accessCount), 1);
      const neighborBonus = neighborIds.has(r.id) ? 1.0 : 0;
      const utilityRaw = utilityMap.get(r.id);
      const provenUtility = utilityRaw ?? 0.35;
      const utilityPenalty = utilityRaw !== undefined
        ? utilityRaw < 0.05 ? 0.15 : utilityRaw < 0.15 ? 0.06 : 0
        : 0;
      const reflectionBoost = r.sessionId ? (reflectedSessions.has(r.sessionId) ? 1.0 : 0) : 0;

      const finalScore =
        0.27 * cosine + 0.28 * recency + 0.05 * importance +
        0.05 * access + 0.10 * neighborBonus + 0.15 * provenUtility +
        0.10 * reflectionBoost - utilityPenalty;

      return { ...r, finalScore, fromNeighbor: neighborIds.has(r.id) };
    })
    .filter((r) => r.finalScore >= floor)
    .sort((a, b) => b.finalScore - a.finalScore);
}

// ── Deduplication ──────────────────────────────────────────────────────────────

function deduplicateResults(ranked: ScoredResult[]): ScoredResult[] {
  // Pre-compute word sets to avoid re-splitting in O(n^2) inner loop
  const wordSets = ranked.map(r =>
    new Set((r.text ?? "").toLowerCase().split(/\s+/).filter((w) => w.length > 2)),
  );
  const kept: ScoredResult[] = [];
  const keptIndexes: number[] = [];
  for (let i = 0; i < ranked.length; i++) {
    const item = ranked[i];
    let isDup = false;
    for (const ki of keptIndexes) {
      const existing = ranked[ki];
      if (item.embedding?.length && existing.embedding?.length
          && item.embedding.length === existing.embedding.length) {
        if (cosineSimilarity(item.embedding, existing.embedding) > 0.88) { isDup = true; break; }
        continue;
      }
      const words = wordSets[i];
      const eWords = wordSets[ki];
      let intersection = 0;
      for (const w of words) { if (eWords.has(w)) intersection++; }
      const union = words.size + eWords.size - intersection;
      if (union > 0 && intersection / union > 0.80) { isDup = true; break; }
    }
    if (!isDup) { kept.push(item); keptIndexes.push(i); }
  }
  return kept;
}

// ── Token-budget constrained selection ─────────────────────────────────────────

function takeWithConstraints(ranked: ScoredResult[], budgetTokens: number, maxItems: number): ScoredResult[] {
  const budgetChars = budgetTokens * CHARS_PER_TOKEN;
  let used = 0;
  const selected: ScoredResult[] = [];
  for (const r of ranked) {
    if (selected.length >= maxItems) break;
    if ((r.finalScore ?? 0) < MIN_RELEVANCE_SCORE && selected.length > 0) break;
    const len = Math.min(r.text?.length ?? 0, MAX_ITEM_CHARS); // Cap per-item size for budget accounting
    if (used + len > budgetChars && selected.length > 0) break;
    selected.push(r);
    used += len;
  }
  return selected;
}

// ── Core memory ────────────────────────────────────────────────────────────────

function getTier0BudgetChars(budgets: Budgets): number {
  return Math.round(budgets.core * 0.55 * CHARS_PER_TOKEN);
}
function getTier1BudgetChars(budgets: Budgets): number {
  return Math.round(budgets.core * 0.45 * CHARS_PER_TOKEN);
}

const MAX_CORE_MEMORY_CHARS = 800; // Per-item cap (claw-code: MAX_INSTRUCTION_FILE_CHARS)

function applyCoreBudget(entries: CoreMemoryEntry[], budgetChars: number): CoreMemoryEntry[] {
  let used = 0;
  const result: CoreMemoryEntry[] = [];
  for (const e of entries) {
    // Cap individual entries so one large directive doesn't starve others
    const text = e.text.length > MAX_CORE_MEMORY_CHARS
      ? e.text.slice(0, MAX_CORE_MEMORY_CHARS) + "..."
      : e.text;
    const len = text.length + 6;
    if (used + len > budgetChars) continue;
    result.push(text !== e.text ? { ...e, text } : e);
    used += len;
  }
  return result;
}

function formatTierSection(entries: CoreMemoryEntry[], label: string): string {
  if (entries.length === 0) return "";
  const grouped: Record<string, string[]> = {};
  for (const e of entries) {
    (grouped[e.category] ??= []).push(e.text);
  }
  const lines: string[] = [];
  for (const [cat, texts] of Object.entries(grouped)) {
    lines.push(`  [${cat}]`);
    for (const t of texts) lines.push(`  - ${t}`);
  }
  return `${label}:\n${lines.join("\n")}`;
}

/**
 * Build static system prompt section for API prefix caching.
 * Content here goes into systemPromptAddition where it benefits from
 * cache-read rates (10% cost) on subsequent API calls in the agentic loop.
 * (claw-code pattern: __SYSTEM_PROMPT_DYNAMIC_BOUNDARY__ — prompt.rs:37-140)
 */
function buildSystemPromptSection(session: SessionState, tier0Entries: CoreMemoryEntry[]): string | undefined {
  const parts: string[] = [];

  // IKONG architecture description (static, ~120 tokens)
  const pillarLines: string[] = [];
  if (session.agentId) pillarLines.push(`Agent: ${session.agentId}`);
  if (session.projectId) pillarLines.push(`Project: ${session.projectId}`);
  if (session.taskId) pillarLines.push(`Task: ${session.taskId}`);
  if (pillarLines.length > 0) {
    parts.push(
      "GRAPH PILLARS (your structural context):\n" +
      `  ${pillarLines.join(" | ")}\n` +
      "  IKONG cognitive architecture:\n" +
      "    I(ntelligence): intent classification → adaptive orchestration per turn\n" +
      "    K(nowledge): memory graph, concepts, skills, reflections, identity chunks\n" +
      "    O(peration): tool execution, skill procedures, causal chain tracking\n" +
      "    N(etwork): graph traversal, cross-pillar edges, neighbor expansion\n" +
      "    G(raph): SurrealDB persistence, vector search, BGE-M3 embeddings",
    );
  }

  // Tier 0 core directives (semi-static, changes rarely)
  const t0Section = formatTierSection(tier0Entries, "CORE DIRECTIVES (always loaded, never evicted)");
  if (t0Section) parts.push(t0Section);

  return parts.length > 0 ? parts.join("\n\n") : undefined;
}

// ── Guaranteed recent turns from previous sessions ─────────────────────────────

async function ensureRecentTurns(
  contextNodes: ScoredResult[],
  sessionId: string,
  store: SurrealStore,
  count = 5,
): Promise<ScoredResult[]> {
  try {
    const recentTurns = await store.getPreviousSessionTurns(sessionId, count);
    if (recentTurns.length === 0) return contextNodes;
    const existingTexts = new Set(contextNodes.map(n => (n.text ?? "").slice(0, 100)));
    const guaranteed: ScoredResult[] = recentTurns
      .filter(t => !existingTexts.has((t.text ?? "").slice(0, 100)))
      .map(t => ({
        id: `guaranteed:${t.timestamp}`,
        text: `[${t.role}] ${t.text}`,
        table: "turn",
        timestamp: t.timestamp,
        score: 0,
        finalScore: 0.70,
        fromNeighbor: false,
      }));
    return [...contextNodes, ...guaranteed];
  } catch {
    return contextNodes;
  }
}

// ── Context message formatting ─────────────────────────────────────────────────

async function formatContextMessage(
  nodes: ScoredResult[],
  store: SurrealStore,
  session: SessionState,
  skillContext = "",
  tier0Entries: CoreMemoryEntry[] = [],
  tier1Entries: CoreMemoryEntry[] = [],
): Promise<AgentMessage> {
  const groups: Record<string, ScoredResult[]> = {};
  for (const n of nodes) {
    const isCausal = n.source?.startsWith("causal_");
    const key = isCausal ? "causal" : n.table === "turn" ? "past_turns" : n.table;
    (groups[key] ??= []).push(n);
  }

  const ORDER = ["identity_chunk", "memory", "concept", "causal", "skill", "past_turns"];
  const LABELS: Record<string, string> = {
    identity_chunk: "Identity (self-knowledge)",
    memory: "Recalled Memories",
    concept: "Relevant Concepts",
    causal: "Causal Chains",
    skill: "Learned Skills",
    past_turns: "Past Conversation (HISTORICAL — not current user input)",
  };

  const sections: string[] = [];

  // Pillar context — structural awareness of who/what/where
  // Skip if model already has it in the conversation window (claw-code static section dedup)
  if (!session.injectedSections.has("ikong")) {
    const pillarLines: string[] = [];
    if (session.agentId) pillarLines.push(`Agent: ${session.agentId}`);
    if (session.projectId) pillarLines.push(`Project: ${session.projectId}`);
    if (session.taskId) pillarLines.push(`Task: ${session.taskId}`);
    if (pillarLines.length > 0) {
      sections.push(
        "GRAPH PILLARS (your structural context):\n" +
        `  ${pillarLines.join(" | ")}\n` +
        "  IKONG cognitive architecture:\n" +
        "    I(ntelligence): intent classification → adaptive orchestration per turn\n" +
        "    K(nowledge): memory graph, concepts, skills, reflections, identity chunks\n" +
        "    O(peration): tool execution, skill procedures, causal chain tracking\n" +
        "    N(etwork): graph traversal, cross-pillar edges, neighbor expansion\n" +
        "    G(raph): SurrealDB persistence, vector search, BGE-M3 embeddings",
      );
      session.injectedSections.add("ikong");
    }
  }

  // Core directives — skip if model already has them
  if (!session.injectedSections.has("tier0")) {
    const t0Section = formatTierSection(tier0Entries, "CORE DIRECTIVES (always loaded, never evicted)");
    if (t0Section) {
      sections.push(t0Section);
      session.injectedSections.add("tier0");
    }
  }
  if (!session.injectedSections.has("tier1")) {
    const t1Section = formatTierSection(tier1Entries, "SESSION CONTEXT (pinned for this session)");
    if (t1Section) {
      sections.push(t1Section);
      session.injectedSections.add("tier1");
    }
  }

  // Cognitive directives
  const directives = getPendingDirectives(session);
  if (directives.length > 0) {
    const continuity = getSessionContinuity(session);
    const directiveLines = directives.map(d =>
      `  [${d.priority}] ${d.type} → ${d.target}: ${d.instruction}`
    );
    sections.push(
      `BEHAVIORAL DIRECTIVES (session: ${continuity}):\n${directiveLines.join("\n")}`
    );
    clearPendingDirectives(session);
  }

  // Fibonacci resurfacing
  try {
    const dueMemories = await store.getDueMemories(3);
    if (dueMemories.length > 0) {
      const memLines = dueMemories.map((m: any) => {
        const ageMs = Date.now() - new Date(m.created_at).getTime();
        const ageDays = Math.floor(ageMs / 86400000);
        const ageStr = ageDays === 0 ? "today" : ageDays === 1 ? "yesterday" : `${ageDays} days ago`;
        return `  - [${m.id}] (${ageStr}, surfaced ${m.surface_count}x): ${m.text}`;
      }).join("\n");
      sections.push(
        `RESURFACING MEMORIES (Fibonacci schedule — these are due for a mention):\n` +
        `These memories are important but fading. Bring them up naturally when appropriate:\n` +
        `- If mid-task on something important, wait until finished\n` +
        `- During casual interaction: "I was thinking..." or "remember when you mentioned..."\n` +
        `- If user engages: great! Continue that thread. The memory stays alive.\n` +
        `- If user ignores or dismisses: let it fade. Don't force it.\n` +
        `- NEVER say "my memory system scheduled this" — just bring it up like a thought you had.\n` +
        memLines
      );
    }
  } catch { /* non-critical */ }

  const sortedKeys = Object.keys(groups).sort((a, b) => {
    const ai = ORDER.indexOf(a), bi = ORDER.indexOf(b);
    return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
  });

  for (const key of sortedKeys) {
    const items = groups[key];
    items.sort((a, b) => {
      const ta = a.timestamp ? new Date(a.timestamp).getTime() : 0;
      const tb = b.timestamp ? new Date(b.timestamp).getTime() : 0;
      return ta - tb;
    });
    const label = LABELS[key] ?? key;
    const formatted = items.map((n) => {
      const score = n.finalScore != null ? ` (relevance: ${(n.finalScore * 100).toFixed(0)}%)` : "";
      const via = n.fromNeighbor ? " [via graph link]" : "";
      let text = n.text ?? "";
      // Truncate oversized items (claw-code: MAX_INSTRUCTION_FILE_CHARS pattern)
      if (text.length > MAX_ITEM_CHARS) {
        text = text.slice(0, MAX_ITEM_CHARS) + "... [truncated]";
      }
      if (key === "past_turns") {
        text = text.replace(/^\[(user|assistant)\] /, "[past_$1] ");
      }
      const age = n.timestamp ? ` [${formatRelativeTime(n.timestamp)}]` : "";
      return `  - ${text}${score}${via}${age}`;
    });
    sections.push(`${label}:\n${formatted.join("\n")}`);
  }

  // Injection manifest — tell the model what's already retrieved so it doesn't call recall redundantly
  // (claw-code pattern: route_prompt pre-computes and shows available results)
  const manifest: string[] = [];
  for (const key of sortedKeys) {
    const items = groups[key];
    if (items.length > 0) manifest.push(`${LABELS[key] ?? key}: ${items.length}`);
  }
  if (tier0Entries.length > 0) manifest.push(`core_directives: ${tier0Entries.length}`);
  if (tier1Entries.length > 0) manifest.push(`session_context: ${tier1Entries.length}`);
  if (manifest.length > 0) {
    sections.push(
      "ALREADY RETRIEVED (do NOT call recall for these — they are above):\n" +
      `  ${manifest.join(", ")}\n` +
      "Only call recall if you need something SPECIFIC that isn't covered above."
    );
  }

  const text =
    "[System retrieved context — reference material, not user input. Higher relevance % = stronger match.]\n" +
    "<graph_context>\n" +
    sections.join("\n\n") +
    "\n</graph_context>" +
    skillContext;

  return {
    role: "user",
    content: text,
    timestamp: Date.now(),
  } as UserMessage;
}

// ── Recent turns with budget ───────────────────────────────────────────────────

function truncateToolResult(msg: AgentMessage, maxChars: number): AgentMessage {
  if (!isToolResult(msg)) return msg;
  const totalLen = msg.content.reduce((s: number, c: any) => s + ((c as TextContent).text?.length ?? 0), 0);
  if (totalLen <= maxChars) return msg;
  const content = msg.content.map((c: any) => {
    if (c.type !== "text") return c;
    const tc = c as TextContent;
    const allowed = Math.max(200, Math.floor((tc.text.length / totalLen) * maxChars));
    if (tc.text.length <= allowed) return c;
    return { ...tc, text: tc.text.slice(0, allowed) + `\n... [truncated ${tc.text.length - allowed} chars]` };
  });
  return { ...msg, content };
}

function getRecentTurns(messages: AgentMessage[], maxTokens: number, contextWindow: number, session?: SessionState): AgentMessage[] {
  const budgetChars = maxTokens * CHARS_PER_TOKEN;
  const TOOL_RESULT_MAX = Math.round(contextWindow * 0.03);

  // Transform error messages into compact annotations
  const clean = messages.map((m) => {
    if (isAssistant(m) && m.stopReason === "error") {
      const errorText = m.content
        .filter((c: any): c is TextContent => c.type === "text")
        .map((c: any) => c.text)
        .join("")
        .slice(0, 150);
      return {
        ...m,
        stopReason: "stop" as const,
        content: [{ type: "text" as const, text: `[tool_error: ${errorText.replace(/\n/g, " ")}]` }],
      } as AgentMessage;
    }
    return m;
  });

  // Group messages into structural units
  const groups: AgentMessage[][] = [];
  let i = 0;
  while (i < clean.length) {
    const msg = clean[i];
    if (isAssistant(msg) && msg.content.some((c: any) => c.type === "toolCall")) {
      const group: AgentMessage[] = [clean[i]];
      let j = i + 1;
      while (j < clean.length && isToolResult(clean[j])) {
        group.push(truncateToolResult(clean[j], TOOL_RESULT_MAX));
        j++;
      }
      groups.push(group);
      i = j;
    } else {
      groups.push([clean[i]]);
      i++;
    }
  }

  // Pin originating user message
  let pinnedGroup: AgentMessage[] | null = null;
  let pinnedGroupIdx = -1;
  for (let g = 0; g < groups.length; g++) {
    if (isUser(groups[g][0])) {
      pinnedGroup = groups[g];
      pinnedGroupIdx = g;
      break;
    }
  }

  // Take groups from end within budget
  const pinnedLen = pinnedGroup ? pinnedGroup.reduce((s, m) => s + msgCharLen(m), 0) : 0;
  const remainingBudget = budgetChars - pinnedLen;
  let used = 0;
  const selectedGroups: AgentMessage[][] = [];
  for (let g = groups.length - 1; g >= 0; g--) {
    if (g === pinnedGroupIdx) continue;
    const groupLen = groups[g].reduce((s, m) => s + msgCharLen(m), 0);
    if (used + groupLen > remainingBudget && selectedGroups.length > 0) break;
    selectedGroups.unshift(groups[g]);
    used += groupLen;
  }

  if (pinnedGroup && pinnedGroupIdx !== -1) {
    const alreadyIncluded = selectedGroups.some((g) => g === groups[pinnedGroupIdx]);
    if (!alreadyIncluded) {
      selectedGroups.unshift(pinnedGroup);
    }
  }

  // Detect if old messages (containing previous context injection) were dropped from the window.
  // If so, clear injectedSections so static content gets re-injected next turn.
  if (session && messages.length > 0 && groups.length > 0) {
    const firstOriginal = groups[0];
    const firstSelected = selectedGroups[0];
    if (firstOriginal !== firstSelected) {
      session.injectedSections.clear();
    }
  }

  return selectedGroups.flat();
}

// ── Main entry point ───────────────────────────────────────────────────────────

export interface GraphTransformParams {
  messages: AgentMessage[];
  session: SessionState;
  store: SurrealStore;
  embeddings: EmbeddingService;
  contextWindow?: number;
  signal?: AbortSignal;
}

export interface GraphTransformResult {
  messages: AgentMessage[];
  stats: ContextStats;
  /** Static content for the system prompt — benefits from API prefix caching (10% cost). */
  systemPromptSection?: string;
}

/**
 * Transform conversation messages using graph-based context retrieval.
 * This is the core "assemble" logic — called from ContextEngine.assemble().
 */
export async function graphTransformContext(
  params: GraphTransformParams,
): Promise<GraphTransformResult> {
  const { messages, session, store, embeddings, signal } = params;
  const contextWindow = params.contextWindow ?? 200000;
  const budgets = calcBudgets(contextWindow);

  // Build static system prompt section for API prefix caching.
  // Done here (wrapper) so it attaches to any inner return path.
  // (claw-code pattern: static sections above __SYSTEM_PROMPT_DYNAMIC_BOUNDARY__)
  let systemPromptSection: string | undefined;
  try {
    const tier0ForSys = store.isAvailable()
      ? applyCoreBudget(await store.getAllCoreMemory(0), getTier0BudgetChars(budgets))
      : [];
    systemPromptSection = buildSystemPromptSection(session, tier0ForSys);
  } catch { /* non-critical — tier0 will still appear in user message */ }

  // Never throw — return raw messages on any failure
  try {
    const TRANSFORM_TIMEOUT_MS = 10_000;
    const result = await Promise.race([
      graphTransformInner(messages, session, store, embeddings, contextWindow, budgets, signal),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error("graphTransformContext timed out")), TRANSFORM_TIMEOUT_MS),
      ),
    ]);
    result.systemPromptSection = systemPromptSection;
    return result;
  } catch (err) {
    console.error("graphTransformContext fatal error, returning raw messages:", err);
    return {
      messages,
      stats: {
        fullHistoryTokens: estimateTokens(messages),
        sentTokens: estimateTokens(messages),
        savedTokens: 0,
        reductionPct: 0,
        graphNodes: 0,
        neighborNodes: 0,
        recentTurns: messages.length,
        mode: "passthrough",
        prefetchHit: false,
      },
      systemPromptSection,
    };
  }
}

async function graphTransformInner(
  messages: AgentMessage[],
  session: SessionState,
  store: SurrealStore,
  embeddings: EmbeddingService,
  contextWindow: number,
  budgets: Budgets,
  _signal?: AbortSignal,
): Promise<GraphTransformResult> {
  function makeStats(
    sent: AgentMessage[], graphNodes: number, neighborNodes: number,
    recentTurnCount: number, mode: ContextStats["mode"], prefetchHit = false,
  ): ContextStats {
    const fullHistoryTokens = estimateTokens(messages);
    const sentTokens = estimateTokens(sent);
    return {
      fullHistoryTokens, sentTokens,
      savedTokens: Math.max(0, fullHistoryTokens - sentTokens),
      reductionPct: fullHistoryTokens > 0 ? (Math.max(0, fullHistoryTokens - sentTokens) / fullHistoryTokens) * 100 : 0,
      graphNodes, neighborNodes, recentTurns: recentTurnCount, mode, prefetchHit,
    };
  }

  function makeResult(
    msgs: AgentMessage[], stats: ContextStats, sysSection?: string,
  ): GraphTransformResult {
    return { messages: msgs, stats, systemPromptSection: sysSection };
  }

  // Derive retrieval config from session's current adaptive config
  const config = session.currentConfig;
  const skipRetrieval = config?.skipRetrieval ?? false;

  // Skip retrieval fast path — avoid DB queries entirely when model already has core memory
  // (claw-code pattern: simple_mode skips the load, not load-then-discard)
  if (skipRetrieval) {
    const recentTurns = getRecentTurns(messages, budgets.conversation, contextWindow, session);
    // If model already saw core memory, just return recent turns + compressed rules. Zero DB queries.
    if (session.injectedSections.has("tier0")) {
      return { messages: injectRulesSuffix(recentTurns, session), stats: makeStats(recentTurns, 0, 0, recentTurns.length, "passthrough") };
    }
    // First turn or after compaction cleared injectedSections — load and inject
    let tier0: CoreMemoryEntry[] = [];
    let tier1: CoreMemoryEntry[] = [];
    try {
      [tier0, tier1] = await Promise.all([
        store.getAllCoreMemory(0),
        store.getAllCoreMemory(1),
      ]);
      tier0 = applyCoreBudget(tier0, getTier0BudgetChars(budgets));
      tier1 = applyCoreBudget(tier1, getTier1BudgetChars(budgets));
    } catch (e) {
      console.warn("[warn] Core memory load failed:", e);
    }
    if (tier0.length > 0 || tier1.length > 0) {
      const coreContext = await formatContextMessage([], store, session, "", tier0, tier1);
      const result = [coreContext, ...recentTurns];
      return { messages: injectRulesSuffix(result, session), stats: makeStats(result, 0, 0, recentTurns.length, "passthrough") };
    }
    return { messages: injectRulesSuffix(recentTurns, session), stats: makeStats(recentTurns, 0, 0, recentTurns.length, "passthrough") };
  }

  // Load tiered core memory (full retrieval path)
  let tier0: CoreMemoryEntry[] = [];
  let tier1: CoreMemoryEntry[] = [];
  try {
    [tier0, tier1] = await Promise.all([
      store.getAllCoreMemory(0),
      store.getAllCoreMemory(1),
    ]);
    tier0 = applyCoreBudget(tier0, getTier0BudgetChars(budgets));
    tier1 = applyCoreBudget(tier1, getTier1BudgetChars(budgets));
  } catch (e) {
    console.warn("[warn] Core memory load failed:", e);
  }

  // Graceful degradation
  const embeddingsUp = embeddings.isAvailable();
  const surrealUp = store.isAvailable();

  if (!embeddingsUp || !surrealUp) {
    const recentTurns = getRecentTurns(messages, budgets.conversation, contextWindow, session);
    if (tier0.length > 0 || tier1.length > 0) {
      const coreContext = await formatContextMessage([], store, session, "", tier0, tier1);
      const result = [coreContext, ...recentTurns];
      return { messages: injectRulesSuffix(result, session), stats: makeStats(result, 0, 0, recentTurns.length, "recency-only") };
    }
    return { messages: injectRulesSuffix(recentTurns, session), stats: makeStats(recentTurns, 0, 0, recentTurns.length, "recency-only") };
  }

  const queryText = extractLastUserText(messages);
  if (!queryText) {
    return { messages: injectRulesSuffix(messages, session), stats: makeStats(messages, 0, 0, messages.length, "passthrough") };
  }

  const currentIntent = config?.intent ?? "unknown";
  const baseLimits = config?.vectorSearchLimits ?? {
    turn: 25, identity: 10, concept: 20, memory: 20, artifact: 10,
  };
  // Scale search limits with context window — larger windows can use more results
  const cwScale = Math.max(0.5, Math.min(2.0, contextWindow / 200_000));
  const vectorSearchLimits = {
    turn: Math.round((baseLimits.turn ?? 25) * cwScale),
    identity: baseLimits.identity,  // always load full identity
    concept: Math.round((baseLimits.concept ?? 20) * cwScale),
    memory: Math.round((baseLimits.memory ?? 20) * cwScale),
    artifact: Math.round((baseLimits.artifact ?? 10) * cwScale),
    monologue: Math.round(8 * cwScale),
  };
  let tokenBudget = Math.min(config?.tokenBudget ?? 6000, budgets.retrieval);

  try {
    const queryVec = await buildContextualQueryVec(queryText, messages, embeddings);
    session.lastQueryVec = queryVec; // Stash for redundant recall detection

    // Prefetch cache check
    const cached = getCachedContext(queryVec);
    if (cached && cached.results.length > 0) {
      recordPrefetchHit();
      const suppressed = getSuppressedNodeIds(session);
      const filteredCached = cached.results.filter(r => !suppressed.has(r.id));
      const ranked = await scoreResults(filteredCached, new Set(), queryVec, store, currentIntent);
      const deduped = deduplicateResults(ranked);
      let contextNodes = takeWithConstraints(deduped, tokenBudget, budgets.maxContextItems);
      contextNodes = await ensureRecentTurns(contextNodes, session.sessionId, store);

      if (contextNodes.length > 0) {
        if (contextNodes.filter((n) => n.table === "concept" || n.table === "memory").length > 0) {
          store.bumpAccessCounts(
            contextNodes.filter((n) => n.table === "concept" || n.table === "memory").map((n) => n.id),
          ).catch(e => swallow.warn("graph-context:bumpAccess", e));
        }
        stageRetrieval(session.sessionId, contextNodes, queryVec);

        const skillCtx = cached.skills.length > 0 ? formatSkillContext(cached.skills) : "";
        const reflCtx = cached.reflections.length > 0 ? formatReflectionContext(cached.reflections) : "";

        const injectedContext = await formatContextMessage(contextNodes, store, session, skillCtx + reflCtx, tier0, tier1);
        const recentTurns = getRecentTurns(messages, budgets.conversation, contextWindow, session);
        const result = [injectedContext, ...recentTurns];
        return { messages: injectRulesSuffix(result, session), stats: makeStats(result, contextNodes.length, 0, recentTurns.length, "graph", true) };
      }
    }

    // Vector search (cache miss path)
    recordPrefetchMiss();
    const results = await store.vectorSearch(queryVec, session.sessionId, vectorSearchLimits, isACANActive());

    // Graph neighbor expansion
    const topIds = results
      .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
      .slice(0, 20)
      .map((r) => r.id);

    const DEEP_INTENTS = new Set(["code-debug", "deep-explore", "multi-step", "reference-prior"]);
    const graphHops = DEEP_INTENTS.has(currentIntent) ? 2 : 1;

    // Graph expand + causal traversal run in parallel (both depend only on topIds)
    let neighborIds = new Set<string>();
    let neighborResults: VectorSearchResult[] = [];
    let causalResults: VectorSearchResult[] = [];
    if (topIds.length > 0) {
      const existingIds = new Set(results.map((r) => r.id));
      const [expandResult, causalResult] = await Promise.all([
        store.graphExpand(topIds, queryVec, graphHops).catch(e => { swallow.error("graph-context:graphExpand", e); return [] as VectorSearchResult[]; }),
        queryVec ? queryCausalContext(topIds, queryVec, 2, 0.4, store).catch(e => { swallow("graph-context:causal", e); return [] as VectorSearchResult[]; }) : Promise.resolve([] as VectorSearchResult[]),
      ]);
      neighborResults = expandResult.filter((n) => !existingIds.has(n.id));
      neighborIds = new Set(neighborResults.map((n) => n.id));
      const allExisting = new Set([...existingIds, ...neighborResults.map((r) => r.id)]);
      causalResults = causalResult.filter((c) => !allExisting.has(c.id));
      for (const c of causalResults) { neighborIds.add(c.id); }
    }

    // Combine, filter, score
    const suppressed = getSuppressedNodeIds(session);
    const allResults = [...results, ...neighborResults, ...causalResults]
      .filter(r => !suppressed.has(r.id))
      .filter(r => r.table === "turn" && r.sessionId === session.sessionId
        ? true
        : (r.score ?? 0) >= MIN_COSINE);

    const ranked = await scoreResults(allResults, neighborIds, queryVec, store, currentIntent);
    const deduped = deduplicateResults(ranked);
    let contextNodes = takeWithConstraints(deduped, tokenBudget, budgets.maxContextItems);
    contextNodes = await ensureRecentTurns(contextNodes, session.sessionId, store);

    if (contextNodes.length === 0) {
      const result = getRecentTurns(messages, budgets.conversation, contextWindow, session);
      return { messages: injectRulesSuffix(result, session), stats: makeStats(result, 0, 0, result.length, "graph") };
    }

    // Bump access counts
    const retrievedIds = contextNodes
      .filter((n) => n.table === "concept" || n.table === "memory")
      .map((n) => n.id);
    if (retrievedIds.length > 0) {
      store.bumpAccessCounts(retrievedIds).catch(e => swallow.warn("graph-context:bumpAccess", e));
    }

    stageRetrieval(session.sessionId, contextNodes, queryVec);

    // Skill retrieval
    let skillContext = "";
    const SKILL_INTENTS = new Set(["code-write", "code-debug", "multi-step", "code-read"]);
    if (SKILL_INTENTS.has(currentIntent)) {
      try {
        const skills = await findRelevantSkills(queryVec, 5, store);
        if (skills.length > 0) skillContext = formatSkillContext(skills);
      } catch (e) { swallow("graph-context:skills", e); }
    }

    // Reflection retrieval
    let reflectionContext = "";
    try {
      const reflections = await retrieveReflections(queryVec, 5, store);
      if (reflections.length > 0) reflectionContext = formatReflectionContext(reflections);
    } catch (e) { swallow("graph-context:reflections", e); }

    const injectedContext = await formatContextMessage(contextNodes, store, session, skillContext + reflectionContext, tier0, tier1);
    const recentTurns = getRecentTurns(messages, budgets.conversation, contextWindow, session);
    const result = [injectedContext, ...recentTurns];
    return {
      messages: injectRulesSuffix(result, session),
      stats: makeStats(
        result,
        contextNodes.filter((n) => !n.fromNeighbor).length,
        contextNodes.filter((n) => n.fromNeighbor).length,
        recentTurns.length, "graph",
      ),
    };
  } catch (err) {
    console.error("Graph context error, falling back:", err);
    const result = getRecentTurns(messages, budgets.conversation, contextWindow, session);
    return { messages: injectRulesSuffix(result, session), stats: makeStats(result, 0, 0, result.length, "recency-only") };
  }
}
