/**
 * Re-embed migration: take rows tagged with one provider and rewrite their
 * embeddings using another provider, updating the embedding_provider tag in
 * the same UPDATE.
 *
 * Resumability: each table is processed in batches of `batchSize` rows
 * matching `embedding_provider = $fromProvider`. After a batch is written,
 * those rows no longer match the filter, so a subsequent run picks up from
 * where the previous one stopped.
 *
 * The text re-embedded for each row is the canonical text field for that
 * table (e.g. concept.content, turn.text). For tables where the original
 * write site embedded a composed string (skill: "name: description"), we
 * reproduce that composition here so the new vectors live in roughly the
 * same conceptual neighborhood as the originals.
 */

import type { EmbeddingService } from "./embeddings.js";
import type { SurrealStore } from "./surreal.js";
import { swallow } from "./errors.js";

/** Tables that store embeddings and need to participate in re-embed. */
export const VECTOR_TABLES = [
  "turn",
  "concept",
  "memory",
  "artifact",
  "identity_chunk",
  "skill",
  "reflection",
  "monologue",
] as const;

export type VectorTable = typeof VECTOR_TABLES[number];

/**
 * Per-table mapping from the row shape to the text that should be embedded.
 * Different tables call their text field different things; skill composes
 * its embedding text from name + description.
 */
type RowTextExtractor = (row: Record<string, unknown>) => string;

const TEXT_EXTRACTORS: Record<VectorTable, RowTextExtractor> = {
  turn: r => String(r.text ?? ""),
  concept: r => String(r.content ?? ""),
  memory: r => String(r.text ?? ""),
  artifact: r => {
    // Match what workspace-migrate.ts does for content-rich artifacts when
    // possible. When content is short, embed it; otherwise embed a header
    // plus a content excerpt.
    const description = String(r.description ?? "");
    const content = String(r.content ?? "");
    if (!content) return description;
    if (content.length < 2000) return content;
    return `${description}\n${content.slice(0, 1500)}`;
  },
  identity_chunk: r => String(r.text ?? ""),
  // skills.ts embeds `${name}: ${description}` — preserve that.
  skill: r => `${String(r.name ?? "")}: ${String(r.description ?? "")}`.trim(),
  reflection: r => String(r.text ?? ""),
  monologue: r => String(r.content ?? ""),
};

/** Fields a row must select for migration (per table). */
const SELECT_FIELDS: Record<VectorTable, string> = {
  turn: "id, text",
  concept: "id, content",
  memory: "id, text",
  artifact: "id, description, content",
  identity_chunk: "id, text",
  skill: "id, name, description",
  reflection: "id, text",
  monologue: "id, content",
};

export interface ReembedOptions {
  /** Provider id rows should be migrated FROM. Required. */
  fromProvider: string;
  /** Provider id to migrate TO. Defaults to `embeddings.providerId`. */
  toProvider?: string;
  /** Tables to migrate. Defaults to all 8 vector tables. */
  tables?: VectorTable[];
  /** Rows fetched + embedded per batch. Default 256. */
  batchSize?: number;
  /** When true, count rows + estimate cost without writing anything. */
  dryRun?: boolean;
  /** Optional progress callback per batch. */
  onProgress?: (event: ProgressEvent) => void;
}

export interface ProgressEvent {
  table: VectorTable;
  /** Rows processed in this batch. */
  batchSize: number;
  /** Cumulative rows processed for this table. */
  tableProcessed: number;
  /** Total rows (counted at start) for this table. */
  tableTotal: number;
}

export interface ReembedResult {
  /** Total rows updated (or counted, when dryRun). */
  total: number;
  /** Per-table breakdown. */
  perTable: Record<VectorTable, number>;
  /** Approximate input character count (sum of text lengths). */
  approxChars: number;
  /** Approximate input token count using a chars/4 heuristic. */
  approxTokens: number;
  /** True if no writes were performed. */
  dryRun: boolean;
  /** Wall clock duration in ms. */
  durationMs: number;
}

/**
 * Approximate token count using the chars/4 heuristic. Real tokenization
 * varies by model; this estimate is good enough for cost ballparks.
 */
function approxTokenCount(chars: number): number {
  return Math.ceil(chars / 4);
}

/**
 * Migrate rows from one provider to another, batching reads + writes.
 *
 * The embedding service passed in MUST already be initialized and produce
 * vectors in the target provider's space. The function does NOT switch
 * providers itself — that is a config-time decision.
 */
export async function reembedAll(
  store: SurrealStore,
  embeddings: EmbeddingService,
  opts: ReembedOptions,
): Promise<ReembedResult> {
  const startedAt = Date.now();
  const tables: VectorTable[] = opts.tables ?? [...VECTOR_TABLES];
  const batchSize = opts.batchSize ?? 256;
  const fromProvider = opts.fromProvider;
  const toProvider = opts.toProvider ?? embeddings.providerId;
  const dryRun = opts.dryRun ?? false;

  if (fromProvider === toProvider) {
    throw new Error(
      `reembedAll: fromProvider (${fromProvider}) and toProvider (${toProvider}) are identical — nothing to do.`,
    );
  }
  if (!dryRun && !embeddings.isAvailable()) {
    throw new Error("reembedAll: embedding service is not initialized.");
  }
  if (!store.isAvailable()) {
    throw new Error("reembedAll: SurrealStore is not initialized.");
  }

  const perTable: Record<VectorTable, number> = Object.fromEntries(
    VECTOR_TABLES.map(t => [t, 0]),
  ) as Record<VectorTable, number>;
  let approxChars = 0;

  for (const table of tables) {
    // Count the rows we'll touch up front so onProgress can report
    // progress against a total. Cheap with the embedding_provider index.
    const countRows = await store.queryFirst<{ count: number }>(
      `SELECT count() AS count FROM ${table}
       WHERE embedding != NONE AND embedding_provider = $provider
       GROUP ALL`,
      { provider: fromProvider },
    );
    const tableTotal = Number(countRows[0]?.count ?? 0);
    if (tableTotal === 0) continue;

    let tableProcessed = 0;
    while (true) {
      const rows = await store.queryFirst<Record<string, unknown>>(
        `SELECT ${SELECT_FIELDS[table]} FROM ${table}
         WHERE embedding != NONE AND embedding_provider = $provider
         LIMIT $lim`,
        { provider: fromProvider, lim: batchSize },
      );
      if (rows.length === 0) break;

      const extract = TEXT_EXTRACTORS[table];
      const texts = rows.map(r => extract(r));
      // Skip blanks: nothing useful to embed, but we still need to flip
      // the provider tag so the row stops matching the FROM filter and
      // the loop terminates. We set embedding to NONE to keep it out of
      // the index entirely.
      const blankIndices: number[] = [];
      const realIndices: number[] = [];
      const realTexts: string[] = [];
      for (let i = 0; i < rows.length; i++) {
        const t = texts[i];
        if (!t || t.trim().length === 0) blankIndices.push(i);
        else { realIndices.push(i); realTexts.push(t); }
      }

      // Sum chars before any embed call so dry-run accumulates the same
      // way as the real run.
      for (const t of realTexts) approxChars += t.length;

      if (dryRun) {
        // Count and continue without writing.
        tableProcessed += rows.length;
        perTable[table] = tableProcessed;
        opts.onProgress?.({
          table,
          batchSize: rows.length,
          tableProcessed,
          tableTotal,
        });
        // In dry-run we cannot move past this batch (we did not flip
        // provider tags), so break after recording the first batch's
        // count and rely on the up-front count() instead.
        perTable[table] = tableTotal;
        break;
      }

      // Real run: embed in one batched call (provider implementations
      // chunk internally if needed).
      let vecs: number[][] = [];
      if (realTexts.length > 0) {
        vecs = await embeddings.embedBatch(realTexts);
        if (vecs.length !== realTexts.length) {
          throw new Error(
            `reembedAll[${table}]: embedBatch returned ${vecs.length} vectors for ${realTexts.length} inputs.`,
          );
        }
      }

      // Write back: one UPDATE per row. Could be batched into a single
      // queryBatch call for speed, but the simpler form is easier to
      // reason about for resumability and is bounded by batchSize.
      for (let j = 0; j < realIndices.length; j++) {
        const row = rows[realIndices[j]];
        const id = String(row.id);
        try {
          await store.queryExec(
            `UPDATE ${id} SET embedding = $emb, embedding_provider = $provider`,
            { emb: vecs[j], provider: toProvider },
          );
        } catch (e) {
          swallow.warn(`reembed:update:${table}`, e);
        }
      }
      // Blank-text rows: drop the embedding and flip the tag so they
      // exit the FROM filter (otherwise we loop forever).
      for (const idx of blankIndices) {
        const row = rows[idx];
        const id = String(row.id);
        try {
          await store.queryExec(
            `UPDATE ${id} SET embedding = NONE, embedding_provider = NONE`,
          );
        } catch (e) {
          swallow.warn(`reembed:blank:${table}`, e);
        }
      }

      tableProcessed += rows.length;
      perTable[table] = tableProcessed;
      opts.onProgress?.({
        table,
        batchSize: rows.length,
        tableProcessed,
        tableTotal,
      });

      // Loop again unless the batch was undersized (no more to do).
      if (rows.length < batchSize) break;
    }
  }

  const total = Object.values(perTable).reduce((a, b) => a + b, 0);
  return {
    total,
    perTable,
    approxChars,
    approxTokens: approxTokenCount(approxChars),
    dryRun,
    durationMs: Date.now() - startedAt,
  };
}

/**
 * Format a result for human display. Used by the CLI; exposed so callers
 * who embed the migrator into their own UIs can reuse the formatting.
 */
export function formatResult(result: ReembedResult, toProvider: string): string {
  const lines: string[] = [];
  lines.push(result.dryRun ? "DRY RUN — no writes performed." : "Migration complete.");
  lines.push(`Target provider: ${toProvider}`);
  lines.push(`Rows ${result.dryRun ? "to be migrated" : "migrated"}: ${result.total}`);
  for (const t of VECTOR_TABLES) {
    const n = result.perTable[t];
    if (n > 0) lines.push(`  ${t}: ${n}`);
  }
  lines.push(`Approx input: ${result.approxChars.toLocaleString()} chars (~${result.approxTokens.toLocaleString()} tokens)`);
  // text-embedding-3-small is $0.02/1M tokens; -3-large is $0.13/1M.
  // We don't know which model the caller is using, so report both.
  const small = (result.approxTokens / 1_000_000) * 0.02;
  const large = (result.approxTokens / 1_000_000) * 0.13;
  lines.push(`Estimated cost: $${small.toFixed(4)} (text-embedding-3-small) | $${large.toFixed(4)} (text-embedding-3-large)`);
  lines.push(`Duration: ${(result.durationMs / 1000).toFixed(2)}s`);
  return lines.join("\n");
}
