#!/usr/bin/env npx tsx
/**
 * One-shot backfill: embed all concepts that have `content` but no embedding vector.
 *
 * Usage:
 *   cd /home/zero/voidorigin/kongbrain
 *   npx tsx scripts/backfill-embeddings.ts
 *
 * Env vars (all have defaults matching the plugin config):
 *   SURREAL_URL       (default: ws://localhost:8042/rpc)
 *   SURREAL_USER      (default: root)
 *   SURREAL_PASS      (default: root)
 *   SURREAL_NS        (default: kong)
 *   SURREAL_DB        (default: memory)
 *   EMBED_MODEL_PATH  (default: ~/.node-llama-cpp/models/bge-m3-q4_k_m.gguf)
 */

import { parsePluginConfig } from "../src/config.js";
import { SurrealStore } from "../src/surreal.js";
import { EmbeddingService } from "../src/embeddings.js";

async function main() {
  const config = parsePluginConfig();
  const store = new SurrealStore(config.surreal);
  const embeddings = new EmbeddingService(config.embedding);

  console.log("[backfill] Connecting to SurrealDB...");
  await store.initialize();

  console.log("[backfill] Loading embedding model...");
  await embeddings.initialize();

  // Find concepts with content but no embedding
  const bare = await store.queryFirst<{ id: string; content: string }>(
    `SELECT id, content FROM concept
     WHERE content IS NOT NONE AND content != ''
       AND (embedding IS NONE OR array::len(embedding) = 0)`,
  );

  console.log(`[backfill] Found ${bare.length} concepts needing embeddings.`);
  if (bare.length === 0) {
    console.log("[backfill] Nothing to do.");
    await embeddings.dispose();
    return;
  }

  let ok = 0;
  let fail = 0;

  for (const concept of bare) {
    const id = String(concept.id);
    try {
      const vec = await embeddings.embed(concept.content);
      await store.queryExec(
        `UPDATE ${id} SET embedding = $emb`,
        { emb: vec },
      );
      ok++;
      if (ok % 10 === 0) console.log(`[backfill] ${ok}/${bare.length} done...`);
    } catch (e) {
      fail++;
      console.error(`[backfill] Failed ${id}: ${e}`);
    }
  }

  console.log(`[backfill] Complete. Embedded: ${ok}, Failed: ${fail}`);
  await embeddings.dispose();
  process.exit(0);
}

main().catch((e) => {
  console.error("[backfill] Fatal:", e);
  process.exit(1);
});
