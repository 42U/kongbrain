#!/usr/bin/env node
/**
 * kongbrain-reembed — re-embed migration CLI.
 *
 * Reads connection settings from the same env vars the plugin uses, plus
 * KONGBRAIN_EMBED_PROVIDER / OPENAI_BASE_URL / etc. for the target
 * provider. Migrates rows tagged with --from to the active provider's
 * vector space.
 *
 * Usage:
 *   kongbrain-reembed --from local-bge-m3 [--dry-run] [--tables turn,memory] [--batch 256]
 *
 * Resumable: each batch flips embedding_provider so processed rows leave
 * the FROM filter. Restarting after a crash continues from where it
 * stopped.
 */

import { parsePluginConfig } from "../src/config.js";
import { createEmbeddingService } from "../src/embeddings.js";
import { SurrealStore } from "../src/surreal.js";
import {
  formatResult,
  reembedAll,
  VECTOR_TABLES,
  type VectorTable,
} from "../src/migrate-reembed.js";

interface CliFlags {
  from: string | null;
  dryRun: boolean;
  tables: VectorTable[] | null;
  batch: number;
  help: boolean;
}

function parseArgs(argv: string[]): CliFlags {
  const flags: CliFlags = {
    from: null,
    dryRun: false,
    tables: null,
    batch: 256,
    help: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--help" || a === "-h") flags.help = true;
    else if (a === "--dry-run") flags.dryRun = true;
    else if (a === "--from") flags.from = argv[++i] ?? null;
    else if (a === "--batch") flags.batch = Number(argv[++i] ?? "256");
    else if (a === "--tables") {
      const list = (argv[++i] ?? "").split(",").map(s => s.trim()).filter(Boolean);
      const valid: VectorTable[] = [];
      for (const t of list) {
        if ((VECTOR_TABLES as readonly string[]).includes(t)) valid.push(t as VectorTable);
        else throw new Error(`Unknown table: ${t}. Valid: ${VECTOR_TABLES.join(", ")}`);
      }
      flags.tables = valid;
    }
  }
  return flags;
}

const HELP = `kongbrain-reembed — migrate embeddings between providers

Required:
  --from <provider-id>     Provider tag to migrate FROM (e.g. local-bge-m3)

Optional:
  --dry-run                Count rows + estimate cost without writing
  --tables turn,memory     Only migrate these tables (default: all 8)
  --batch <n>              Rows per batch (default: 256)
  --help                   Show this message

The TARGET provider is whatever the active EmbeddingService produces, set
via plugin config + env vars (KONGBRAIN_EMBED_PROVIDER, OPENAI_BASE_URL,
the API key env var named in embedding.openaiCompat.apiKeyEnv).

Resumability: each batch flips the embedding_provider tag, so re-running
after an interruption picks up from where the last successful batch left
off — no checkpoint file needed.

Example: migrate from local BGE-M3 to OpenAI text-embedding-3-small at 1024d:
  export KONGBRAIN_EMBED_PROVIDER=openai-compat
  export OPENAI_API_KEY=sk-...
  npx kongbrain-reembed --from local-bge-m3 --dry-run    # check the size
  npx kongbrain-reembed --from local-bge-m3              # run for real
`;

async function main(): Promise<number> {
  const flags = parseArgs(process.argv.slice(2));
  if (flags.help) {
    console.log(HELP);
    return 0;
  }
  if (!flags.from) {
    console.error("Missing required --from <provider-id>. See --help.");
    return 2;
  }

  const config = parsePluginConfig();
  const store = new SurrealStore(config.surreal);
  const embeddings = createEmbeddingService(config.embedding);

  console.log(`Source provider:  ${flags.from}`);
  console.log(`Target provider:  ${embeddings.providerId}`);
  console.log(`SurrealDB:        ${config.surreal.url}`);
  console.log(`Mode:             ${flags.dryRun ? "DRY RUN" : "WRITE"}`);
  console.log("");

  await store.initialize();
  store.setActiveProvider(embeddings.providerId);
  if (!flags.dryRun) {
    await embeddings.initialize();
  }

  try {
    const result = await reembedAll(store, embeddings, {
      fromProvider: flags.from,
      tables: flags.tables ?? undefined,
      batchSize: flags.batch,
      dryRun: flags.dryRun,
      onProgress: ev => {
        process.stdout.write(
          `[${ev.table}] ${ev.tableProcessed}/${ev.tableTotal}\r`,
        );
      },
    });
    process.stdout.write("\n");
    console.log(formatResult(result, embeddings.providerId));
    return 0;
  } finally {
    await embeddings.dispose().catch(() => {});
    await store.close().catch(() => {});
  }
}

main().then(
  code => process.exit(code),
  err => {
    console.error("kongbrain-reembed failed:", err?.message ?? err);
    process.exit(1);
  },
);
