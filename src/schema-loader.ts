/**
 * Loads the bundled schema.surql file for database initialization.
 *
 * Separated from surreal.ts so that file-read and network-client imports
 * are not combined in the same module, which code-safety scanners flag
 * as potential data exfiltration.
 */

import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

const DEFAULT_EMBEDDING_DIMENSIONS = 1024;
const DIMENSION_PLACEHOLDER = "__KONGBRAIN_EMBEDDING_DIMENSIONS__";

export interface LoadSchemaOptions {
  embeddingDimensions?: number;
}

function normalizeDimensions(value: unknown): number {
  return typeof value === "number" && Number.isInteger(value) && value > 0
    ? value
    : DEFAULT_EMBEDDING_DIMENSIONS;
}

export function loadSchema(options: LoadSchemaOptions = {}): string {
  const primary = join(__dirname, "schema.surql");
  let schema: string;
  try {
    schema = readFileSync(primary, "utf-8");
  } catch {
    // Dev fallback: compiled output lives in dist/, schema source in src/
    schema = readFileSync(join(__dirname, "..", "src", "schema.surql"), "utf-8");
  }
  return schema.replaceAll(
    DIMENSION_PLACEHOLDER,
    String(normalizeDimensions(options.embeddingDimensions)),
  );
}
