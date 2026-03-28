import { homedir } from "node:os";
import { join } from "node:path";

export interface SurrealConfig {
  url: string;
  httpUrl: string;
  user: string;
  pass: string;
  ns: string;
  db: string;
}

export interface EmbeddingConfig {
  modelPath: string;
  dimensions: number;
}

export interface KongBrainConfig {
  surreal: SurrealConfig;
  embedding: EmbeddingConfig;
}

/**
 * Parse plugin config from openclaw.plugin.json configSchema values,
 * with env var overrides and sensible defaults.
 */
export function parsePluginConfig(raw?: Record<string, unknown>): KongBrainConfig {
  const surreal = (raw?.surreal ?? {}) as Record<string, unknown>;
  const embedding = (raw?.embedding ?? {}) as Record<string, unknown>;

  const url =
    process.env.SURREAL_URL ??
    (typeof surreal.url === "string" ? surreal.url : "ws://localhost:8042/rpc");

  return {
    surreal: {
      url,
      get httpUrl() {
        const override = process.env.SURREAL_HTTP_URL;
        if (override) return override;
        return this.url
          .replace("ws://", "http://")
          .replace("wss://", "https://")
          .replace("/rpc", "/sql");
      },
      user: process.env.SURREAL_USER ?? (typeof surreal.user === "string" ? surreal.user : "root"),
      pass: process.env.SURREAL_PASS ?? (typeof surreal.pass === "string" ? surreal.pass : "root"),
      ns: process.env.SURREAL_NS ?? (typeof surreal.ns === "string" ? surreal.ns : "kong"),
      db: process.env.SURREAL_DB ?? (typeof surreal.db === "string" ? surreal.db : "memory"),
    },
    embedding: {
      modelPath:
        process.env.EMBED_MODEL_PATH ??
        (typeof embedding.modelPath === "string"
          ? embedding.modelPath
          : join(homedir(), ".node-llama-cpp", "models", "bge-m3-q4_k_m.gguf")),
      dimensions:
        typeof embedding.dimensions === "number" ? embedding.dimensions : 1024,
    },
  };
}
