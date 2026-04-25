import { existsSync } from "node:fs";
import type { EmbeddingConfig } from "./config.js";
import { OpenAICompatEmbeddingService } from "./embeddings-openai.js";
import { swallow } from "./errors.js";
import { log } from "./log.js";

// Lazy-import node-llama-cpp to avoid top-level await issues with jiti.
// The actual import happens inside initialize() at runtime.
type LlamaEmbeddingContext = import("node-llama-cpp").LlamaEmbeddingContext;
type LlamaModel = import("node-llama-cpp").LlamaModel;

/**
 * Provider-agnostic embedding service.
 *
 * Implementations must guarantee that vectors they produce are in the same
 * vector space across calls within a single instance. Different implementations
 * (or different models within the same implementation) produce vectors in
 * different spaces and must not be compared with cosine similarity. The
 * `providerId` field is the stable tag used to detect cross-space mixing.
 */
export interface EmbeddingService {
  /** Stable identifier for the (provider, model, dimension) tuple. */
  readonly providerId: string;
  /** Dimensionality of the vectors this service produces. */
  readonly dimensions: number;

  /** Initialize the underlying model. Returns true on first init, false if already ready. */
  initialize(): Promise<boolean>;
  /** Return the embedding vector for a single text. */
  embed(text: string): Promise<number[]>;
  /** Return embedding vectors for an array of texts. */
  embedBatch(texts: string[]): Promise<number[][]>;
  /** True once initialize() has succeeded. */
  isAvailable(): boolean;
  /** Release any underlying resources (model handles, sockets, etc.). */
  dispose(): Promise<void>;
}

/** BGE-M3 embedding service (1024-dim via GGUF) with an LRU cache of up to 512 entries. */
export class LocalEmbeddingService implements EmbeddingService {
  readonly providerId: string;
  readonly dimensions: number;

  private model: LlamaModel | null = null;
  private ctx: LlamaEmbeddingContext | null = null;
  private ready = false;
  /** LRU embedding cache keyed by text, capped at maxCacheSize entries. */
  private cache = new Map<string, number[]>();
  private readonly maxCacheSize = 512;

  constructor(private readonly config: EmbeddingConfig) {
    this.providerId = "local-bge-m3";
    this.dimensions = config.dimensions;
  }

  async initialize(): Promise<boolean> {
    if (this.ready) return false;
    if (!existsSync(this.config.modelPath)) {
      throw new Error(
        `Embedding model not found at: ${this.config.modelPath}\n  Download BGE-M3 GGUF or set EMBED_MODEL_PATH`,
      );
    }
    const { getLlama, LlamaLogLevel } = await import("node-llama-cpp");
    const llama = await getLlama({
      logLevel: LlamaLogLevel.error,
      logger: (level, message) => {
        if (message.includes("missing newline token")) return;
        if (level === LlamaLogLevel.error) log.error(`[llama] ${message}`);
        else if (level === LlamaLogLevel.warn) log.warn(`[llama] ${message}`);
      },
    });
    this.model = await llama.loadModel({ modelPath: this.config.modelPath });
    this.ctx = await this.model.createEmbeddingContext();
    this.ready = true;
    return true;
  }

  async embed(text: string): Promise<number[]> {
    if (!this.ready || !this.ctx) throw new Error("Embeddings not initialized");
    const cached = this.cache.get(text);
    if (cached) {
      this.cache.delete(text);
      this.cache.set(text, cached);
      return cached;
    }
    const result = await this.ctx.getEmbeddingFor(text);
    const vec = Array.from(result.vector);
    if (this.cache.size >= this.maxCacheSize) {
      this.cache.delete(this.cache.keys().next().value!);
    }
    this.cache.set(text, vec);
    return vec;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    return Promise.all(texts.map(text => this.embed(text)));
  }

  isAvailable(): boolean {
    return this.ready;
  }

  async dispose(): Promise<void> {
    try {
      await this.ctx?.dispose();
      await this.model?.dispose();
      this.ready = false;
      this.cache.clear();
    } catch (e) {
      swallow("embeddings:dispose", e);
    }
  }
}

/** Construct the configured embedding service. Adding a new provider plugs in here. */
export function createEmbeddingService(config: EmbeddingConfig): EmbeddingService {
  if (config.provider === "openai-compat") {
    // Lazy import keeps the local-only deployment path from paying the cost
    // of parsing the OpenAI module on startup.
    const { OpenAICompatEmbeddingService } = require("./embeddings-openai.js") as
      typeof import("./embeddings-openai.js");
    return new OpenAICompatEmbeddingService(config);
  }
  return new LocalEmbeddingService(config);
}
