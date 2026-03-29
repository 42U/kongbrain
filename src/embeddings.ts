import { existsSync } from "node:fs";
import type { EmbeddingConfig } from "./config.js";
import { swallow } from "./errors.js";

// Lazy-import node-llama-cpp to avoid top-level await issues with jiti.
// The actual import happens inside initialize() at runtime.
type LlamaEmbeddingContext = import("node-llama-cpp").LlamaEmbeddingContext;
type LlamaModel = import("node-llama-cpp").LlamaModel;

export class EmbeddingService {
  private model: LlamaModel | null = null;
  private ctx: LlamaEmbeddingContext | null = null;
  private ready = false;
  private embedCallCount = 0;

  constructor(private readonly config: EmbeddingConfig) {}

  async initialize(): Promise<void> {
    if (this.ready) return;
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
        if (level === LlamaLogLevel.error) console.error(`[llama] ${message}`);
        else if (level === LlamaLogLevel.warn) console.warn(`[llama] ${message}`);
      },
    });
    this.model = await llama.loadModel({ modelPath: this.config.modelPath });
    this.ctx = await this.model.createEmbeddingContext();
    this.ready = true;
  }

  async embed(text: string): Promise<number[]> {
    if (!this.ready || !this.ctx) throw new Error("Embeddings not initialized");
    this.embedCallCount++;
    const result = await this.ctx.getEmbeddingFor(text);
    return Array.from(result.vector);
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    const results: number[][] = [];
    for (const text of texts) {
      results.push(await this.embed(text));
    }
    return results;
  }

  isAvailable(): boolean {
    return this.ready;
  }

  drainEmbedCallCount(): number {
    const count = this.embedCallCount;
    this.embedCallCount = 0;
    return count;
  }

  getEmbedCallCount(): number {
    return this.embedCallCount;
  }

  async dispose(): Promise<void> {
    try {
      await this.ctx?.dispose();
      await this.model?.dispose();
      this.ready = false;
    } catch (e) {
      swallow("embeddings:dispose", e);
    }
  }
}
