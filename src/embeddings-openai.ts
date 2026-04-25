import type { EmbeddingService } from "./embeddings.js";
import type { EmbeddingConfig } from "./config.js";
import { swallow } from "./errors.js";
import { log } from "./log.js";

/**
 * OpenAI-compatible embedding service. Speaks the /v1/embeddings shape that
 * OpenAI, Azure OpenAI, Together, Anyscale, vLLM, LM Studio, Ollama (compat
 * endpoint), DeepInfra, and others all conform to. Switching between any of
 * them is a baseURL change.
 *
 * The vectors this service produces are NOT in the same space as a
 * different provider's vectors, even at the same dimensionality. The
 * providerId field is what the rest of the system uses to keep them apart.
 */
export class OpenAICompatEmbeddingService implements EmbeddingService {
  readonly providerId: string;
  readonly dimensions: number;

  private readonly model: string;
  private readonly baseURL: string;
  private readonly apiKey: string | null;
  private ready = false;

  /** Per-batch limit. OpenAI accepts up to 2048 inputs; most compat servers are stricter. */
  private readonly maxBatchSize = 96;

  constructor(config: EmbeddingConfig) {
    this.model = config.openaiCompat.model;
    this.baseURL = config.openaiCompat.baseURL.replace(/\/+$/, "");
    this.dimensions = config.dimensions;
    // Resolve the API key from the named env var. Empty string is treated as
    // missing — handled at initialize() time so the error is clear and early.
    const keyName = config.openaiCompat.apiKeyEnv;
    const keyVal = process.env[keyName];
    this.apiKey = keyVal && keyVal.length > 0 ? keyVal : null;

    // providerId encodes (provider, model, dim) so vectors written today can
    // be distinguished from the same model at a different output dim later.
    this.providerId = `openai-compat-${this.model}-${this.dimensions}d`;
  }

  async initialize(): Promise<boolean> {
    if (this.ready) return false;
    if (!this.apiKey) {
      throw new Error(
        `OpenAI-compatible embeddings: API key not set. Configure embedding.openaiCompat.apiKeyEnv (default OPENAI_API_KEY) and put the key in that env var.`,
      );
    }
    // Sanity: require dimensions to be set. The OpenAI text-embedding-3-*
    // models support a `dimensions` parameter; non-OpenAI compat servers
    // generally ignore it and return their native dim. We verify on the
    // first embed() call rather than here so we don't burn a request just
    // to validate config.
    if (!Number.isFinite(this.dimensions) || this.dimensions <= 0) {
      throw new Error(
        `OpenAI-compatible embeddings: invalid dimensions ${this.dimensions}`,
      );
    }
    this.ready = true;
    return true;
  }

  async embed(text: string): Promise<number[]> {
    const result = await this.request([text]);
    return result[0];
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    if (texts.length <= this.maxBatchSize) return this.request(texts);
    // Split into chunks so we never exceed the per-request limit.
    const out: number[][] = [];
    for (let i = 0; i < texts.length; i += this.maxBatchSize) {
      const chunk = texts.slice(i, i + this.maxBatchSize);
      const vecs = await this.request(chunk);
      out.push(...vecs);
    }
    return out;
  }

  isAvailable(): boolean {
    return this.ready;
  }

  async dispose(): Promise<void> {
    this.ready = false;
  }

  /**
   * POST one batch to /embeddings with retry-and-backoff on 429.
   * 401/403 fail hard (config problem, retry will not help).
   */
  private async request(input: string[]): Promise<number[][]> {
    if (!this.ready) throw new Error("OpenAI-compat embeddings not initialized");
    const url = `${this.baseURL}/embeddings`;
    const body = {
      model: this.model,
      input,
      // text-embedding-3-* honors `dimensions`. Compat servers that ignore
      // it will return their native dim — we verify after the fact.
      dimensions: this.dimensions,
      encoding_format: "float",
    };

    const maxAttempts = 4;
    let attempt = 0;
    let lastErr: unknown = null;
    while (attempt < maxAttempts) {
      attempt++;
      let res: Response;
      try {
        res = await fetch(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${this.apiKey}`,
          },
          body: JSON.stringify(body),
        });
      } catch (e) {
        // Network-level failure — retry with backoff.
        lastErr = e;
        await this.sleep(backoffMs(attempt));
        continue;
      }

      if (res.ok) {
        const json = await res.json() as {
          data?: Array<{ embedding: number[]; index: number }>;
        };
        const data = json.data ?? [];
        // Sort by index — most servers return in order but the spec only
        // guarantees the index field, so we honor it.
        data.sort((a, b) => a.index - b.index);
        const vecs = data.map(d => d.embedding);
        if (vecs.length !== input.length) {
          throw new Error(
            `OpenAI-compat embeddings: returned ${vecs.length} vectors for ${input.length} inputs`,
          );
        }
        // Verify dim once per response so a misconfigured server fails
        // loudly instead of writing wrong-sized vectors into the DB.
        if (vecs[0].length !== this.dimensions) {
          throw new Error(
            `OpenAI-compat embeddings: server returned ${vecs[0].length}-dim vectors but config requested ${this.dimensions}. ` +
              `For non-OpenAI providers that ignore the 'dimensions' parameter, set embedding.dimensions in plugin config to match the server's native output.`,
          );
        }
        return vecs;
      }

      // Hard fail on auth / not found — retrying will not help.
      if (res.status === 401 || res.status === 403) {
        const text = await readBodyText(res);
        throw new Error(
          `OpenAI-compat embeddings: auth failed (${res.status}). Check the API key in env var. Response: ${text.slice(0, 200)}`,
        );
      }
      if (res.status === 404) {
        const text = await readBodyText(res);
        throw new Error(
          `OpenAI-compat embeddings: endpoint not found at ${url}. Check baseURL. Response: ${text.slice(0, 200)}`,
        );
      }

      // 429 (rate limit) and 5xx — retry with backoff. Honor Retry-After
      // when present. Note: OpenAI returns HTTP 429 for both transient
      // rate limits and "out of credits" (insufficient_quota) — the
      // latter is not retryable, so peek at the body and fail fast.
      if (res.status === 429 || res.status >= 500) {
        const text = await readBodyText(res);
        if (res.status === 429 && /insufficient_quota/i.test(text)) {
          throw new Error(
            `OpenAI-compat embeddings: insufficient quota on this API key. ` +
              `Add credits / a payment method at the provider's billing page, or switch keys. ` +
              `Response: ${text.slice(0, 200)}`,
          );
        }
        const retryAfter = parseRetryAfter(res.headers.get("retry-after"));
        const wait = retryAfter ?? backoffMs(attempt);
        log.warn(`[embeddings:openai] ${res.status} from ${url}, retrying in ${wait}ms (attempt ${attempt}/${maxAttempts})`);
        lastErr = new Error(`HTTP ${res.status}: ${text.slice(0, 200)}`);
        await this.sleep(wait);
        continue;
      }

      // Other 4xx — body usually has the reason. Don't retry.
      const text = await readBodyText(res);
      throw new Error(
        `OpenAI-compat embeddings: HTTP ${res.status}. Response: ${text.slice(0, 300)}`,
      );
    }

    throw new Error(
      `OpenAI-compat embeddings: exhausted ${maxAttempts} attempts. Last error: ${String(lastErr)}`,
    );
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(r => setTimeout(r, ms));
  }
}

/** Exponential backoff with jitter. 1s, 2s, 4s, 8s base, +/- 25%. */
function backoffMs(attempt: number): number {
  const base = 1000 * Math.pow(2, attempt - 1);
  const jitter = 1 + (Math.random() * 0.5 - 0.25);
  return Math.round(base * jitter);
}

/** Parse Retry-After header (seconds or HTTP-date) into ms; null if absent or unparseable. */
function parseRetryAfter(value: string | null): number | null {
  if (!value) return null;
  const asInt = parseInt(value, 10);
  if (Number.isFinite(asInt)) return asInt * 1000;
  const asDate = Date.parse(value);
  if (Number.isFinite(asDate)) {
    const ms = asDate - Date.now();
    return ms > 0 ? ms : 0;
  }
  return null;
}

async function readBodyText(res: Response): Promise<string> {
  try {
    return await res.text();
  } catch (e) {
    swallow("embeddings:openai:readBody", e);
    return "";
  }
}
