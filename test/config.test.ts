import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { homedir } from "node:os";
import { join } from "node:path";
import { parsePluginConfig } from "../src/config.js";

describe("parsePluginConfig", () => {
  const originalEnv = { ...process.env };

  beforeEach(() => {
    // Clear all KongBrain-related env vars
    delete process.env.SURREAL_URL;
    delete process.env.SURREAL_HTTP_URL;
    delete process.env.SURREAL_USER;
    delete process.env.SURREAL_PASS;
    delete process.env.SURREAL_NS;
    delete process.env.SURREAL_DB;
    delete process.env.EMBED_MODEL_PATH;
  });

  afterEach(() => {
    Object.assign(process.env, originalEnv);
  });

  it("returns sensible defaults with no input", () => {
    const config = parsePluginConfig();
    expect(config.surreal.url).toBe("ws://localhost:8042/rpc");
    expect(config.surreal.user).toBe("root");
    expect(config.surreal.pass).toBe("root");
    expect(config.surreal.ns).toBe("kong");
    expect(config.surreal.db).toBe("memory");
    expect(config.embedding.dimensions).toBe(1024);
    expect(config.embedding.modelPath).toBe(
      join(homedir(), ".node-llama-cpp", "models", "bge-m3-q4_k_m.gguf"),
    );
  });

  it("returns defaults with empty object", () => {
    const config = parsePluginConfig({});
    expect(config.surreal.url).toBe("ws://localhost:8042/rpc");
    expect(config.surreal.ns).toBe("kong");
  });

  it("reads values from plugin config", () => {
    const config = parsePluginConfig({
      surreal: {
        url: "ws://db.example.com:9000/rpc",
        user: "admin",
        pass: "secret",
        ns: "prod",
        db: "brain",
      },
      embedding: {
        modelPath: "/custom/model.gguf",
        dimensions: 768,
      },
    });
    expect(config.surreal.url).toBe("ws://db.example.com:9000/rpc");
    expect(config.surreal.user).toBe("admin");
    expect(config.surreal.pass).toBe("secret");
    expect(config.surreal.ns).toBe("prod");
    expect(config.surreal.db).toBe("brain");
    expect(config.embedding.modelPath).toBe("/custom/model.gguf");
    expect(config.embedding.dimensions).toBe(768);
  });

  it("env vars override plugin config", () => {
    process.env.SURREAL_URL = "ws://env-override:1234/rpc";
    process.env.SURREAL_USER = "envuser";
    process.env.SURREAL_PASS = "envpass";
    process.env.SURREAL_NS = "envns";
    process.env.SURREAL_DB = "envdb";
    process.env.EMBED_MODEL_PATH = "/env/model.gguf";

    const config = parsePluginConfig({
      surreal: { url: "ws://should-be-overridden:8042/rpc", user: "ignored" },
    });

    expect(config.surreal.url).toBe("ws://env-override:1234/rpc");
    expect(config.surreal.user).toBe("envuser");
    expect(config.surreal.pass).toBe("envpass");
    expect(config.surreal.ns).toBe("envns");
    expect(config.surreal.db).toBe("envdb");
    expect(config.embedding.modelPath).toBe("/env/model.gguf");
  });

  it("derives httpUrl from ws url", () => {
    const config = parsePluginConfig({
      surreal: { url: "ws://localhost:8042/rpc" },
    });
    expect(config.surreal.httpUrl).toBe("http://localhost:8042/sql");
  });

  it("derives httpUrl from wss url", () => {
    const config = parsePluginConfig({
      surreal: { url: "wss://secure.db.com:443/rpc" },
    });
    expect(config.surreal.httpUrl).toBe("https://secure.db.com:443/sql");
  });

  it("SURREAL_HTTP_URL overrides derived httpUrl", () => {
    process.env.SURREAL_HTTP_URL = "http://custom:9999/sql";
    const config = parsePluginConfig({
      surreal: { url: "ws://localhost:8042/rpc" },
    });
    expect(config.surreal.httpUrl).toBe("http://custom:9999/sql");
  });

  it("ignores non-string config values and uses defaults", () => {
    const config = parsePluginConfig({
      surreal: { url: 12345, user: null, pass: undefined },
    });
    expect(config.surreal.url).toBe("ws://localhost:8042/rpc");
    expect(config.surreal.user).toBe("root");
    expect(config.surreal.pass).toBe("root");
  });
});
