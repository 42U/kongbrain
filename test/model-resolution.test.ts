import { describe, expect, it } from "vitest";
import { resolveModelRef } from "../src/model-resolution.js";

describe("resolveModelRef", () => {
  it("resolves OpenClaw config primary OpenRouter refs by the first slash", () => {
    const resolved = resolveModelRef({
      config: {
        agents: {
          defaults: {
            model: { primary: "openrouter/google/gemini-3-flash-preview" },
          },
        },
      },
      runtimeDefaults: { provider: "anthropic", model: "claude-opus-4.6" },
    });

    expect(resolved).toEqual({
      provider: "openrouter",
      modelId: "google/gemini-3-flash-preview",
    });
  });

  it("does not let a stale runtime provider override a qualified runtime model", () => {
    const resolved = resolveModelRef({
      runtimeDefaults: {
        provider: "anthropic",
        model: { primary: "openrouter/google/gemini-3-flash-preview" },
      },
    });

    expect(resolved).toEqual({
      provider: "openrouter",
      modelId: "google/gemini-3-flash-preview",
    });
  });

  it("uses explicit unqualified provider and model", () => {
    const resolved = resolveModelRef({
      explicitProvider: "openrouter",
      explicitModel: "google/gemini-3-flash-preview",
      runtimeDefaults: { provider: "anthropic", model: "claude-opus-4.6" },
    });

    expect(resolved).toEqual({
      provider: "openrouter",
      modelId: "google/gemini-3-flash-preview",
    });
  });

  it("treats an explicit qualified model as authoritative", () => {
    const resolved = resolveModelRef({
      explicitModel: "openrouter/google/gemini-3-flash-preview",
      runtimeDefaults: { provider: "anthropic", model: "claude-opus-4.6" },
    });

    expect(resolved).toEqual({
      provider: "openrouter",
      modelId: "google/gemini-3-flash-preview",
    });
  });

  it("falls back to legacy provider plus unqualified model", () => {
    const resolved = resolveModelRef({
      runtimeDefaults: {
        provider: "anthropic",
        model: "claude-opus-4.6",
      },
    });

    expect(resolved).toEqual({
      provider: "anthropic",
      modelId: "claude-opus-4.6",
    });
  });
});
