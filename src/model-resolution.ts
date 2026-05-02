export interface ResolveModelRefInput {
  explicitProvider?: unknown;
  explicitModel?: unknown;
  config?: unknown;
  runtimeDefaults?: unknown;
}

export interface ResolvedModelRef {
  provider: string;
  modelId: string;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null
    ? value as Record<string, unknown>
    : null;
}

function nonEmptyString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim().length > 0
    ? value.trim()
    : undefined;
}

function unwrapModelRef(value: unknown): string | undefined {
  const direct = nonEmptyString(value);
  if (direct) return direct;

  const record = asRecord(value);
  if (!record) return undefined;

  return unwrapModelRef(record.primary) ?? unwrapModelRef(record.id);
}

function defaultModelFromConfig(config: unknown): string | undefined {
  const cfg = asRecord(config);
  const agents = asRecord(cfg?.agents);
  const defaults = asRecord(agents?.defaults);
  return unwrapModelRef(defaults?.model);
}

function defaultModelFromRuntime(runtimeDefaults: unknown): string | undefined {
  const defaults = asRecord(runtimeDefaults);
  return unwrapModelRef(defaults?.model);
}

function providerFromRuntime(runtimeDefaults: unknown): string | undefined {
  const defaults = asRecord(runtimeDefaults);
  return nonEmptyString(defaults?.provider);
}

function splitQualifiedModelRef(modelRef: string): { provider: string; modelId: string } | null {
  const slash = modelRef.indexOf("/");
  if (slash <= 0 || slash === modelRef.length - 1) return null;
  return {
    provider: modelRef.slice(0, slash),
    modelId: modelRef.slice(slash + 1),
  };
}

function chooseModelRef(input: ResolveModelRefInput): { modelRef: string; explicit: boolean } | null {
  const explicit = unwrapModelRef(input.explicitModel);
  if (explicit) return { modelRef: explicit, explicit: true };

  const configDefault = defaultModelFromConfig(input.config);
  if (configDefault) return { modelRef: configDefault, explicit: false };

  const runtimeDefault = defaultModelFromRuntime(input.runtimeDefaults);
  if (runtimeDefault) return { modelRef: runtimeDefault, explicit: false };

  return null;
}

export function resolveModelRef(input: ResolveModelRefInput): ResolvedModelRef {
  const chosen = chooseModelRef(input);

  if (!chosen) {
    throw new Error("No LLM model configured for KongBrain internal completion");
  }

  const explicitProvider = nonEmptyString(input.explicitProvider);
  if (chosen.explicit && explicitProvider) {
    return { provider: explicitProvider, modelId: chosen.modelRef };
  }

  const qualified = splitQualifiedModelRef(chosen.modelRef);
  if (qualified) return qualified;

  const provider =
    explicitProvider ??
    providerFromRuntime(input.runtimeDefaults);

  if (!provider) {
    throw new Error(`No LLM provider configured for model "${chosen.modelRef}"`);
  }

  return { provider, modelId: chosen.modelRef };
}
