const LEVELS = { error: 0, warn: 1, info: 2, debug: 3 } as const;
type Level = keyof typeof LEVELS;

const currentLevel: Level = (process.env.KONGBRAIN_LOG_LEVEL as Level) ?? "warn";

export const log = {
  error: (...args: unknown[]) => { if (LEVELS[currentLevel] >= LEVELS.error) console.error("[kongbrain]", ...args); },
  warn: (...args: unknown[]) => { if (LEVELS[currentLevel] >= LEVELS.warn) console.warn("[kongbrain]", ...args); },
  info: (...args: unknown[]) => { if (LEVELS[currentLevel] >= LEVELS.info) console.info("[kongbrain]", ...args); },
  debug: (...args: unknown[]) => { if (LEVELS[currentLevel] >= LEVELS.debug) console.debug("[kongbrain]", ...args); },
};
