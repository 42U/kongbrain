# OpenClaw Upstream Proposals

Changes made to OpenClaw core that should be proposed upstream as separate PRs
if/when contributing back to the main OpenClaw project.

## PR 1: thinkingLevel per-turn override

**Files:** `src/plugins/types.ts`, `src/agents/pi-embedded-runner/run/attempt.ts`

Added `thinkingLevel?: "none" | "low" | "medium" | "high"` to
`PluginHookBeforePromptBuildResult`. Context-engine plugins that classify intent
(e.g. simple question vs complex reasoning) need to adjust thinking level
per-turn. Currently thinkingLevel is set once at session creation with no
per-turn override path.

Also added `PluginPromptBuildNonMutationFields` type to exclude behavioral
overrides from the prompt-mutation exhaustiveness assertion.

## PR 2: assistantTextLengthSoFar on before_tool_call

**Files:** `src/plugins/types.ts`, `src/agents/pi-tools.before-tool-call.ts`,
`src/agents/pi-tools.ts`, `src/agents/pi-embedded-subscribe.ts`

Added `assistantTextLengthSoFar?: number` to `PluginHookBeforeToolCallEvent`.
Context-engine plugins implementing a "planning gate" (block first tool call if
no reasoning text was emitted) need to know how much text the assistant has
output so far in the current turn. Without this, plugins must track streaming
text themselves via `llm_output` hooks — fragile and duplicative.

Includes TurnState registry infrastructure (module-level Map keyed by runId)
for sharing mutable per-turn state between the streaming subscriber and tool
call hooks.

## PR 3: toolCallIndexInTurn on before_tool_call

**Files:** `src/plugins/types.ts`, `src/agents/pi-tools.before-tool-call.ts`

Added `toolCallIndexInTurn?: number` to `PluginHookBeforeToolCallEvent`.
Plugins enforcing tool-call limits per turn (e.g. max 15 tool calls before
forcing a pause) need to know the zero-based index of the current tool call
within the turn. Without this, plugins must maintain their own counter via
repeated `before_tool_call` invocations.

## PR 4: runtime.complete() — low-level LLM completion for plugins

**Files:** `src/plugins/runtime/types.ts`, `src/plugins/runtime/types-core.ts`,
implementation in runtime provider.

Context-engine plugins need to make structured LLM calls (multi-message prompts,
temperature control, max_tokens, JSON output) for background tasks like memory
extraction, skill graduation, cognitive checks, and soul generation. The existing
`runtime.subagent.run()` API only accepts a single message string — it cannot
express multi-turn structured prompts with system instructions, and lacks
temperature/maxTokens control.

Without `runtime.complete()`, plugins must import provider SDKs directly
(e.g. `@mariozechner/pi-ai`), hardcoding provider+model and bypassing OpenClaw's
auth resolution, model configuration, and provider abstraction entirely. This
breaks multi-provider support — the core value proposition of OpenClaw.
