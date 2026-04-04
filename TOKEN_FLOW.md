# Token Flow Trace

This document maps the complete lifecycle of tokens through the Claw CLI,
from user input to API consumption. It identifies every injection point,
the growth characteristics of each category, and the key findings that
inform optimization work.

## Complete Token Lifecycle

```
User input
  |
  v
[1] ConversationRuntime::run_turn(user_input)
    @ rust/crates/runtime/src/conversation.rs:290-294
    - ConversationMessage::user_text() pushed to session.messages
    - TOKEN IN: user text stored verbatim (no truncation)
  |
  v
[2] API Request Assembly (inner loop start)
    @ conversation.rs:312-315
    - ApiRequest {
        system_prompt: self.system_prompt.clone(),   // Vec<String> cloned every call
        messages: self.session.messages.clone(),      // FULL deep clone of all messages
      }
    - TOKEN IN: entire system prompt + full message history
    - PERF NOTE: O(n) deep clone on every API call
  |
  v
[3] Bridge to MessageRequest
    @ main.rs (AnthropicRuntimeClient::stream)
    - system: request.system_prompt.join("\n\n")      // sections → single string
    - messages: convert_messages(&request.messages)    // ConversationMessage → InputMessage
    - tools: filter_tool_specs()                       // ToolDefinition array (JSON schemas)
    - TOKEN IN: tool definitions (~5-15K chars of JSON schema)
  |
  v
[4] System Prompt (assembled once per session, sent every call)
    @ rust/crates/runtime/src/prompt.rs:134-156

    STATIC sections (before SYSTEM_PROMPT_DYNAMIC_BOUNDARY):
    ┌──────────────────────────────────┬────────────┐
    │ Section                          │ ~Chars     │
    ├──────────────────────────────────┼────────────┤
    │ Intro                            │ ~400       │
    │ System guidelines                │ ~600       │
    │ Doing tasks guidelines           │ ~600       │
    │ Actions section                  │ ~300       │
    │ DYNAMIC_BOUNDARY marker          │ 37         │
    └──────────────────────────────────┴────────────┘

    DYNAMIC sections (after DYNAMIC_BOUNDARY):
    ┌──────────────────────────────────┬────────────┐
    │ Section                          │ ~Chars     │
    ├──────────────────────────────────┼────────────┤
    │ Environment context              │ ~150       │
    │ Project context (date, cwd)      │ ~100       │
    │ Git status snapshot              │ variable   │
    │ Git diff snapshot                │ UNBOUNDED  │
    │ Instruction files                │ ≤12,000    │
    │ Runtime config                   │ variable   │
    └──────────────────────────────────┴────────────┘
  |
  v
[5] HTTP POST to Anthropic
    @ rust/crates/api/src/providers/anthropic.rs:336-354
    - Full system prompt + full message history + full tool definitions sent
    - Anthropic prompt caching may cache the prefix (5-min TTL)
    - API returns Usage { input_tokens, output_tokens,
      cache_creation_input_tokens, cache_read_input_tokens }
  |
  v
[6] Streaming Response Processing
    @ conversation.rs:316-330
    - build_assistant_message() collects:
      TextDelta, ToolUse, Usage, PromptCache events
    - AssistantEvent::Usage carries the actual API-reported token counts
  |
  v
[7] Usage Recording
    @ conversation.rs:331-333
    - usage_tracker.record(usage) → cumulative counters updated
    - TOKEN OBSERVATION POINT: where we learn actual consumption
  |
  v
[8] Assistant Message Storage
    @ conversation.rs:351-354
    - ConversationMessage with text blocks + tool_use blocks stored in session
    - TOKEN IN: assistant text + tool_use blocks (id, name, input JSON)
  |
  v
[9] Tool Execution (if tool_uses present)
    @ conversation.rs:360-458
    For each pending tool use:
    a. Pre-tool hook (may modify input)
    b. Permission check (may deny)
    c. tool_executor.execute(name, input) → output String
    d. Post-tool hook (may append feedback)
    e. ConversationMessage::tool_result(id, name, output, is_error)
    - TOKEN IN: tool output stored VERBATIM — NO TRUNCATION
    - This is the #1 source of context bloat
  |
  v
[10] Loop back to step [2] if tool_uses were present
    - Each iteration re-sends ALL accumulated messages
    - Context grows monotonically within a single turn
  |
  v
[11] Auto-Compaction Check
    @ conversation.rs:462, 507-530
    - Triggers when cumulative input_tokens >= threshold (default 100K)
    - BUG: uses cumulative (lifetime) tokens, not current context size
    - compact_session() preserves last 4 messages, summarizes the rest
    - TOKEN REDUCTION: the only meaningful reduction mechanism in the system
  |
  v
[12] TurnSummary returned to caller
    - Contains: assistant_messages, tool_results, prompt_cache_events,
      iterations, cumulative usage, auto_compaction event (if any)
```

## Token Contribution Breakdown

| Category | Per-Call Cost | Growth Pattern | Bounded? |
|---|---|---|---|
| System prompt (static sections) | ~2,000 chars / ~500 tokens | Constant per session | Yes |
| System prompt (environment) | ~150 chars | Constant per session | Yes |
| Git diff in system prompt | 0 – 50K+ chars | Changes rarely within session | **No** |
| Instruction files (CLAUDE.md etc) | 0 – 12,000 chars | Constant per session | Yes (budgeted) |
| Runtime config in system prompt | ~200 chars | Constant per session | Yes |
| Tool definitions | ~5,000 – 15,000 chars | Constant per session | Yes (fixed schema) |
| User messages | Variable | Linear with turns | No (until compaction) |
| Assistant text responses | Variable | Linear with turns | No (until compaction) |
| Tool use blocks (id+name+input) | ~100 – 500 chars each | Linear with tool calls | No (until compaction) |
| **Tool result outputs** | **1K – 100K+ chars each** | **Linear with tool calls** | **No (verbatim, unbounded)** |

## Static vs Dynamic vs Growing Context

### Static (per-session, never changes)
- System prompt static sections (intro, system, tasks, actions)
- Tool definitions
- Model, max_tokens configuration

### Dynamic (per-session, changes rarely)
- Environment context (cwd, date, platform)
- Git status/diff snapshot (captured once at session start)
- Instruction files (loaded once at session start)

### Growing (per-turn, monotonically increasing)
- User messages
- Assistant responses (text + tool_use blocks)
- Tool result outputs ← **dominant growth factor**

## Key Findings

### 1. Tool results are the #1 source of context bloat
Tool outputs are stored verbatim in `ContentBlock::ToolResult { output: String }`
(session.rs:36-40). A single `read_file` call can inject 50K+ characters.
A `grep_search` with many matches can inject thousands of lines. These persist
in the session and are re-sent on every subsequent API call until compaction.

### 2. Messages are deep-cloned on every API call
`self.session.messages.clone()` at conversation.rs:314 creates a full deep copy
of all message content on every API call. For sessions with large tool results,
this is significant memory allocation churn (O(n*m) where n=messages, m=avg size).

### 3. The len/4+1 token estimation heuristic systematically undercounts
The heuristic in compact.rs:392-404 uses `text.len() / 4 + 1` per content block.
This ignores:
- JSON serialization overhead (role markers, type tags, key names)
- Tool metadata fields (tool_use_id, is_error flag)
- System prompt tokens (not counted by estimate_session_tokens at all)
- The conversion overhead from ConversationMessage → InputMessage → JSON

### 4. Auto-compaction trigger uses cumulative instead of current tokens
`maybe_auto_compact()` at conversation.rs:508 checks
`self.usage_tracker.cumulative_usage().input_tokens` against the threshold.
Cumulative never decreases — after compaction, every subsequent turn
immediately exceeds the threshold, causing unnecessary re-compaction.

### 5. Git diff in the system prompt is unbounded
`read_git_diff()` in prompt.rs:245-263 captures the full staged + unstaged diff
with no size limit. A developer with many uncommitted changes can have 50K+
characters injected into the system prompt, sent on every single API call.