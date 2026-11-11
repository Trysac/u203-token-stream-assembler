# Use provider-specific state machine for streamed LLM delta assembly

Status: accepted (generated, pending human review)

## Context

Streaming LLM responses from OpenAI and Anthropic arrive as many small deltas that must be accumulated into a single message. The wire formats differ, and applications repeatedly implement accumulation incorrectly (argument fragments, block indexing, stop reasons).

## Decision

Implement a small, dependency-free TypeScript assembler with a dedicated state machine per dialect. The assembler accepts already-parsed event objects, accumulates text and tool call fragments, enforces size limits, provides snapshots, and parses tool arguments on `finish()`. Errors are typed with stable codes and poison the instance after any throw.

## Consequences

Positive: single reusable utility for both dialects; no runtime dependencies; explicit validation and limits. Negative: scope excludes raw SSE/HTTP parsing and non-first choices; maintainers must extend state machines for future dialects.
