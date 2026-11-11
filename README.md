# token-stream-assembler

[![npm version](https://img.shields.io/npm/v/@trysac/token-stream-assembler.svg)](https://www.npmjs.com/package/@trysac/token-stream-assembler)
[![license: MIT](https://img.shields.io/npm/l/@trysac/token-stream-assembler.svg)](LICENSE)
[![CI](https://github.com/Trysac/u203-token-stream-assembler/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u203-token-stream-assembler/actions/workflows/ci.yml)

Assemble streamed LLM deltas from OpenAI chat chunks or Anthropic message events into one complete message with text, parsed tool call arguments and stop reason

## Problem

Streaming LLM responses arrive as many small deltas: text pieces, tool call names and argument JSON split
across chunks, and a stop reason at the end. Every application re-implements the accumulation and gets the
edge cases wrong: argument fragments keyed by index, tool calls without arguments, events after the end or a
stream that never finishes. This library turns already-parsed stream events into one validated message, with
snapshots while streaming, and requires no provider SDK or runtime dependency.

## Install

```sh
npm install @trysac/token-stream-assembler
```

Requires Node.js 20 or newer. The package ships ES modules with TypeScript declarations.

## Usage

```ts
import { AssembleError, assemble, createAssembler } from "@trysac/token-stream-assembler";

const assembler = createAssembler("openai");

// Feed already-parsed events (the SSE / HTTP layer is out of scope).
assembler.push({ choices: [{ index: 0, delta: { role: "assistant", content: "Hel" } }] });
assembler.push({ choices: [{ index: 0, delta: { content: "lo" } }] });
assembler.push({
  choices: [
    {
      index: 0,
      delta: {
        tool_calls: [
          { index: 0, id: "call_1", function: { name: "get_weather", arguments: '{"city":' } },
        ],
      },
    },
  ],
});
assembler.push({
  choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '"Lima"}' }] } }],
});
assembler.push({ choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] });

const snapshot = assembler.snapshot(); // text so far, toolCalls[].arguments === null
const message = assembler.finish();
message.text; // "Hello"
message.toolCalls[0]?.arguments; // { city: "Lima" }
message.stopReason; // "tool_calls"

// Or assemble a whole batch of events at once:
const anthropicMessage = assemble("anthropic", [
  { type: "message_start" },
  { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
  { type: "text_delta", index: 0, text: "Searching" },
  { type: "message_delta", delta: { stop_reason: "end_turn" } },
  { type: "message_stop" },
]);

try {
  assembler.finish();
} catch (error) {
  if (error instanceof AssembleError) {
    console.error(error.code); // e.g. "invalid_tool_arguments"
  }
}
```

## API

| Export | Signature | Description |
|---|---|---|
| `Dialect` | `type Dialect = "openai" \| "anthropic"` | The wire dialect of the pushed events. |
| `createAssembler` | `(dialect: Dialect, opts?: AssemblerOptions) => Assembler` | Creates a stateful assembler. Limits: `maxTextChars` (default 1 000 000), `maxToolCalls` (default 128), `maxArgumentChars` per call (default 1 000 000). |
| `Assembler` | `{ push(event: unknown): void; snapshot(): AssembledMessage; finish(): AssembledMessage }` | `push` feeds one parsed event; `snapshot` returns the state so far with `arguments: null`; `finish` requires the terminal event, parses tool arguments and returns `done: true` (repeat calls return an equal value). |
| `assemble` | `(dialect: Dialect, events: Iterable<unknown>) => AssembledMessage` | Pushes every event then calls `finish()`. |
| `AssembledMessage` | `{ role: "assistant"; text: string; toolCalls: ToolCall[]; stopReason: string \| null; done: boolean }` | The assembled message. |
| `ToolCall` | `{ id: string; name: string; arguments: unknown; rawArguments: string }` | One tool call; `arguments` is the parsed JSON (`{}` for an empty argument string) or `null` in snapshots. |
| `AssemblerOptions` | `{ maxTextChars?: number; maxToolCalls?: number; maxArgumentChars?: number }` | Optional limits passed to `createAssembler`. |
| `AssembleError` | `class AssembleError extends Error { readonly code: AssembleErrorCode }` | Thrown on invalid input or limits. Codes: `invalid_event`, `unsupported_choice`, `unknown_block`, `invalid_tool_arguments`, `incomplete`, `after_finish`, `stream_error`, `too_large`. After any error the assembler is poisoned and keeps throwing the same code. |

## Development

```sh
npm install
npm run check   # lint, format, types, tests with coverage, docs, secrets
```

See [AGENTS.md](AGENTS.md) for the full command list and [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow.

## License

[MIT](LICENSE) © 2026 Miguel Gomez
