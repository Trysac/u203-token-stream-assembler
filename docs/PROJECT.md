# Project record: token-stream-assembler

## Objective

Assemble streamed LLM deltas from OpenAI chat chunks or Anthropic message events into one complete message with text, parsed tool call arguments and stop reason

State the measurable outcome this utility must deliver and who it is for.

## Scope

In scope:

- The public API documented in the README.

Out of scope:

- List what this utility deliberately does not do.

## Timeline

| Date | Milestone |
|---|---|
| 2026 | Repository created from the factory utility template. |

## Effort metrics

Measured values only. Leave "pending measurement" until a real figure is recorded; never estimate here.

| Metric | Value | Source |
|---|---|---|
| Tokens (input / output) | 750694 / 68252 | factory ledger |
| Hours (human) | pending measurement | time log |
| Cost (USD) | 0.0793 | factory ledger |

## Factory run u203 (measured from API usage)

| Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|
| generation | deepseek-v4-pro | yes | 1545 | 17907 | 0 | 3584 | 0.0366 |
| fix-1 | deepseek-flash | no | 16634 | 9655 | 0 | 153344 | 0.0087 |
| audit | deepseek-flash | no | 20865 | 19232 | 0 | 346368 | 0.0157 |
| fix-2 | deepseek-flash | no | 14044 | 5458 | 0 | 175488 | 0.0059 |
| docs | deepseek-flash | no | 18566 | 16000 | 0 | 256 | 0.0124 |

Total: $0.0793. Prices: deepseek line, as measured by its adapter.
