import { describe, expect, it } from "vitest";
import { AssembleError, assemble, createAssembler } from "../src/index.js";

describe("OpenAI text assembly", () => {
  it("snapshot after the first chunk reports partial text and done false", () => {
    const a = createAssembler("openai");
    a.push({ choices: [{ index: 0, delta: { role: "assistant", content: "Hel" } }] });
    expect(a.snapshot()).toEqual({
      role: "assistant",
      text: "Hel",
      toolCalls: [],
      stopReason: null,
      done: false,
    });
    a.push({ choices: [{ index: 0, delta: { content: "lo" } }] });
    a.push({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] });
    expect(a.finish()).toEqual({
      role: "assistant",
      text: "Hello",
      toolCalls: [],
      stopReason: "stop",
      done: true,
    });
  });

  it("assembles text deltas and finish_reason", () => {
    const a = createAssembler("openai");
    a.push({ choices: [{ index: 0, delta: { role: "assistant", content: "Hel" } }] });
    a.push({ choices: [{ index: 0, delta: { content: "lo" } }] });
    a.push({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] });

    expect(a.snapshot()).toEqual({
      role: "assistant",
      text: "Hello",
      toolCalls: [],
      stopReason: "stop",
      done: false,
    });

    const result = a.finish();
    expect(result).toEqual({
      role: "assistant",
      text: "Hello",
      toolCalls: [],
      stopReason: "stop",
      done: true,
    });
    expect(a.finish()).toEqual(result);
    expect(a.snapshot()).toEqual(result);
  });

  it("throws after_finish on push after terminal, and poisons", () => {
    const a = createAssembler("openai");
    a.push({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] });
    expect(() => a.push({ choices: [{ index: 0, delta: { content: "x" } }] })).toThrowError(
      AssembleError,
    );
    try {
      a.push({ choices: [{ index: 0, delta: { content: "x" } }] });
    } catch (e) {
      expect(e).toBeInstanceOf(AssembleError);
      expect((e as AssembleError).code).toBe("after_finish");
    }
    // poisoned: finish also throws after_finish
    expect(() => a.finish()).toThrowError(AssembleError);
    try {
      a.finish();
    } catch (e) {
      expect((e as AssembleError).code).toBe("after_finish");
    }
  });
});

describe("OpenAI tool calls", () => {
  it("assembles tool call fragments keyed by index and parses arguments", () => {
    const events = [
      {
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: "call_1",
                  function: { name: "get_weather", arguments: '{"city":' },
                },
              ],
            },
          },
        ],
      },
      {
        choices: [
          { index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '"Lima"}' } }] } },
        ],
      },
      { choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] },
    ];
    const result = assemble("openai", events);
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]).toEqual({
      id: "call_1",
      name: "get_weather",
      arguments: { city: "Lima" },
      rawArguments: '{"city":"Lima"}',
    });
    expect(result.stopReason).toBe("tool_calls");
  });

  it("handles two parallel tool calls ordered by index", () => {
    const events = [
      {
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                { index: 1, id: "call_b", function: { name: "fn_b", arguments: '{"b":' } },
              ],
            },
          },
        ],
      },
      {
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                { index: 0, id: "call_a", function: { name: "fn_a", arguments: '{"a":' } },
              ],
            },
          },
        ],
      },
      {
        choices: [
          { index: 0, delta: { tool_calls: [{ index: 1, function: { arguments: "1}" } }] } },
        ],
      },
      {
        choices: [
          { index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: "2}" } }] } },
        ],
      },
      { choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] },
    ];
    const result = assemble("openai", events);
    expect(result.toolCalls).toHaveLength(2);
    expect(result.toolCalls[0]?.id).toBe("call_a");
    expect(result.toolCalls[1]?.id).toBe("call_b");
    expect(result.toolCalls[0]?.arguments).toEqual({ a: 2 });
    expect(result.toolCalls[1]?.arguments).toEqual({ b: 1 });
  });

  it("snapshot with partial tool arguments reports null", () => {
    const a = createAssembler("openai");
    a.push({
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [{ index: 0, id: "call_1", function: { name: "f", arguments: '{"x":' } }],
          },
        },
      ],
    });
    const snap = a.snapshot();
    expect(snap.done).toBe(false);
    expect(snap.toolCalls).toHaveLength(1);
    expect(snap.toolCalls[0]?.arguments).toBeNull();
    expect(snap.toolCalls[0]?.rawArguments).toBe('{"x":');
  });

  it("throws invalid_tool_arguments when JSON.parse fails, message contains id but not raw args", () => {
    const a = createAssembler("openai");
    a.push({
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [
              { index: 0, id: "call_1", function: { name: "f", arguments: '{"city":' } },
            ],
          },
        },
      ],
    });
    a.push({ choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] });
    expect(() => a.finish()).toThrowError(AssembleError);
    try {
      a.finish();
    } catch (e) {
      expect(e).toBeInstanceOf(AssembleError);
      const err = e as AssembleError;
      expect(err.code).toBe("invalid_tool_arguments");
      expect(err.message).toContain("call_1");
      expect(err.message).not.toContain('{"city":');
    }
  });

  it("tool call with no arguments parses to {}", () => {
    const a = createAssembler("openai");
    a.push({
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [{ index: 0, id: "call_1", function: { name: "f", arguments: "" } }],
          },
        },
      ],
    });
    a.push({ choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] });
    const result = a.finish();
    expect(result.toolCalls[0]?.arguments).toEqual({});
    expect(result.toolCalls[0]?.rawArguments).toBe("");
  });
});

describe("Anthropic stream handling", () => {
  it("assembles text, tool_use, stop_reason and message_stop", () => {
    const events = [
      { type: "message_start" },
      { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
      { type: "text_delta", index: 0, text: "Searching" },
      {
        type: "content_block_start",
        index: 1,
        content_block: { type: "tool_use", id: "toolu_1", name: "search", input: {} },
      },
      { type: "input_json_delta", index: 1, partial_json: '{"q":' },
      { type: "input_json_delta", index: 1, partial_json: '"bm25"}' },
      { type: "message_delta", delta: { stop_reason: "tool_use" } },
      { type: "message_stop" },
    ];
    const result = assemble("anthropic", events);
    expect(result.text).toBe("Searching");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.arguments).toEqual({ q: "bm25" });
    expect(result.toolCalls[0]?.id).toBe("toolu_1");
    expect(result.toolCalls[0]?.name).toBe("search");
    expect(result.stopReason).toBe("tool_use");
  });

  it("multiple text blocks concatenated in index order", () => {
    const events = [
      { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
      { type: "text_delta", index: 0, text: "a" },
      { type: "content_block_start", index: 1, content_block: { type: "text", text: "" } },
      { type: "text_delta", index: 1, text: "b" },
      { type: "content_block_start", index: 2, content_block: { type: "text", text: "" } },
      { type: "text_delta", index: 2, text: "c" },
      { type: "message_stop" },
    ];
    const result = assemble("anthropic", events);
    expect(result.text).toBe("abc");
  });

  it("emoji split across UTF-16 halves", () => {
    const a = createAssembler("anthropic");
    a.push({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
    a.push({ type: "text_delta", index: 0, text: "\ud83d" });
    a.push({ type: "text_delta", index: 0, text: "\ude00" });
    a.push({ type: "message_stop" });
    const result = a.finish();
    expect(result.text).toBe("😀");
  });

  it("ping is ignored", () => {
    const a = createAssembler("anthropic");
    a.push({ type: "ping" });
    a.push({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
    a.push({ type: "text_delta", index: 0, text: "ok" });
    a.push({ type: "message_stop" });
    expect(a.finish().text).toBe("ok");
  });

  it("throws stream_error with error type only", () => {
    const a = createAssembler("anthropic");
    expect(() => a.push({ type: "error", error: { type: "overloaded_error" } })).toThrowError(
      AssembleError,
    );
    try {
      a.push({ type: "error", error: { type: "overloaded_error" } });
    } catch (e) {
      expect(e).toBeInstanceOf(AssembleError);
      const err = e as AssembleError;
      expect(err.code).toBe("stream_error");
      expect(err.message).toBe("stream error: overloaded_error");
    }
  });

  it("throws unknown_block for delta on never-started block", () => {
    const a = createAssembler("anthropic");
    expect(() => a.push({ type: "text_delta", index: 0, text: "x" })).toThrowError(AssembleError);
    try {
      a.push({ type: "text_delta", index: 0, text: "x" });
    } catch (e) {
      expect((e as AssembleError).code).toBe("unknown_block");
    }
  });

  it("content_block_stop is accepted", () => {
    const a = createAssembler("anthropic");
    a.push({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
    a.push({ type: "text_delta", index: 0, text: "x" });
    a.push({ type: "content_block_stop", index: 0 });
    a.push({ type: "message_stop" });
    expect(a.finish().text).toBe("x");
  });
});

describe("error handling and limits", () => {
  it("throws incomplete when finish called before terminal", () => {
    const a = createAssembler("openai");
    expect(() => a.finish()).toThrowError(AssembleError);
    try {
      a.finish();
    } catch (e) {
      expect((e as AssembleError).code).toBe("incomplete");
    }
  });

  it("poisons after incomplete", () => {
    const a = createAssembler("openai");
    expect(() => a.finish()).toThrow();
    expect(() => a.snapshot()).toThrowError(AssembleError);
    try {
      a.snapshot();
    } catch (e) {
      expect((e as AssembleError).code).toBe("incomplete");
    }
  });

  it("throws too_large for text beyond maxTextChars", () => {
    const a = createAssembler("openai", { maxTextChars: 5 });
    a.push({ choices: [{ index: 0, delta: { content: "Hel" } }] });
    expect(() => a.push({ choices: [{ index: 0, delta: { content: "lo!" } }] })).toThrowError(
      AssembleError,
    );
    try {
      a.push({ choices: [{ index: 0, delta: { content: "lo!" } }] });
    } catch (e) {
      expect((e as AssembleError).code).toBe("too_large");
    }
  });

  it("throws unsupported_choice for choices[0].index !== 0", () => {
    const a = createAssembler("openai");
    expect(() => a.push({ choices: [{ index: 1, delta: {} }] })).toThrowError(AssembleError);
    try {
      a.push({ choices: [{ index: 1, delta: {} }] });
    } catch (e) {
      expect((e as AssembleError).code).toBe("unsupported_choice");
    }
  });

  it("throws invalid_event for non-object push", () => {
    const a = createAssembler("openai");
    expect(() => a.push(null)).toThrowError(AssembleError);
    try {
      a.push(null);
    } catch (e) {
      expect((e as AssembleError).code).toBe("invalid_event");
    }
  });

  it("usage-only OpenAI chunk (empty choices) is ignored", () => {
    const a = createAssembler("openai");
    a.push({ choices: [] });
    a.push({ choices: [{ index: 0, delta: { content: "ok" } }] });
    a.push({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] });
    expect(a.finish().text).toBe("ok");
  });

  it("accepts a usage-only chunk after the terminal chunk", () => {
    const a = createAssembler("openai");
    a.push({ choices: [{ index: 0, delta: { content: "ok" } }] });
    a.push({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] });
    a.push({ choices: [], usage: { total_tokens: 7 } });
    const result = a.finish();
    expect(result.text).toBe("ok");
    expect(result.stopReason).toBe("stop");
    expect(result.done).toBe(true);
  });
});

function errorCode(fn: () => void): string | undefined {
  try {
    fn();
  } catch (e) {
    return (e as AssembleError).code;
  }
  return undefined;
}

describe("validation and limits", () => {
  it("rejects malformed OpenAI chunks", () => {
    const events = [
      {},
      { choices: [null] },
      { choices: [{ index: 0, delta: { tool_calls: [null] } }] },
      { choices: [{ index: 0, delta: { tool_calls: [{ index: -1 }] } }] },
      { choices: [{ index: 0, delta: { tool_calls: [{ index: "0" }] } }] },
    ];
    for (const event of events) {
      const a = createAssembler("openai");
      expect(errorCode(() => a.push(event))).toBe("invalid_event");
    }
  });

  it("accepts missing delta and finish without content", () => {
    const a = createAssembler("openai");
    a.push({ choices: [{ index: 0 }] });
    a.push({ choices: [{ index: 0, delta: null }] });
    a.push({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] });
    const result = a.finish();
    expect(result.text).toBe("");
    expect(result.stopReason).toBe("stop");
  });

  it("accepts an OpenAI tool call without a function object", () => {
    const a = createAssembler("openai");
    a.push({ choices: [{ index: 0, delta: { tool_calls: [{ index: 0 }] } }] });
    a.push({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] });
    const result = a.finish();
    expect(result.toolCalls[0]?.rawArguments).toBe("");
    expect(result.toolCalls[0]?.arguments).toEqual({});
  });

  it("rejects a repeated content_block_start for an existing index", () => {
    const a = createAssembler("anthropic");
    a.push({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
    expect(
      errorCode(() =>
        a.push({
          type: "content_block_start",
          index: 0,
          content_block: { type: "text", text: "" },
        }),
      ),
    ).toBe("invalid_event");
  });

  it("bounds the number of anthropic blocks", () => {
    const a = createAssembler("anthropic");
    for (let i = 0; i < 1024; i++) {
      a.push({ type: "content_block_start", index: i, content_block: { type: "thinking" } });
    }
    expect(
      errorCode(() =>
        a.push({ type: "content_block_start", index: 1024, content_block: { type: "thinking" } }),
      ),
    ).toBe("too_large");
  });

  it("rejects malformed Anthropic events", () => {
    const events = [
      { type: 42 },
      { type: "mystery" },
      { type: "content_block_start", index: "0", content_block: { type: "text" } },
      { type: "content_block_start", index: 0, content_block: null },
      { type: "text_delta", index: "0", text: "x" },
      { type: "message_delta", delta: null },
    ];
    for (const event of events) {
      const a = createAssembler("anthropic");
      expect(errorCode(() => a.push(event))).toBe("invalid_event");
    }
  });

  it("rejects non-string text and partial_json payloads", () => {
    const a = createAssembler("anthropic");
    a.push({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
    expect(errorCode(() => a.push({ type: "text_delta", index: 0, text: 1 }))).toBe(
      "invalid_event",
    );

    const b = createAssembler("anthropic");
    b.push({
      type: "content_block_start",
      index: 0,
      content_block: { type: "tool_use", id: "t1", name: "f", input: {} },
    });
    expect(errorCode(() => b.push({ type: "input_json_delta", index: 0, partial_json: 1 }))).toBe(
      "invalid_event",
    );
  });

  it("throws unknown_block for input_json_delta on a text block", () => {
    const a = createAssembler("anthropic");
    a.push({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
    expect(
      errorCode(() => a.push({ type: "input_json_delta", index: 0, partial_json: "{}" })),
    ).toBe("unknown_block");
  });

  it("ignores non-text blocks and message_delta without stop_reason", () => {
    const a = createAssembler("anthropic");
    a.push({ type: "content_block_start", index: 0, content_block: { type: "thinking" } });
    a.push({ type: "message_delta", delta: {} });
    a.push({ type: "message_stop" });
    const result = a.finish();
    expect(result.text).toBe("");
    expect(result.stopReason).toBeNull();
    expect(result.toolCalls).toEqual([]);
  });

  it("reports stream_error when the error object is missing", () => {
    const a = createAssembler("anthropic");
    expect(errorCode(() => a.push({ type: "error", error: null }))).toBe("stream_error");
  });

  it("snapshot reports null arguments for anthropic tool calls", () => {
    const a = createAssembler("anthropic");
    a.push({
      type: "content_block_start",
      index: 0,
      content_block: { type: "tool_use", id: "t1", name: "f", input: {} },
    });
    a.push({ type: "input_json_delta", index: 0, partial_json: '{"a"' });
    const snap = a.snapshot();
    expect(snap.toolCalls[0]?.arguments).toBeNull();
    expect(snap.toolCalls[0]?.rawArguments).toBe('{"a"');
  });

  it("orders anthropic tool blocks by index and defaults missing arguments to {}", () => {
    const a = createAssembler("anthropic");
    a.push({
      type: "content_block_start",
      index: 1,
      content_block: { type: "tool_use", id: "t2", name: "second", input: {} },
    });
    a.push({
      type: "content_block_start",
      index: 0,
      content_block: { type: "tool_use", id: "t1", name: "first", input: {} },
    });
    a.push({ type: "message_stop" });
    const result = a.finish();
    expect(result.toolCalls.map((tc) => tc.id)).toEqual(["t1", "t2"]);
    expect(result.toolCalls[0]?.arguments).toEqual({});
  });

  it("enforces maxTextChars for anthropic text", () => {
    const a = createAssembler("anthropic", { maxTextChars: 3 });
    a.push({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
    a.push({ type: "text_delta", index: 0, text: "abc" });
    expect(errorCode(() => a.push({ type: "text_delta", index: 0, text: "d" }))).toBe("too_large");
  });

  it("enforces maxArgumentChars", () => {
    const a = createAssembler("openai", { maxArgumentChars: 3 });
    expect(
      errorCode(() =>
        a.push({
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [{ index: 0, id: "c1", function: { name: "f", arguments: '{"a":' } }],
              },
            },
          ],
        }),
      ),
    ).toBe("too_large");
  });

  it("enforces maxToolCalls for openai", () => {
    const a = createAssembler("openai", { maxToolCalls: 1 });
    a.push({
      choices: [
        { index: 0, delta: { tool_calls: [{ index: 0, id: "c1", function: { name: "f" } }] } },
      ],
    });
    expect(
      errorCode(() =>
        a.push({
          choices: [
            { index: 0, delta: { tool_calls: [{ index: 1, id: "c2", function: { name: "g" } }] } },
          ],
        }),
      ),
    ).toBe("too_large");
  });

  it("enforces maxToolCalls for anthropic", () => {
    const a = createAssembler("anthropic", { maxToolCalls: 1 });
    a.push({
      type: "content_block_start",
      index: 0,
      content_block: { type: "tool_use", id: "t1", name: "f", input: {} },
    });
    expect(
      errorCode(() =>
        a.push({
          type: "content_block_start",
          index: 1,
          content_block: { type: "tool_use", id: "t2", name: "g", input: {} },
        }),
      ),
    ).toBe("too_large");
  });

  it("invalid_tool_arguments without an id omits the id from the message", () => {
    const a = createAssembler("openai");
    a.push({
      choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: "{" } }] } }],
    });
    a.push({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] });
    expect(() => a.finish()).toThrowError(AssembleError);
    try {
      a.finish();
    } catch (e) {
      expect((e as AssembleError).code).toBe("invalid_tool_arguments");
      expect((e as AssembleError).message).toBe("invalid tool arguments");
    }
  });
});
