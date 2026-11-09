/**
 * Public entry point of @trysac/token-stream-assembler.
 *
 * Assembles streamed LLM deltas from OpenAI chat completion chunks or Anthropic message events into
 * one complete message with text, parsed tool call arguments and stop reason.
 */

/** Supported wire dialects. */
export type Dialect = "openai" | "anthropic";

/** A fully assembled tool call, present after `finish()` has parsed the raw arguments. */
export interface ToolCall {
  id: string;
  name: string;
  /** Parsed JSON arguments, or `null` in snapshots before parsing. */
  arguments: unknown;
  /** The exact raw argument string received from the stream. */
  rawArguments: string;
}

/** The message returned by `snapshot()` and `finish()`. */
export interface AssembledMessage {
  role: "assistant";
  text: string;
  toolCalls: ToolCall[];
  stopReason: string | null;
  done: boolean;
}

/** Error code union used by {@link AssembleError}. */
export type AssembleErrorCode =
  | "invalid_event"
  | "unsupported_choice"
  | "unknown_block"
  | "invalid_tool_arguments"
  | "incomplete"
  | "after_finish"
  | "stream_error"
  | "too_large";

/**
 * Error thrown by the assembler. The `code` property is a stable machine-readable identifier.
 * After any error is thrown, the assembler is poisoned and every subsequent call throws the same error.
 */
export class AssembleError extends Error {
  readonly code: AssembleErrorCode;

  constructor(code: AssembleErrorCode, message?: string) {
    super(message ?? code);
    this.name = "AssembleError";
    this.code = code;
  }
}

/** Options for {@link createAssembler}. */
export interface AssemblerOptions {
  /** Maximum allowed total text characters across all text deltas. Default: 1_000_000. */
  maxTextChars?: number;
  /** Maximum allowed number of tool calls. Default: 128. */
  maxToolCalls?: number;
  /** Maximum allowed raw JSON argument characters per tool call. Default: 1_000_000. */
  maxArgumentChars?: number;
}

/**
 * A stateful assembler for one message stream.
 */
export interface Assembler {
  /**
   * Push a single already-parsed event object (OpenAI chunk or Anthropic event).
   * @param event - The event object. Must be a non-null object corresponding to the configured dialect.
   * @throws {AssembleError} If the event is invalid, the stream is already finished or an internal limit is exceeded.
   */
  push(event: unknown): void;

  /**
   * Return the current assembled state without terminating the stream.
   * Tool call arguments are reported as `null` until `finish()` parses them.
   * @throws {AssembleError} If the assembler has been poisoned by a previous error.
   */
  snapshot(): AssembledMessage;

  /**
   * Finish the stream. Requires the terminal event to have been received.
   * Parses each tool call's `rawArguments` with `JSON.parse` (`''` becomes `{}`).
   * Calling `finish()` again returns an equal value without re-parsing.
   * @throws {AssembleError} `incomplete` if the terminal event was not seen, `invalid_tool_arguments` if parsing fails,
   *   or the previously thrown error if the assembler is poisoned.
   */
  finish(): AssembledMessage;
}

interface ToolDraft {
  id: string;
  name: string;
  rawArguments: string;
}

type BlockKind = "text" | "tool_use" | "other";

interface BlockInfo {
  kind: BlockKind;
  id?: string;
  name?: string;
  rawArguments?: string;
}

const DEFAULT_MAX_TEXT_CHARS = 1_000_000;
const DEFAULT_MAX_TOOL_CALLS = 128;
const DEFAULT_MAX_ARGUMENT_CHARS = 1_000_000;
const MAX_ANTHROPIC_BLOCKS = 1024;

class AssemblerImpl implements Assembler {
  private readonly dialect: Dialect;
  private readonly maxTextChars: number;
  private readonly maxToolCalls: number;
  private readonly maxArgumentChars: number;

  // Text accumulation
  private openAiText = "";
  private anthropicTextParts = new Map<number, string>();
  private anthropicTotalTextChars = 0;

  // Tool call accumulation
  private openAiToolDrafts = new Map<number, ToolDraft>();
  private anthropicBlocks = new Map<number, BlockInfo>();
  private anthropicToolCallCount = 0;

  private stopReason: string | null = null;
  private terminalSeen = false;
  private finishedSuccessfully = false;
  private finalResult: AssembledMessage | null = null;
  private poisonError: AssembleError | null = null;

  constructor(dialect: Dialect, opts?: AssemblerOptions) {
    this.dialect = dialect;
    this.maxTextChars = opts?.maxTextChars ?? DEFAULT_MAX_TEXT_CHARS;
    this.maxToolCalls = opts?.maxToolCalls ?? DEFAULT_MAX_TOOL_CALLS;
    this.maxArgumentChars = opts?.maxArgumentChars ?? DEFAULT_MAX_ARGUMENT_CHARS;
  }

  push(event: unknown): void {
    this.throwIfPoisoned();

    // Usage-only OpenAI chunks (empty choices) carry no content and are always accepted and
    // ignored, even after the terminal finish_reason chunk of a `stream_options.include_usage` stream.
    if (typeof event === "object" && event !== null) {
      const choices = (event as Record<string, unknown>).choices;
      if (this.dialect === "openai" && Array.isArray(choices) && choices.length === 0) {
        return;
      }
    }

    if (this.terminalSeen) {
      this.poisonError = new AssembleError("after_finish", "after finish");
      throw this.poisonError;
    }

    if (typeof event !== "object" || event === null) {
      this.poisonError = new AssembleError("invalid_event", "invalid event");
      throw this.poisonError;
    }

    if (this.dialect === "openai") {
      this.pushOpenAi(event as Record<string, unknown>);
    } else {
      this.pushAnthropic(event as Record<string, unknown>);
    }
  }

  snapshot(): AssembledMessage {
    this.throwIfPoisoned();
    return this.buildMessage();
  }

  finish(): AssembledMessage {
    this.throwIfPoisoned();
    if (this.finishedSuccessfully) {
      return this.finalResult as AssembledMessage;
    }
    if (!this.terminalSeen) {
      this.poisonError = new AssembleError("incomplete", "incomplete");
      throw this.poisonError;
    }

    const toolCalls = this.parseToolCalls();
    const text = this.getFinalText();
    const result: AssembledMessage = {
      role: "assistant",
      text,
      toolCalls,
      stopReason: this.stopReason,
      done: true,
    };
    this.finalResult = result;
    this.finishedSuccessfully = true;
    return result;
  }

  private pushOpenAi(event: Record<string, unknown>): void {
    const choices = event.choices;
    if (!Array.isArray(choices)) {
      this.poisonError = new AssembleError("invalid_event", "invalid event");
      throw this.poisonError;
    }
    if (choices.length === 0) {
      // Usage-only chunk: accepted and ignored.
      return;
    }
    const choice = choices[0];
    if (typeof choice !== "object" || choice === null) {
      this.poisonError = new AssembleError("invalid_event", "invalid event");
      throw this.poisonError;
    }
    const choiceObj = choice as Record<string, unknown>;
    if (choiceObj.index !== 0) {
      this.poisonError = new AssembleError("unsupported_choice", "unsupported choice");
      throw this.poisonError;
    }

    const delta = choiceObj.delta;
    if (typeof delta === "object" && delta !== null) {
      const deltaObj = delta as Record<string, unknown>;
      if (typeof deltaObj.content === "string") {
        this.appendOpenAiText(deltaObj.content);
      }
      const toolCalls = deltaObj.tool_calls;
      if (Array.isArray(toolCalls)) {
        for (const rawTc of toolCalls) {
          this.processOpenAiToolCall(rawTc);
        }
      }
    }

    if (choiceObj.finish_reason !== null && choiceObj.finish_reason !== undefined) {
      this.stopReason = String(choiceObj.finish_reason);
      this.terminalSeen = true;
    }
  }

  private processOpenAiToolCall(rawTc: unknown): void {
    if (typeof rawTc !== "object" || rawTc === null) {
      this.poisonError = new AssembleError("invalid_event", "invalid event");
      throw this.poisonError;
    }
    const tc = rawTc as Record<string, unknown>;
    const index = tc.index;
    if (typeof index !== "number" || !Number.isInteger(index) || index < 0) {
      this.poisonError = new AssembleError("invalid_event", "invalid event");
      throw this.poisonError;
    }
    let draft = this.openAiToolDrafts.get(index);
    if (!draft) {
      if (this.openAiToolDrafts.size >= this.maxToolCalls) {
        this.poisonError = new AssembleError("too_large", "too large");
        throw this.poisonError;
      }
      const fn = tc.function;
      const fnObj = (typeof fn === "object" && fn !== null ? fn : {}) as Record<string, unknown>;
      draft = {
        id: typeof tc.id === "string" ? tc.id : "",
        name: typeof fnObj.name === "string" ? fnObj.name : "",
        rawArguments: "",
      };
      this.openAiToolDrafts.set(index, draft);
    }
    // Later fragments must not override id/name.
    const fn = tc.function;
    if (typeof fn === "object" && fn !== null) {
      const fnObj = fn as Record<string, unknown>;
      if (typeof fnObj.arguments === "string") {
        this.appendToolArguments(draft, fnObj.arguments);
      }
    }
  }

  private appendOpenAiText(piece: string): void {
    if (this.openAiText.length + piece.length > this.maxTextChars) {
      this.poisonError = new AssembleError("too_large", "too large");
      throw this.poisonError;
    }
    this.openAiText += piece;
  }

  private pushAnthropic(event: Record<string, unknown>): void {
    const type = event.type;
    if (typeof type !== "string") {
      this.poisonError = new AssembleError("invalid_event", "invalid event");
      throw this.poisonError;
    }
    switch (type) {
      case "ping":
      case "message_start":
      case "content_block_stop":
        // Accepted and ignored (or handled by message_start in future).
        return;
      case "error": {
        const errObj = event.error;
        const errType =
          typeof errObj === "object" &&
          errObj !== null &&
          typeof (errObj as Record<string, unknown>).type === "string"
            ? (errObj as Record<string, unknown>).type
            : "unknown";
        this.poisonError = new AssembleError("stream_error", `stream error: ${errType}`);
        throw this.poisonError;
      }
      case "content_block_start": {
        const index = event.index;
        if (typeof index !== "number" || !Number.isInteger(index) || index < 0) {
          this.poisonError = new AssembleError("invalid_event", "invalid event");
          throw this.poisonError;
        }
        const block = event.content_block;
        if (typeof block !== "object" || block === null) {
          this.poisonError = new AssembleError("invalid_event", "invalid event");
          throw this.poisonError;
        }
        const blockObj = block as Record<string, unknown>;
        if (this.anthropicBlocks.has(index)) {
          this.poisonError = new AssembleError("invalid_event", "duplicate block");
          throw this.poisonError;
        }
        if (this.anthropicBlocks.size >= MAX_ANTHROPIC_BLOCKS) {
          this.poisonError = new AssembleError("too_large", "too large");
          throw this.poisonError;
        }
        const blockType = blockObj.type;
        if (blockType === "text") {
          this.anthropicBlocks.set(index, { kind: "text" });
          this.anthropicTextParts.set(index, "");
        } else if (blockType === "tool_use") {
          if (this.anthropicToolCallCount >= this.maxToolCalls) {
            this.poisonError = new AssembleError("too_large", "too large");
            throw this.poisonError;
          }
          this.anthropicToolCallCount++;
          this.anthropicBlocks.set(index, {
            kind: "tool_use",
            id: typeof blockObj.id === "string" ? blockObj.id : "",
            name: typeof blockObj.name === "string" ? blockObj.name : "",
            rawArguments: "",
          });
        } else {
          // Other block types (e.g. thinking) are tracked and their deltas ignored.
          this.anthropicBlocks.set(index, { kind: "other" });
        }
        return;
      }
      case "text_delta": {
        const index = event.index;
        if (typeof index !== "number" || !Number.isInteger(index) || index < 0) {
          this.poisonError = new AssembleError("invalid_event", "invalid event");
          throw this.poisonError;
        }
        const block = this.anthropicBlocks.get(index);
        if (block?.kind !== "text") {
          this.poisonError = new AssembleError("unknown_block", `unknown block ${index}`);
          throw this.poisonError;
        }
        const text = event.text;
        if (typeof text !== "string") {
          this.poisonError = new AssembleError("invalid_event", "invalid event");
          throw this.poisonError;
        }
        this.appendAnthropicText(index, text);
        return;
      }
      case "input_json_delta": {
        const index = event.index;
        if (typeof index !== "number" || !Number.isInteger(index) || index < 0) {
          this.poisonError = new AssembleError("invalid_event", "invalid event");
          throw this.poisonError;
        }
        const block = this.anthropicBlocks.get(index);
        if (block?.kind !== "tool_use") {
          this.poisonError = new AssembleError("unknown_block", `unknown block ${index}`);
          throw this.poisonError;
        }
        const partialJson = event.partial_json;
        if (typeof partialJson !== "string") {
          this.poisonError = new AssembleError("invalid_event", "invalid event");
          throw this.poisonError;
        }
        const toolDraft: ToolDraft = {
          id: block.id ?? "",
          name: block.name ?? "",
          rawArguments: block.rawArguments ?? "",
        };
        this.appendToolArguments(toolDraft, partialJson);
        block.rawArguments = toolDraft.rawArguments;
        return;
      }
      case "message_delta": {
        const delta = event.delta;
        if (typeof delta !== "object" || delta === null) {
          this.poisonError = new AssembleError("invalid_event", "invalid event");
          throw this.poisonError;
        }
        const stopReason = (delta as Record<string, unknown>).stop_reason;
        if (stopReason !== null && stopReason !== undefined) {
          this.stopReason = String(stopReason);
        }
        return;
      }
      case "message_stop": {
        this.terminalSeen = true;
        return;
      }
      default:
        this.poisonError = new AssembleError("invalid_event", "invalid event");
        throw this.poisonError;
    }
  }

  private appendAnthropicText(index: number, piece: string): void {
    const current = this.anthropicTextParts.get(index) ?? "";
    if (this.anthropicTotalTextChars + piece.length > this.maxTextChars) {
      this.poisonError = new AssembleError("too_large", "too large");
      throw this.poisonError;
    }
    this.anthropicTextParts.set(index, current + piece);
    this.anthropicTotalTextChars += piece.length;
  }

  private appendToolArguments(draft: ToolDraft, piece: string): void {
    if (draft.rawArguments.length + piece.length > this.maxArgumentChars) {
      this.poisonError = new AssembleError("too_large", "too large");
      throw this.poisonError;
    }
    draft.rawArguments += piece;
  }

  private parseToolCalls(): ToolCall[] {
    const result: ToolCall[] = [];
    if (this.dialect === "openai") {
      const sortedIndexes = Array.from(this.openAiToolDrafts.keys()).sort((a, b) => a - b);
      for (const idx of sortedIndexes) {
        const draft = this.openAiToolDrafts.get(idx) as ToolDraft;
        result.push(this.parseToolDraft(draft));
      }
    } else {
      const sortedIndexes = Array.from(this.anthropicBlocks.keys())
        .filter((idx) => this.anthropicBlocks.get(idx)?.kind === "tool_use")
        .sort((a, b) => a - b);
      for (const idx of sortedIndexes) {
        const block = this.anthropicBlocks.get(idx) as BlockInfo & { rawArguments: string };
        const draft: ToolDraft = {
          id: block.id ?? "",
          name: block.name ?? "",
          rawArguments: block.rawArguments ?? "",
        };
        result.push(this.parseToolDraft(draft));
      }
    }
    return result;
  }

  private parseToolDraft(draft: ToolDraft): ToolCall {
    const raw = draft.rawArguments;
    let args: unknown;
    if (raw === "") {
      args = {};
    } else {
      try {
        args = JSON.parse(raw);
      } catch {
        const idPart = draft.id ? ` for ${draft.id}` : "";
        this.poisonError = new AssembleError(
          "invalid_tool_arguments",
          `invalid tool arguments${idPart}`,
        );
        throw this.poisonError;
      }
    }
    return {
      id: draft.id,
      name: draft.name,
      arguments: args,
      rawArguments: raw,
    };
  }

  private getFinalText(): string {
    if (this.dialect === "openai") {
      return this.openAiText;
    }
    const sortedIndexes = Array.from(this.anthropicTextParts.keys()).sort((a, b) => a - b);
    return sortedIndexes.map((idx) => this.anthropicTextParts.get(idx) ?? "").join("");
  }

  private buildMessage(): AssembledMessage {
    const text = this.getFinalText();
    const toolCalls: ToolCall[] = [];
    if (this.dialect === "openai") {
      const sortedIndexes = Array.from(this.openAiToolDrafts.keys()).sort((a, b) => a - b);
      for (const idx of sortedIndexes) {
        const draft = this.openAiToolDrafts.get(idx) as ToolDraft;
        toolCalls.push({
          id: draft.id,
          name: draft.name,
          arguments: null,
          rawArguments: draft.rawArguments,
        });
      }
    } else {
      const toolBlockIndexes = Array.from(this.anthropicBlocks.keys())
        .filter((idx) => this.anthropicBlocks.get(idx)?.kind === "tool_use")
        .sort((a, b) => a - b);
      for (const idx of toolBlockIndexes) {
        const block = this.anthropicBlocks.get(idx) as BlockInfo & { rawArguments: string };
        const draft: ToolDraft = {
          id: block.id ?? "",
          name: block.name ?? "",
          rawArguments: block.rawArguments ?? "",
        };
        toolCalls.push({
          id: draft.id,
          name: draft.name,
          arguments: null,
          rawArguments: draft.rawArguments,
        });
      }
    }
    return {
      role: "assistant",
      text,
      toolCalls,
      stopReason: this.stopReason,
      done: this.finishedSuccessfully,
    };
  }

  private throwIfPoisoned(): void {
    if (this.poisonError) {
      throw this.poisonError;
    }
  }
}

/**
 * Create a stateful assembler for the given dialect.
 *
 * @param dialect - `'openai'` for OpenAI chat completion chunks, `'anthropic'` for Anthropic message events.
 * @param opts - Optional limits.
 * @returns A new {@link Assembler} instance.
 */
export function createAssembler(dialect: Dialect, opts?: AssemblerOptions): Assembler {
  return new AssemblerImpl(dialect, opts);
}

/**
 * Convenience function that pushes all events from an iterable and then finishes.
 *
 * @param dialect - The stream dialect.
 * @param events - An iterable of already-parsed event objects.
 * @returns The final {@link AssembledMessage}.
 */
export function assemble(dialect: Dialect, events: Iterable<unknown>): AssembledMessage {
  const assembler = createAssembler(dialect);
  for (const event of events) {
    assembler.push(event);
  }
  return assembler.finish();
}
