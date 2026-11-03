# Template & tokenizer traps — the non-math ship-killers

More conversions have died here than on any numerical issue. Everything
below was hit on a real model on litert-lm 0.15–0.17; each entry gives the
observable signature — re-check the renderer's behaviour on each release.
The peek command below was re-run on 0.18.0.

## The renderer is minijinja, not Jinja2 — and not Python

The runtime renders chat templates with a Rust minijinja build (no Python
compatibility layer). Python-style *methods* on dicts/strings are
unknown at render time; support varies by runtime build, so the only safe
template uses attribute access, `|` filters, and plain tests. Observed
crash signatures (all thrown on the user's *first or second message*,
never at import):

```
Failed to apply template: unknown method: map has no method named get
Failed to apply template: unknown method: string has no method named strip
Failed to apply template: unknown method: string has no method named startswith
Failed to apply template: unknown method: string has no method named split
too many arguments            (tojson(ensure_ascii=False) vs the built-in filter)
```

A template can also crash **only on message two**: assistant-history
branches (reasoning re-render logic) execute for the first time when a
completed assistant turn enters the history. Single-turn tests cannot
catch this — the multi-turn gate exists for it. The repair pattern for a
crashing history branch in a non-thinking ship: replace the whole
assistant-history branch with a plain verbatim emit of the turn.

Three sharper facts about the render environment:

- **The `| split` *filter* form works where the `.split()` *method*
  crashes** — minijinja has the filter, jinja2 does not; method-to-filter
  rewrites are the safe direction.
- ⭐ **The pip wheel and an OSS source build of the same runtime version
  render differently** — a template that renders on the wheel has thrown
  `unknown method` on a source build. "It renders in my venv" is not
  evidence it renders in a shipped app; only the minimal-template route
  is build-independent.
- litert-torch 0.9.4 ships an **experimental jinja→minijinja transpiler**
  flag (`--experimental_transpile_chat_template_for_minijinja`). It
  rewrites some string methods to filters but not `.get()` /
  `.startswith()` — and **on any exception it returns the template
  unchanged**, so enabling it can look like a fix and silently not be
  one. Don't substitute it for the minimal template.

Also real jinja semantics, not a runtime bug: `{% set x = x ~ ... %}`
inside a `for` loop does not escape the loop — templates that accumulate
tool-call arguments that way silently emit empty arguments on re-render.

Exporter packaging behaviors worth knowing (litert-torch 0.9.x): a
`thought` channel is auto-declared when the packed template contains
`<think>`; stop tokens get punctuation-prefix expansion (a SentencePiece
greedy-merge guard); `sampler_top_k/top_p/temperature` export args bake a
sampler config (`top_k=1` = greedy); `suppress_tokens` is pulled from
`generation_config`. Check the peek output rather than assuming any of
these fired.

## The standing defense: embed no Jinja at all

Export with `use_jinja_template=False` and a **minimal** template swapped
onto the tokenizer (`recipe-selector.md` has the driver). The exporter
then extracts plain prefix/suffix turn markers and the bundle carries no
template code. Two caveats:

- Extraction can fail on complex vendor templates (namespaces, tool
  branches) and **fall back to embedding the raw Jinja** — exactly the
  trap. That is why the driver swaps in the minimal template rather than
  trusting extraction of the vendor one.
- Triage any bundle (yours or a third party's) in one line:

```bash
python -m litert_lm_builder.litertlm_peek_main --litertlm_file model.litertlm
```

`prompt_templates`-only = safe. A `jinja_prompt_template` containing
`.get(`/`.strip(`/`.split(` = the first-message crasher.

## Fixed system prompts belong in the user prefix

When a model's system prompt is part of its **inference contract** — the
card states it verbatim and the model was trained against it — rather than
something the caller tunes, ship it inside the user prefix instead of as a
system slot:

```
user.prefix = "<sys-open>" + THE_FIXED_SYSTEM_PROMPT + "<sys-close>" + "<turn-open>"
user.suffix = "<turn-close>"
```

The reason is API shape, not aesthetics: only the conversation path can
carry a `Role.SYSTEM` message, while `Session.run_prefill` (the scoring
entry point — `verification-gates.md` §0) takes plain strings. A separate
system slot therefore makes the model's own reference prompt *unreachable
from the API you actually classify with*, and every caller who forgets it
silently gets a different model.

Extraction handles this cleanly: a template that simply ignores the system
role takes the exporter's "no system prompt" path, and the fixed text
lands in the user prefix where it belongs. Verify with peek, then check
that one rendered turn is byte-identical to the vendor template's render.

Caveat to state on the card: a caller who *does* pass a system message
will have it dropped. That is the right trade when the prompt is fixed by
the model, and the wrong one when it is a real knob.

## Multi-turn: the prefix contract

At each message the engine renders the history *without* a generation
prompt and requires the new render to **string-extend** the previous one
— it prefills only the suffix. Hard failure:

```
new rendered template string does not start with the previous
```

Templates that rewrite history violate it — the classic case is a
reasoning template that strips `<think>` blocks from past assistant
turns: turn 2 dies. Fix pattern for hybrid-thinking models shipped
non-thinking: the generation prompt appends
`<|im_start|>assistant\n<think>\n\n</think>\n\n`, and history assistant
turns render **with the same empty think block**, so
render(history) ≡ what was actually prefilled + generated. Subtler
hazard even when the check passes: if the render diverges from the live
token stream, the engine resolves the difference by rewinding — safe for
slot-addressed KV, corrupting for running-state models. History renders
must be append-only extensions of the real stream.

The check runs a second time inside the refill: after a reply that
carried a thinking channel, the next message rewinds the KV cache and
re-prefills the history from the template, comparing that render with
the render up to the rewind point, so a template can pass message two
and die on message three. From LiteRT-LM 0.18.0 the template also
receives each reply's `reasoning_content` for every model type (0.17.1
did so only for the `lfm2` model type and, on the generic type, for
templates that read `content` only as parts; otherwise it passed a
one-part reply as a plain string without it), which is what exposes
every history turn rendered by its position: a think block kept only
on the latest assistant turn (`truncate_history_thinking`,
`preserve_thinking`) dies from 0.18.0 on, and on 0.17 already in those
two cases, and the Qwen3 `last_query_index` pattern and a trailing
token that only the non-generation render emits (an `{% else %}` branch
on `add_generation_prompt`) die on 0.17.1 as well, thinking on or off.
Before shipping a bundle that embeds Jinja, render its template on the
desk the way the runtime does (minijinja with `trim_blocks`,
`lstrip_blocks` and `keep_trailing_newline` on; replies as content
parts plus `channels` and `reasoning_content`) for three turns with
thinking unset, on and off, after matching one logged old / new pair
byte for byte: the INTERNAL error prints both strings.

## Tokenizer packaging

- **The HF-tokenizer bundle path can mis-tokenize.** Two observed forms:
  prompts garbled end-to-end (fluent output ignoring the prompt), and
  SentencePiece-style `▁` markers unhandled on decode so **all spaces
  vanish** ("Theusersaid42"). When the source ships a real SP model or a
  convertible BPE, bundle an `SP_Tokenizer`; verify by round-tripping a
  prompt through the packed tokenizer.
- **BPE→SP conversion trap**: converters that read `tokenizer.vocab_file`
  choke when it points at `vocab.json` ("Wire format was corrupt" — it
  expects an SP ModelProto). Clear `vocab_file` unless it ends in
  `.model`/`.spiece` so the conversion builds from vocab+merges.
- **Added special tokens get dropped** by SP conversion (they live in
  `added_tokens_decoder`, not the base vocab). A thinking model then
  *generates* `<think>` and the runtime crashes:

  ```
  NOT_FOUND: Token id 166103 is out of range. Vocab size is 166100
  ```

  Fix: append every added token as a `USER_DEFINED` piece **at its exact
  id**, pad with `UNUSED` pieces up to the model's embedding
  `vocab_size`. Test by encoding `<think>` and asserting the id. This
  bites models whose specials sit **beyond** the base vocab (typical for
  Llama/Mistral-based finetunes that append ChatML or `<think>` tokens)
  — also as a missing stop: `<|im_end|>` absent from the SP model means
  the stop never fires. It is a **no-op for the Qwen family**, whose
  specials live inside the base vocab — check
  `added_tokens_decoder` ids against the base vocab size before planning
  the patch.

## Stop tokens

- The bundle must declare **every** turn-end token, not just
  `config.json`'s `eos_token_id`. ChatML finetunes commonly stop on
  `<|im_end|>` while eos is `<|endoftext|>` — declaring only eos makes
  the literal `<|im_end|>` text leak into every reply. Current
  litert-torch exporters usually get this right by reading
  `generation_config.eos_token_id` (often a list) — so **verify via
  peek first, and intervene only if the union is missing**. The union =
  `generation_config.eos_token_id` ∪ the template's actual turn suffix.
- Metadata stop tokens can be token **ids** or literal **strings**;
  string stops catch a model that spells a stop marker in text form that
  never tokenizes to the stop id.
- Reusing a sibling model's metadata as the recipe carries the sibling's
  **token ids**. Stop ids are per-tokenizer, not per-family: the same
  `<|im_end|>` was id 7 in a 1.2B and 124900 in its 2.6B sibling.
  Re-derive every id from the target model's tokenizer before packing —
  peek only proves ids are present, not that they are the right ones.

## Thinking-model metadata

- Declare the thought channel with the model's real markers:
  `channels { channel_name: "thought" start: "<think>" end: "</think>" }`
  — placeholder marker strings never fire and the thought text floods
  the main channel.
- Thinking-budget enforcement is inert unless the metadata carries
  `thinking_end_token_ids` (the runtime logs a warning and ignores the
  budget).
- Beware `enable_thinking | default(...)` in templates: the engine
  injects `enable_thinking` into the render context only when a thinking
  config is set, so the template's default silently decides behavior for
  every plain conversation. Choose the default to match how you gated
  the model.

## Adding tool_call_config to a generic_model bundle

From LiteRT-LM 0.19.0, a bundle whose `llm_model_type` is
`generic_model` and whose LlmMetadata carries `tool_call_config` (field
15 in `runtime/proto/llm_metadata.proto`) returns the model's tool calls
as `tool_calls` (function name and JSON arguments); without it, the call
stays in the reply as text. The runtime accepts the config only on
`generic_model` (`tool_call_config is only supported for generic_model.`,
so a `fast_vlm` bundle is refused), and it parses only what the model
writes: if the chat template does not render the `tools` it is given,
the model never sees them and the config alone is not enough.

Unpack the bundle with `litert-lm unpack`, add the two fields below to
its `LlmMetadataProto.pbtext` and pack it with `litert-lm pack` from
0.19.0 (flags and the overwrite trap: `recipe-selector.md` §Hybrids,
point 1; gates: `verification-gates.md` §6); for the Qwen3.5 call format
the fields are those of LiteRT-LM's
`models/qwen3_5/LlmMetadataProto.pbtext` at v0.19.0, byte for byte:

```
supports_function_calling: true
tool_call_config {
  fence_start: "<tool_call>"
  fence_end: "</tool_call>"
  regex_syntax {
    tool_call_regex: "(?s)<function=([^>]+)>(.*?)</function>"
    argument_regex: "(?s)<parameter=([^>]+)>\\s*(.*?)\\s*</parameter>"
  }
}
```

Change nothing else: weights, template, stop tokens, sampler and model
type stay as they were, and `supports_function_calling` is only reported
back through `ModelInfo` (the parsing comes from `tool_call_config`). Do
not copy the upstream file's `pad_token` (`<|endoftext|>`) into a bundle
you exported yourself without reading your graph's padding guard: 0.19.0
fills the unused positions of each prefill input with the metadata's pad
token, and with id 0 only when none is declared. In a Qwen3.5 hybrid
export whose guard treats only id 0 as padding, the copied token changed
the greedy answer on 2 of 3 prompts on CPU and 3 of 3 on GPU (Qwen3.5-4B
int8 on a Mac, 0.19.0), while the same re-pack without it matched the
original on all 3 CPU prompts. Leave out the upstream file's
`min_runtime_version: "0.19.0"` as well and keep the card's minimum:
re-packed files loaded on 0.15.0, 0.18.0 and 0.18.1 and gave the
original's answer to a plain prompt; run yours once on the oldest
runtime the card names, and say on the card that tool-call parsing needs
0.19.0.

Parsing runs only when the conversation is given tools, and then a
closed `<tool_call>…</tool_call>` block that matches no `tool_call_regex`
fails the whole reply with `INVALID_ARGUMENT: Failed to parse tool calls
from code block`. The switch that keeps such a block as text
(`return_error_on_parse_failure`, default true) exists only in the C++
API (`ConversationConfig::Builder::SetReturnErrorOnParseFailure`); no
other language binding of 0.19.0 exposes it. Seen with Qwen3.5-4B Mixed
INT4 on 0.19.0 (Python API, Mac CPU): with tools passed and a request
for a Python function that removes the `<tool_call>` and `</tool_call>`
tags, the model wrote both tags in its code and the reply failed, where
the original file returned the same code as text; list the case on the
card as a known limitation, naming the prompt.

## Start token

BOS convention lives in `recipe-selector.md` §Start token — wrong BOS is
a metadata bug that looks exactly like a quality problem, and small
models are the ones it flips. Check what the exporter actually shipped
before planning a repair: no `start_token` block in the peek output (and
a None BOS from the engine API) means the bundle already matches a
no-BOS template.
