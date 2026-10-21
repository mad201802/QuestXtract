# Command line

How to run QuestXtract from a terminal: every command, every option, and recipes for the common
jobs. For the library API see the [README](../README.md#python-api); for how the pipeline works
inside, [architecture.md](architecture.md).

## Quick start

```bash
questxtract process lecture.mp4                # video -> lecture.questions.json
questxtract transcribe lecture.mp4             # transcription only -> lecture.transcript.json
questxtract process lecture.transcript.json    # questions from an existing transcript
questxtract filter lecture.questions.json --text eigenvalue   # the questions about a word
```

Run `questxtract --help`, or `questxtract <command> --help`, for the same information in the
terminal. From a source checkout, put `uv run` in front: `uv run questxtract process lecture.mp4`.

## What decides how much runs

The file you pass in. There are no flags for picking stages.

| You pass | What runs |
|---|---|
| an `.mp4` or `.mp3` recording | everything: check the recording, transcribe it, then find the questions; a recording without a video track (any `.mp3`, or an audio-only `.mp4`) has no board to read |
| a `.json` transcript | everything after transcription (much faster: no speech recognition) |
| anything else | nothing: `error: ...: expected an .mp4 recording or an .mp3 recording or a .json transcript` |

## Commands

The defaults below apply when neither a config file nor an explicit flag supplies a value.

### `questxtract process <input>`

Finds the questions asked in a recording or a transcript and writes them to a questions file.

| Option | Takes | Default | What it does |
|---|---|---|---|
| `-o`, `--output` | a path | next to the input | Where to write the questions. `-o -` prints them to the terminal instead. A folder in the path that doesn't exist yet is created. |
| `-t`, `--threshold` | a number, 0 to 1 | `0.4` | Keeps only questions with at least this confidence. Lower gives more questions, including more false ones; higher gives fewer. |
| `-m`, `--model` | a model name | `turbo` | The Whisper model used to transcribe. Only matters for `.mp4` or `.mp3` input. See [Choosing a model](#choosing-a-model). |
| `--from-scratch` | nothing | off | Ignores cached results and recomputes everything, transcription included. |
| `--detector` | `rules` or `llm` | `rules` | How questions are found: fast offline rules, or a language model that reads the transcript. See [Finding questions with an LLM](#finding-questions-with-an-llm). |
| `--llm-model` | a model name | `gpt-4o` | The model for `--detector llm`, as the endpoint names it, e.g. `qwen3.5:9b` in Ollama. |
| `--llm-url` | a URL | OpenAI | An OpenAI-compatible endpoint for `--detector llm`, e.g. `http://localhost:11434/v1` for Ollama. Without it, OpenAI is used with the key from `OPENAI_API_KEY`. Vision goes there too, with the same key, unless `vision_base_url` is set in `questxtract.toml`. |
| `--llm-reasoning` | `none`, `low`, `medium`, `high`, `server` | `server` | How long a thinking model may think before it judges. Slower, and what small models such as `qwen3.5:9b` need to judge well. `server` sends no setting and leaves it to the server, for a model without one such as `gpt-4o`, the default. `low`, `medium` and `high` allow each request 10 minutes instead of 2; for a server that thinks on its own with `server`, raise `timeout_seconds` in the settings file. |
| `--vision`, `--no-vision` | nothing | off | Also reads the slides or the board with a vision model: by default on NTNU's LLM service with the key from `QUESTXTRACT_VISION_API_KEY` (NTNU network or VPN), or at `vision_base_url`, or at `--llm-url` when only that is set. Needs the `questxtract[llm]` extra. `--no-vision` turns off `vision_enabled` from the settings file for one run. A recording without video (an `.mp3`, an audio-only `.mp4`) or a transcript has no board, so the board is not read (a note says so). See [Reading the board](../README.md#reading-the-board). |
| `--vision-model` | a model name | `Qwen/Qwen3.8-27B-FP8` | The model that reads the board, as the endpoint names it. It must take images. |
| `--vision-reader` | `vlm` or `ocr` | `vlm` | How keyframes are read: `vlm`, the vision model (text, formulas and a description), or `ocr`, `vision_ocr_model` (GLM-OCR: text and formulas only, faster). The layout probe always asks `--vision-model`. |
| `--quiz` | nothing | off | Also writes a multiple-choice quiz next to the questions file (`lecture.quiz.json`), as [`questxtract quiz`](#questxtract-quiz-questionsjson) does. Needs a file output, so not `-o -`. If the quiz fails, the questions file is kept and the exit code is 1. |
| `--quiz-model` | a model name | `gpt-4o` | The model that writes the quiz. Only with `--quiz`. |
| `--strictness`, `--options` | | | As for `questxtract quiz`. Only with `--quiz`. |
| `-h`, `--help` | nothing | | Shows the options and examples. |

The vision options may still change while the vision work continues in M3 (#133). The other vision settings (`vision_fps`, `vision_states_per_minute`, `vision_min_states`, `vision_change_fraction`,
`vision_change_hold`) are set in the TOML file only.

One thing to know about `-t` with the rules: a question with no answer scores at most `0.55`, so
a threshold above `0.55` keeps only questions the next sentence answers. With `--detector llm`
the confidence is the model's own, answered or not.

### `questxtract quiz <questions.json>`

Writes multiple-choice quiz items for the questions in a questions file: per question a stem,
options with one correct, an explanation, and why each option is right or wrong. It is the only
command that writes text that was not spoken: every text carries `origin` (`spoken` only when it
is the question as it was asked), and every item starts as pending review.

The model sees the question, the lecture's answer when there is one, what was said in the few
sentences around it, and the text, LaTeX and description of the board states on screen when the
lecture was processed with vision. The speech comes from `--transcript`, else from the cached
transcript of the recording (when `process` ran on this machine); without either, a `note:`
says so and the quiz is written from the question, answer and board alone, which gives fewer and
weaker items. `process --quiz` always has the speech. Rejected
questions (`review.status` `deleted` or `false_positive`) and classroom check-ins are left out,
and so is a question the model judges not to be quiz material: addressed to the room,
rhetorical, or too vague to answer on its own.
A question the model fails on is left out with a `warning:` line, and the rest are written; an
unreachable endpoint or a bad key ends the run. Answers are cached, so a re-run sends nothing,
and the same input gives the same file. The last line gives the model requests, tokens and time
the quiz cost; a transcript that does not contain the questions' sentences gets a `note:` and is
not used.

| Option | Takes | Default | What it does |
|---|---|---|---|
| `-o`, `--output` | a path | next to the input | Where to write the quiz: `lecture.questions.json` gives `lecture.quiz.json`. `-o -` prints it instead. |
| `--strictness` | `strict` or `lenient` | `lenient` | `lenient` also uses questions the lecture did not answer, whose correct answer the model writes from what was said around them; `strict` uses only questions the lecture answered. |
| `--options` | a number, 3 to 5 | `3` | Options per item, one of them correct. |
| `--transcript` | a path | the cached transcript | The lecture's transcript, for what was said around each question. |
| `--quiz-model` | a model name | `gpt-4o` | The model that writes the quiz (`quiz_model`). |
| `--llm-url` | a URL | OpenAI | An OpenAI-compatible endpoint, as for `process`. |
| `--llm-reasoning` | `none`, `low`, `medium`, `high`, `server` | `server` | As for `process`. |
| `--config` | a path | `questxtract.toml` | The settings file. `quiz_model`, `quiz_strictness` and `quiz_options` can be set there. |
| `-h`, `--help` | nothing | | Shows the options and examples. |

### `questxtract transcribe <recording>`

Transcribes an `.mp4` or `.mp3` recording into a transcript file, without looking for questions. The file
it writes is exactly what `process` accepts as input, and what `scripts/dropped_questions.py`
reads.

| Option | Takes | Default | What it does |
|---|---|---|---|
| `-o`, `--output` | a path | next to the recording | Where to write the transcript. `-o -` prints it instead. |
| `-m`, `--model` | a model name | `turbo` | The Whisper model used to transcribe. |
| `--from-scratch` | nothing | off | Transcribes again even if a cached transcript exists. |
| `-h`, `--help` | nothing | | Shows the options and examples. |

### `questxtract filter <questions.json>`

Keeps the questions of a questions file that mention some words or fall in a time range, without
running the pipeline again. It prints the result unless `-o` is given. The result is a questions
file like the input, with only the matching questions.

| Option | Takes | Default | What it does |
|---|---|---|---|
| `--text` | words | | Keeps questions where every word appears, as a whole word and ignoring case, somewhere in the question, the sentence it was spoken in, its answer, its `review` text, or the text, LaTeX or description of a board state on screen while it was asked. The words may come from different places: `--text "eigenvalue zero"` finds "what if this is zero?" asked under a board that says "eigenvalue". `eigenvalue` does not find `eigenvalues`. |
| `--from` | a time | start | Keeps questions asked at or after this time: seconds (`300`), `M:SS` (`5:00`) or `H:MM:SS` (`1:05:00`). |
| `--to` | a time | end | Keeps questions asked before this time, in the same forms. |
| `-o`, `--output` | a path | the terminal | Where to write the result. Keyframe paths are rewritten so the images are still found from there. |
| `-h`, `--help` | nothing | | Shows the options and examples. |

Give at least one of `--text`, `--from` and `--to`; together, a question must match all of them.

### `questxtract qti <quiz.json>`

Exports a quiz file as a QTI ZIP package for an LMS: QTI 3.0 by default, or QTI 2.1 for an LMS
that does not import 3.0. Every item becomes a single-choice item scored against its correct
option, with each option's explanation as feedback; the quiz's instructions set shuffling and
whether feedback is shown. Items a reviewer marked `rejected` are left out. The same quiz file
and version always give the same ZIP, byte for byte.

**QTI 2.1 has not been tried in a real LMS yet.** Blackboard imports QTI 2.1 only, and Canvas New
Quizzes imports 2.x but not 3.0, but no package has been imported into either. Blackboard skips
questions whose response processing it does not support. Each 2.1 item scores with its own rules
and also names the standard `match_correct` template, which an LMS that supports only templates
can use instead: there an item scores 1 and shows no feedback per option. After an import, check
that every question arrived.

| Option | Takes | Default | What it does |
|---|---|---|---|
| `-o`, `--output` | a path | required | Where to write the ZIP. Its folder must exist. |
| `--qti-version` | `2.1` or `3.0` | `3.0` | The QTI version of the package: 2.1 for an LMS that does not import 3.0, with the caveat above. |
| `-h`, `--help` | nothing | | Shows the options. |

### Everywhere

| Option | What it does |
|---|---|
| `--version` | Prints the installed version. |
| `-h`, `--help` | Without a command, lists the commands; after one, lists that command's options. |

`questxtract eval <files|dirs> [--json] [--board]` scores the pipeline against the labelled set: recall,
precision, false positives by kind, a threshold sweep, and every miss
([tests/data/README.md](../tests/data/README.md)). `--board` adds the board states committed
beside each transcript (`<name>.visual.json`), as if `--vision` had read them, and names the
recordings that had them on the report's first line. It takes `--detector`, `--llm-model` and
`--llm-url` too, so the rules and a model can be compared on the same recordings; the report's
first line names the detector, the model and the prompt version. With `--score-vision`, it
scores the vision model instead (`--vision-model`, or with `--vision-reader ocr` the OCR model, at vision's endpoint, as for `--vision`) on the labelled
keyframes in `tests/data/vision/`: character error rate, blank boards read as blank and formula
recall per lecture type, then every frame, worst first ([tests/data/vision/README.md](../tests/data/vision/README.md)). It calls
the model once per frame and caches the answers.

## Configuration files

`process`, `transcribe` and `eval` accept `--config PATH`. Without that option, they look for
`questxtract.toml` in the **current working directory**, not next to the recording. If that file
does not exist, the commands use their defaults. An explicitly selected missing file is an error.
When `--config` is given, only that file is read, even if `questxtract.toml` also exists.

The precedence is **explicit command flag > file value > Config default**. For example:

```toml
# questxtract.toml: one flat table, with the names from config.py
model_size = "small"
min_confidence = 0.6
detector = "rules"
judge_window = 10
cache_dir = ".questxtract"
```

```bash
questxtract process lecture.mp4                    # uses small and threshold 0.6
questxtract process lecture.mp4 --threshold 0.3    # overrides only the threshold
questxtract transcribe lecture.mp4 --config other.toml
questxtract eval tests/data/ --config other.toml
```

Every `Config` field can be set, including those without flags. Names must match exactly:
`--model` corresponds to `model_size`, `--threshold` to `min_confidence`, `--llm-model` to
`judge_model`, `--llm-url` to `base_url`, and `--llm-reasoning` to `judge_reasoning_effort`.
`judge_reasoning_effort = ""`, the default, is the equivalent of `--llm-reasoning server`. A
reasoning level (`"low"`, `"medium"` or `"high"`) raises the request timeout to at least 600
seconds for `process` and `eval`, including when it comes from the file.

The file must be UTF-8; a leading BOM from a Windows editor is accepted. No nested sections or
`[tool.questxtract]` table are read. Integer fields require integers, not `5.5` or `true`;
float fields also accept integers. TOML has no null: leave an optional key out to retain its
default. Typos and wrong types are errors. A threshold must be between 0 and 1, and the
detector must be `rules` or `llm`. Relative `cache_dir` paths are resolved from the **working
directory**, even when the selected config file lives elsewhere. Changing a setting used by
a cached stage makes that stage recompute as usual.

### API keys and .env

TOML files must never hold secrets. A key containing `key`, `token`, `secret` or `password`
(case-insensitive) is rejected without echoing its value. For a hosted LLM, the optional
`questxtract[llm]` dependency and an API key are still required.

`Config.api_key` uses `OPENAI_API_KEY` from the environment if set. Otherwise, it reads that
one name from `.env` in the working directory. This file is ignored by Git. The supported
format is `OPENAI_API_KEY=VALUE`, with optional single or double quotes, blank lines and `#`
comments. Other variables are ignored; parent and user directories are not searched. The key
is never stored in Config, logs, cache entries or `questions.json`.

This reader does not run shell commands or expand variables. If your setup uses a password
store and a shell script to retrieve the key, load it into the environment before running
QuestXtract. That environment value wins, so the `.env` reader is not used in that case.

## Where files go

Without `-o`, output lands next to the input, named after it:

| Input | `process` writes | `transcribe` writes |
|---|---|---|
| `lecture.mp4` | `lecture.questions.json` | `lecture.transcript.json` |
| `lecture.transcript.json` | `lecture.questions.json` | (needs an `.mp4` or `.mp3`) |

`filter` prints to the terminal unless `-o` names a file.

When vision has read the board, the keyframe images are copied into a folder next to the
questions file and named after it (`lecture.keyframes/` for `lecture.questions.json`); the file
refers to them by relative path, so move or share the two together. With `-o -` nothing is
copied and the file points into the cache instead.

A second run overwrites the file. Pass `-o` to keep several versions side by side. The
questions file format is described in the [README](../README.md#output) and in
[architecture.md 12.1](architecture.md#121-questionsjson).

## Recipes

**Get the questions from a lecture video**

```bash
questxtract process lecture.mp4
```

**Do a quick test run on a laptop.** A small model transcribes several times faster, less
accurately:

```bash
questxtract process lecture.mp4 -m base
```

**Try another threshold without transcribing again.** The transcript is cached after the first
run, so the second command takes seconds:

```bash
questxtract process lecture.mp4 -t 0.4
questxtract process lecture.mp4 -t 0.3 -o lecture.questions-0.3.json
```

**Keep the transcript, to read it, share it or reuse it**

```bash
questxtract transcribe lecture.mp4                 # writes lecture.transcript.json
questxtract process lecture.transcript.json        # writes lecture.questions.json
```

If you already ran `process lecture.mp4`, `transcribe` reuses its cached transcript and finishes
almost instantly. It works the other way round too.

**Print the result instead of writing a file**

```bash
questxtract process lecture.transcript.json -o -
```

**Everything in one run: the board, the LLM detector and a quiz.** Vision reads the board on
NTNU's service with `QUESTXTRACT_VISION_API_KEY`; the judge and the quiz go to OpenAI with
`OPENAI_API_KEY`. It writes `lecture.questions.json`, the keyframes in `lecture.keyframes/` and
`lecture.quiz.json`; `qti` then packs the quiz for an LMS.

```bash
questxtract process lecture.mp4 --vision --detector llm --quiz
questxtract qti lecture.quiz.json -o lecture.qti.zip
```

**Export for an LMS (QTI 3.0 or 2.1)**

```bash
questxtract qti lecture.quiz.json -o lecture.qti.zip   # QTI 3.0, every item but the rejected ones
questxtract qti lecture.quiz.json -o lecture.qti21.zip --qti-version 2.1   # QTI 2.1, see the caveat
```

**See which questions the threshold threw away.** This is a development tool, not a CLI
command:

```bash
questxtract transcribe lecture.mp4
uv run python scripts/dropped_questions.py lecture.transcript.json -o lecture.questions.json
# writes lecture.questions.json (kept) and lecture.questions.dropped.json (dropped)
```

**Find the questions about a word, said or on the board**

```bash
questxtract filter lecture.questions.json --text eigenvalue
questxtract filter lecture.questions.json --text "transfer function" --from 5:00 --to 20:00 -o tf.json
```

**Start over** after changing the code or when you suspect a stale result:

```bash
questxtract process lecture.mp4 --from-scratch
```

## Finding questions with an LLM

`--detector llm` hands the decision to a language model. Every sentence ending in `?`, and every
question the rules found, goes to the model with the lecture around it; it decides which are real
questions, finds the sentences that answer them, and rewrites a messy question into a clean one.
The spoken version stays in the file as `original_text`, and the model can never add a question
that was not spoken: it only judges the sentences it was given, by id. The rules still run first;
their `reasons` stay on the questions the model keeps, and those gain `llm`.

It needs the `llm` extra: `pip install "questxtract[llm] @ git+https://github.com/mad201802/QuestXtract"`.

**Hosted (OpenAI).** Put the key in `OPENAI_API_KEY`, the safe way described in the README
([Setting up the API key](../README.md#setting-up-the-api-key)), never typed into a command. The
default model is `gpt-4o`, which takes no reasoning setting, so none is sent. A thinking model
such as `gpt-5.6-terra` needs a level.

```bash
questxtract process lecture.mp4 --detector llm
questxtract process lecture.mp4 --detector llm --llm-model gpt-5.6-terra --llm-reasoning low
```

**Another hosted provider (OpenRouter and the like).** Anything that speaks the OpenAI API works:
put that provider's key in `OPENAI_API_KEY` (stored the same way) and pass its URL and model name.

```bash
questxtract process lecture.mp4 --detector llm \
    --llm-url https://openrouter.ai/api/v1 --llm-model deepseek/deepseek-v4.1-flash
```

**Local (Ollama).** Any OpenAI-compatible server works (Ollama, llama.cpp's `llama-server`,
vLLM, LM Studio); pass its URL. With [Ollama](https://ollama.com), give it an 8k context first:
below 24 GB of graphics memory it defaults to 4k and silently cuts longer prompts.

```bash
OLLAMA_CONTEXT_LENGTH=8192 ollama serve        # in one terminal
ollama pull qwen3.6:35b                        # once; needs about 24 GB
questxtract process lecture.mp4 --detector llm \
    --llm-url http://localhost:11434/v1 --llm-model qwen3.6:35b
```

On a smaller machine, `qwen3.5:9b` needs to think first, which makes it slow but good:

```bash
questxtract process lecture.mp4 --detector llm --llm-reasoning low \
    --llm-url http://localhost:11434/v1 --llm-model qwen3.5:9b
```

Speed depends on the machine: without thinking, a request takes seconds; with it, minutes, and a
lecture makes 15 to 25 of them. Which model finds the questions best is measured with `eval`:

```bash
questxtract eval tests/data/ --detector llm --llm-url http://localhost:11434/v1 --llm-model qwen3.5:9b
```

**Caching.** Every request to the model is cached in `.questxtract/judge/`, keyed on what was sent.
Another `-t` or a second `eval` makes no calls; a different model, endpoint, transcript or
prompt asks again (a new questxtract version may change the prompt). `--from-scratch` ignores the cache.

**What it cost.** The `stats` block of `questions.json` counts the model calls of the run:
`judge_requests`, `judge_input_tokens`, `judge_output_tokens` and `judge_ms`, and the same with
`layout_` and `vision_` when vision is on. Tokens are what the endpoint reports (0 from a server
that reports none); a run answered from the cache shows 0. Multiply by your provider's prices
for the cost of a lecture.

**When the endpoint fails.** You asked for the model, so there is no quiet fallback to the rules:
the run stops with one `error:` line (see [When something goes wrong](#when-something-goes-wrong)).
Parts already answered stay cached, so running the same command again picks up where it stopped.

## Choosing a model

`-m` picks the Whisper model size. Smaller is faster, larger is more accurate.

| Name | Download | When to use it |
|---|---|---|
| `tiny`, `base` | ~140 MB for `base` | quick test runs, CI, checking that everything works |
| `small`, `medium` | in between | when `turbo` is too slow or too big for the machine |
| `turbo` (default) | ~1.5 GB | real lectures: close to `large-v2` quality, roughly 8x faster than `large-v3` |
| `large-v3` | largest | the most accurate, and the slowest |

Any other name faster-whisper knows also works, such as `small.en`. An unknown name fails with
the list of valid ones. The first run with a model downloads it once into
`~/.cache/huggingface`. While that happens, the progress bar sits at `Transcribing 0%`, and the
download may print its own progress bar.

## The cache

Results are cached in a `.questxtract/` folder **in the folder you run the command from**. Run
from the same folder every time to benefit from it.

| Changing this... | ...transcribes again? |
|---|---|
| `-t`, `-o`, `--detector`, `--llm-model`, `--llm-url`, `--llm-reasoning` | no |
| `-m` | yes: a different model gives a different transcript |
| `--vision-model`, `--vision-reader`, a `vision_*` setting, or `--llm-url` / `base_url` with vision on | no, but the board is read again; keyframes already read with the same model are not sent twice |
| `--from-scratch` | yes, and it recomputes everything else too |
| the video file itself | yes: the cache is keyed on the file's content |

A `.json` transcript input has no transcription to save. Only the requests to the model are
cached, with `--detector llm` (see [Finding questions with an LLM](#finding-questions-with-an-llm)).

Within one run, a slide or page that comes back with nothing changed is not read again: its
earlier reading is copied, also with `--from-scratch`, and counted in `stats.vision_reused`. This
is not the cache, so it works on a first run too ([architecture.md 6.5](architecture.md#65-captioning)).

## When something goes wrong

| Exit code | Meaning |
|---|---|
| `0` | success |
| `1` | an error, explained in one line starting with `error:` |
| `2` | the command itself was wrong: a missing argument, or `-t abc` |
| `130` | stopped with Ctrl+C |

| Message | What to do |
|---|---|
| `...: no such file` | Check the path. |
| `...: expected an .mp4 recording or an .mp3 recording or a .json transcript` | Convert the recording to MP4 or MP3 first, e.g. with ffmpeg. |
| `...: recording contains no audio track` | The file has video only; there is nothing to transcribe. |
| `...: invalid or unreadable recording` | The file is damaged, or not really an MP4 or MP3. |
| `Invalid model size '...', expected one of: ...` | Use one of the names listed in the message. |
| `...: not valid JSON` / `expected a list of segments...` | The `.json` file is not a transcript. |
| `--detector llm needs OPENAI_API_KEY in the environment or .env, or --llm-url (base_url) for a local server` | Make the key available in the environment or local `.env`, or pass `--llm-url` for a local server. |
| `<file>: invalid TOML at line N` | Correct the TOML syntax at the reported line. |
| `<file>: <key>: expected float, got string` | Use the expected type; numbers must not be quoted. |
| `<file>: unknown config field(s): ...` | Correct the name to match a field in `config.py`. |
| `<file>: <key>: secrets belong in OPENAI_API_KEY or QUESTXTRACT_VISION_API_KEY, not TOML` | Remove the secret from TOML and use the environment or `.env`. |
| `min_confidence / --threshold must be between 0 and 1` | Correct the effective threshold in the file or command. |
| `detector must be rules or llm` | Correct the detector name. |
| `.env: line N: expansion is not supported; ...` | Load the key into the environment using your shell or password store before running the command. |
| `--detector llm needs the LLM extra: ...` | Install `questxtract[llm]`. |
| `--vision / vision_enabled reads the board through NTNU's LLM service (...), so it needs QUESTXTRACT_VISION_API_KEY ...` | Make the vision key available in the environment or local `.env`, set `vision_base_url` or pass `--llm-url` for another endpoint, or turn vision off with `--no-vision`. |
| `--vision / vision_enabled needs the LLM extra: ...` | Install `questxtract[llm]`, or turn vision off with `--no-vision`. |
| `note: <file> has no video, so the board is not read` | Not an error: a recording without video, or a transcript, has no board to read. The questions are written as usual. |
| `judge: <endpoint>: ...` | The model endpoint failed: it is not running, the key is wrong, or the model name is unknown to it. The message after the endpoint says which. |
| `judge: <endpoint>: ... Unrecognized request argument supplied: reasoning_effort` | `--llm-reasoning` set a level, but the model has no reasoning setting, e.g. `gpt-4o`. Drop `--llm-reasoning`, or pass `--llm-reasoning server`. |
| `judge: <endpoint>: no answer within 120 s, twice` | The model did not answer one request in time, and not on a second try either. With a thinking model, pass a `--llm-reasoning` level, which allows 10 minutes; otherwise the provider is overloaded, try again later. |
| `judge: <model> did not return the expected JSON, twice` | The model could not follow the output format. Try a larger model; rerun to try again. |
| `judge: <model> judged none of the N candidates; ...` | The endpoint answered, but in a shape other than the one asked for, so no candidate was judged. Check that the server supports structured output (`response_format`); try another model or server. |

A missing file or wrong file type is caught before anything runs, and so is `--detector llm`
without a key or without the extra. A damaged recording or one
without audio is caught by the first stage. Either way, a bad input never costs you a
transcription.

## Not available yet

- **`--language`**: every lecture is transcribed as English. Transcribing another language would
  work, but question detection matches English words: the question words (what, why, how) and
  auxiliary verbs (is, can, does). It would find almost nothing beyond sentences ending in `?`.
  Requirement 4.2 assumes English lectures, so no flag is planned. The Python API accepts
  `Config(language=...)` or `language` in the TOML file for experiments.
- **YouTube URLs**: the CLI takes a local file. Download the video first.
