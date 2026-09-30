# Command line

How to run QuestXtract from a terminal: every command, every option, and recipes for the common
jobs. For the library API see the [README](../README.md#python-api); for how the pipeline works
inside, [architecture.md](architecture.md).

## Quick start

```bash
questxtract process lecture.mp4                # video -> lecture.questions.json
questxtract transcribe lecture.mp4             # transcription only -> lecture.transcript.json
questxtract process lecture.transcript.json    # questions from an existing transcript
```

Run `questxtract --help`, or `questxtract <command> --help`, for the same information in the
terminal. From a source checkout, put `uv run` in front: `uv run questxtract process lecture.mp4`.

## What decides how much runs

The file you pass in. There are no flags for picking stages.

| You pass | What runs |
|---|---|
| an `.mp4` recording | everything: check the recording, transcribe it, then find the questions |
| a `.json` transcript | everything after transcription (much faster: no speech recognition) |
| anything else | nothing: `error: ...: expected an .mp4 recording or a .json transcript` |

## Commands

### `questxtract process <input>`

Finds the questions asked in a recording or a transcript and writes them to a questions file.

| Option | Takes | Default | What it does |
|---|---|---|---|
| `-o`, `--output` | a path | next to the input | Where to write the questions. `-o -` prints them to the terminal instead. A folder in the path that doesn't exist yet is created. |
| `-t`, `--threshold` | a number, 0 to 1 | `0.4` | Keeps only questions with at least this confidence. Lower gives more questions, including more false ones; higher gives fewer. |
| `-m`, `--model` | a model name | `turbo` | The Whisper model used to transcribe. Only matters for `.mp4` input. See [Choosing a model](#choosing-a-model). |
| `--from-scratch` | nothing | off | Ignores cached results and recomputes everything, transcription included. |
| `--detector` | `rules` or `llm` | `rules` | How questions are found: fast offline rules, or a language model that reads the transcript. See [Finding questions with an LLM](#finding-questions-with-an-llm). |
| `--llm-model` | a model name | `gpt-5.6-terra` | The model for `--detector llm`, as the endpoint names it, e.g. `qwen3.5:9b` in Ollama. |
| `--llm-url` | a URL | OpenAI | An OpenAI-compatible endpoint for `--detector llm`, e.g. `http://localhost:11434/v1` for Ollama. Without it, OpenAI is used with the key from `OPENAI_API_KEY`. |
| `--llm-reasoning` | `none`, `low`, `medium`, `high` | `none` | How long a thinking model may think before it judges. Slower, and what small models such as `qwen3.5:9b` need to judge well. Any level but `none` allows each request 10 minutes instead of 2. |
| `-h`, `--help` | nothing | | Shows the options and examples. |

One thing to know about `-t` with the rules: a question with no answer scores at most `0.55`, so
a threshold above `0.55` keeps only questions the next sentence answers. With `--detector llm`
the confidence is the model's own, answered or not.

### `questxtract transcribe <recording>`

Transcribes an `.mp4` recording into a transcript file, without looking for questions. The file
it writes is exactly what `process` accepts as input, and what `scripts/dropped_questions.py`
reads.

| Option | Takes | Default | What it does |
|---|---|---|---|
| `-o`, `--output` | a path | next to the recording | Where to write the transcript. `-o -` prints it instead. |
| `-m`, `--model` | a model name | `turbo` | The Whisper model used to transcribe. |
| `--from-scratch` | nothing | off | Transcribes again even if a cached transcript exists. |
| `-h`, `--help` | nothing | | Shows the options and examples. |

### Everywhere

| Option | What it does |
|---|---|
| `--version` | Prints the installed version. |
| `-h`, `--help` | Without a command, lists the commands; after one, lists that command's options. |

`questxtract eval <files|dirs> [--json]` scores the pipeline against the labelled set: recall,
precision, false positives by kind, a threshold sweep, and every miss
([tests/data/README.md](../tests/data/README.md)). It takes `--detector`, `--llm-model` and
`--llm-url` too, so the rules and a model can be compared on the same recordings; the report's
first line names the detector, the model and the prompt version. `questxtract quiz <questions.json>` is
reserved for quiz generation; it is not implemented yet (#79) and exits with a one-line error.

## Where files go

Without `-o`, output lands next to the input, named after it:

| Input | `process` writes | `transcribe` writes |
|---|---|---|
| `lecture.mp4` | `lecture.questions.json` | `lecture.transcript.json` |
| `lecture.transcript.json` | `lecture.questions.json` | (needs an `.mp4`) |

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

**See which questions the threshold threw away.** This is a development tool, not a CLI
command:

```bash
questxtract transcribe lecture.mp4
uv run python scripts/dropped_questions.py lecture.transcript.json -o lecture.questions.json
# writes lecture.questions.json (kept) and lecture.questions.dropped.json (dropped)
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

**Hosted (OpenAI).** Put the key in the environment; the default model is `gpt-5.6-terra`, and
`gpt-5.6-luna` is cheaper. A 90-minute lecture costs cents.

```bash
export OPENAI_API_KEY=sk-...
questxtract process lecture.mp4 --detector llm
questxtract process lecture.mp4 --detector llm --llm-model gpt-5.6-luna
```

**Another hosted provider (OpenRouter and the like).** Anything that speaks the OpenAI API works:
put that provider's key in `OPENAI_API_KEY` and pass its URL and model name.

```bash
export OPENAI_API_KEY=sk-or-...                # the OpenRouter key
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
| `--from-scratch` | yes, and it recomputes everything else too |
| the video file itself | yes: the cache is keyed on the file's content |

A `.json` transcript input never touches the cache. There is nothing expensive left to save.

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
| `...: expected an .mp4 recording or a .json transcript` | Convert the video to MP4 first, e.g. with ffmpeg. |
| `...: recording contains no audio track` | The file has video only; there is nothing to transcribe. |
| `...: invalid or unreadable MP4 recording` | The file is damaged or not really an MP4. |
| `Invalid model size '...', expected one of: ...` | Use one of the names listed in the message. |
| `...: not valid JSON` / `expected a list of segments...` | The `.json` file is not a transcript. |
| `--detector llm needs OPENAI_API_KEY, or --llm-url for a local server` | Set the key, or pass `--llm-url` for a local server. |
| `--detector llm needs the LLM extra: ...` | Install `questxtract[llm]`. |
| `judge: <endpoint>: ...` | The model endpoint failed: it is not running, the key is wrong, or the model name is unknown to it. The message after the endpoint says which. |
| `judge: <endpoint>: no answer within 120 s, twice` | The model did not answer one request in time, and not on a second try either. With a thinking model, pass `--llm-reasoning`, which allows 10 minutes; otherwise the provider is overloaded, try again later. |
| `judge: <model> did not return the expected JSON, twice` | The model could not follow the output format. Try a larger model; rerun to try again. |

A missing file or wrong file type is caught before anything runs, and so is `--detector llm`
without a key or without the extra. A damaged recording or one
without audio is caught by the first stage. Either way, a bad input never costs you a
transcription.

## Not available yet

- **`--language`**: every lecture is transcribed as English. Transcribing another language would
  work, but question detection matches English words: the question words (what, why, how) and
  auxiliary verbs (is, can, does). It would find almost nothing beyond sentences ending in `?`.
  Requirement 4.2 assumes English lectures, so no flag is planned. The Python API accepts
  `Config(language=...)` for experiments.
- **YouTube URLs**: the CLI takes a local file. Download the video first.
