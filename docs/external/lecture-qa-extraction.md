# The QA Extraction Pipeline

A technical reference for how this repository turns a lecture recording into
structured question-answer candidates. The repo link is: https://github.com/matteopogetta/lecture-qa-extraction

## How to read this document

Each stage is described twice. The prose explains *what* the stage does and
*why* it exists. The **Reference** block that follows it names the files,
functions, configuration fields and constants an engineer needs in order to
change the behaviour. If you only want the method, read the prose. If you are
about to edit the code, read both.

Points where the implementation diverges from what the surrounding
documentation or the code's own naming implies are marked **Known gap**. They
are descriptions of the current state, not proposals.

Snapshot: commit `985c6f1`. Schema version `0.6.0`. Related documents:
[architecture.md](architecture.md) for package layout,
[quality_evaluation.md](quality_evaluation.md) for the evaluation protocol,
[schema_notes.md](schema_notes.md) for the output contract.

---

## 1. Overview

### 1.1 What the pipeline is trying to do

The input is a lecture recording. The desired output is a list of
question-answer pairs that actually occurred in the lecture, each one grounded
in the audio: it must be possible to point at the exact seconds where the
question was asked, the exact seconds where it was answered, and the exact
transcript spans both came from.

The central difficulty is that a lecture is not a dialogue corpus. Most
interrogative-sounding utterances in a lecture are not questions in the sense we
want: they are rhetorical framing ("what happens if we double this?"), classroom
check-ins ("can you all hear me?"), polls ("one or two?"), or discourse tags
("right?"). Of the ones that are real questions, many are answered several
sentences later, by the same speaker, after a digression. And the raw material
is an ASR transcript, which has no reliable sentence boundaries, no punctuation
guarantees, and only approximate speaker information.

The pipeline therefore does most of its work *before* QA extraction: it rebuilds
sentence structure from word timings, attributes speakers, and groups sentences
into topical segments. QA extraction then operates on a clean, timed,
speaker-annotated sentence stream, and spends most of its logic on rejecting the
false positives listed above.

### 1.2 Stage order

`LectureProcessingPipeline.process()` runs these timed stages in order. The
names are the literal keys used in the timing summary and in every exported
report.

```
                    media file (audio or video)
                              |
  1 session_loading           |  validate path, classify media type
                              v
  2 audio_normalization       |  ffmpeg -> 16 kHz mono PCM WAV
                              v
  3 transcription             |  faster-whisper -> TranscriptChunk[]
                              v
  4 alignment                 |  WhisperX -> AlignedWord[] (word timings)
                              v
  5 utterance_building        |  gap-split words -> Utterance[]
                              v
  6 diarization               |  pyannote -> DiarizationSegment[]   (off by default)
                              v
  7 speaker_attribution       |  overlap voting -> Utterance.speaker_id
                              v
  8 sentence_reconstruction   |  wtpsplit SaT -> Sentence[]   <-- the QA input layer
                              v
  9 transcript_post_processing|  merge + normalize -> MergedTranscript  (fallback layer)
                              v
 10 transcript_segmentation   |  Sentence[] -> Segment[]
                              v
 11 qa_extraction             |  Sentence[] + Segment[] -> QAPairCandidate[]
                              v
 12 qa_speaker_check          |  ECAPA embeddings -> confidence adjustment
                              v
 13 json_export               |  session.to_dict() -> <name>_<mode>.json
    (+ debug_excel_export, ai_review_packet_export, evaluation_run_export)
```

Two structural facts about this order matter downstream:

**Sentences are built before the merged transcript.** Stage 8 consumes
utterances, not the merged transcript produced in stage 9. The merged transcript
exists for the human-readable full text and as a *degraded fallback* for QA
extraction when the sentence layer is unavailable.

**QA extraction can run more than once per session.** When
`segmentation_mode` is `both`, `process_segmentation_modes` deep-copies the
session once per mode (`structural`, `windowed`, `adaptive`), re-segments each
copy, and re-runs QA extraction on it. Each mode produces its own JSON file.
Segment membership changes QA scores, so the candidate sets differ between
modes; this is the intended way to compare segmentation strategies.

> **Reference**
>
> - Orchestrator: [_processing_pipeline_impl.py:427](../src/lecture_analyzer/core/_processing_pipeline_impl.py#L427) (`process`)
> - Stages 4-9 are chained inside `transcribe()` at [:162](../src/lecture_analyzer/core/_processing_pipeline_impl.py#L162)
> - Mode fan-out: `process_segmentation_modes` at [:638](../src/lecture_analyzer/core/_processing_pipeline_impl.py#L638); mode list resolved by `_configured_segmentation_modes` at [:654](../src/lecture_analyzer/core/_processing_pipeline_impl.py#L654)
> - Stage status vocabulary: `_normalize_stage_status` at [:914](../src/lecture_analyzer/core/_processing_pipeline_impl.py#L914) - `executed`, `executed_forced`, `reused_from_cache`, `reused_from_artifact`, `partial`, `skipped`, `disabled`, `failed`
> - Data models: [_models_impl.py](../src/lecture_analyzer/core/_models_impl.py), all `@dataclass(slots=True)`; session root `LectureSession` at [:686](../src/lecture_analyzer/core/_models_impl.py#L686)

**Known gap.** The QA stage reports its status from the presence of
`session.merged_transcript` (`_resolve_qa_stage_status`,
[:970](../src/lecture_analyzer/core/_processing_pipeline_impl.py#L970)), but the
extractor consumes `session.sentences` first. A run with sentences and an empty
merged transcript is reported as `skipped` while having produced candidates, and
the inverse is also possible. The status field and the candidate list can
disagree.

### 1.3 Profiles

A profile is a named bundle of configuration overrides. It is the only
high-level knob most users touch.

| Profile | Alignment | Diarization | Answer search | Answer ranking | Speaker check | Excel |
|---|---|---|---|---|---|---|
| `current` | on | off | semantic | semantic | auto | on |
| `light` | **off** | off | rule-based | rule-based | auto | off |
| `quality` | on | off | semantic | semantic | auto | off |
| `quality_local` | on | off | **rule-based** | **rule-based** | **on if model present** | off |
| `full` | on | **on** | semantic | semantic | auto | on |
| `diagnostic` | on | on | semantic | semantic | auto | on + review packet, `segmentation_mode=both` |

`current` applies no overrides at all, so it inherits the dataclass defaults,
which enable both semantic layers. `quality_local` is the profile used for the
frozen benchmark in [quality_evaluation.md](quality_evaluation.md): it turns the
semantic layers *off* and instead enables the acoustic speaker check plus the
full set of aggressive emission gates (see section 5.8).

Precedence: profile defaults are applied first, explicit overrides second.
`apply_overrides` pops `pipeline_profile`, re-applies that profile's defaults,
then sets each explicit override on top, so a CLI flag always beats a profile.

> **Reference**
>
> - `SUPPORTED_PIPELINE_PROFILES` and `PIPELINE_PROFILE_SETTINGS`: [_pipeline_config_impl.py:15](../src/lecture_analyzer/core/_pipeline_config_impl.py#L15)
> - Precedence: `apply_overrides` at [:366](../src/lecture_analyzer/core/_pipeline_config_impl.py#L366); unknown field names raise `AttributeError`
> - Profile aliases `default`, `compat`, `compatibility` all map to `current` (`_normalize_pipeline_profile`)

**Known gap.** The dataclass defaults enable `semantic_retrieval` and
`semantic_reranker`, while the profile documented as recommended
(`quality_local`) disables both. Running without `--pipeline-profile` therefore
exercises a different QA path than the one the benchmark numbers describe.

---

## 2. Front end: from media to timed words

These stages exist to produce one thing that QA extraction cannot work without:
**word-level timings**. Everything else the front end produces is either
derived from those timings or is a fallback for when they are missing.

### 2.1 Session loading

Validates the input paths, classifies each as audio or video by extension, and
creates the `LectureSession` that every later stage mutates. Multiple inputs are
supported and are given an `order_index`; each input's audio is later offset by
`session_offset_seconds` so the whole session shares one clock.

> **Reference**
>
> - [_session_loader_impl.py](../src/lecture_analyzer/input/_session_loader_impl.py); extensions from `audio_extensions` / `video_extensions` config tuples
> - Every timed model carries both local (`start_seconds`) and session-absolute (`session_start_seconds`) times

### 2.2 Audio normalization

Converts any input to the single audio geometry the rest of the pipeline
assumes: 16 kHz, mono, 16-bit PCM WAV. Whisper expects 16 kHz; the speaker
attribution code reads raw PCM frames with the `wave` module and supports only
a handful of sample widths; the ECAPA speaker check slices this same file.

The result is cached next to the artifacts directory with a
`.metadata.json` sidecar recording the source path, source mtime, format,
sample rate, channels and bit depth. A rerun regenerates only when one of those
differs. After conversion, `ffprobe` verifies the output actually matches the
requested contract and raises `AudioValidationError` if not.

```
ffmpeg -y -i <source> -vn -ac 1 -ar 16000 -sample_fmt s16 -c:a pcm_s16le <out>
```

> **Reference**
>
> - Command construction: [_audio_normalizer_impl.py:189](../src/lecture_analyzer/preprocessing/_audio_normalizer_impl.py#L189); codec from `_resolve_audio_codec` (`pcm_s{bit_depth}le`, or `flac`)
> - Verification: `ffprobe -v error -select_streams a:0 -show_entries stream=... -of json`, checked by `_assert_probe_matches_contract`
> - Reuse predicate: `_metadata_matches`; output name `{order:03d}_{stem}_mono_{rate}hz.wav`
> - Config: `normalized_audio_format` (`wav`|`flac`), `normalized_audio_sample_rate` 16000, `normalized_audio_channels` 1, `normalized_audio_bit_depth` 16, `overwrite_normalized_audio` False

**Known gap.** Despite the stage name, there is no loudness normalization - no
`loudnorm`, no `dynaudnorm`, no `-af` filter of any kind. "Normalization" here
means format and geometry only. Downstream, the audio quality analyzer
compensates by measuring RMS *relative to the whole file* rather than against an
absolute level. The output filename also hardcodes `_mono_` regardless of the
configured channel count.

### 2.3 Transcription

A single backend: `faster-whisper`, model `small`, beam size 5. Language is
either auto-detected per source or fixed by configuration. Per-segment decoder
diagnostics (`avg_logprob`, `no_speech_probability`, `compression_ratio`) are
preserved in chunk metadata, and the detected language plus its probability are
recorded at source level.

```python
model.transcribe(str(audio_path), beam_size=5, language=language, task="transcribe")
```

> **Reference**
>
> - [_backend_impl.py:54](../src/lecture_analyzer/transcription/_backend_impl.py#L54) (`FasterWhisperBackend`); `build_transcription_backend` rejects any other name
> - `compute_type` is passed to `WhisperModel` only when it is not `"auto"`
> - Language resolution `_resolve_language`: mode `auto` -> `None` (Whisper detects); mode `fixed` -> lowercased `transcription_language`, `ValueError` if unset
> - Chunk ids: `{audio_source_id}_chunk_{index:04d}`; empty-text segments dropped; timestamps clamped to `start >= 0`, `end >= start`
> - This is the only hard-failure path in the whole pipeline: a missing faster-whisper raises `BackendInitializationError` rather than degrading

**Known gap.** No VAD is configured anywhere - `vad_filter` and `vad_parameters`
are never passed, so faster-whisper's classic default (VAD off) applies. And
`word_timestamps=True` is never requested, which is why word timings must come
from a separate alignment stage.

### 2.4 Alignment

WhisperX forced alignment turns each transcript chunk into a sequence of
`AlignedWord` objects with individual start and end times and per-word
confidence. This is the **sole source of word timings** in the pipeline, and
therefore the sole basis for utterances and sentences.

The alignment language is resolved by majority vote over the chunks' detected
languages, falling back to the source-level detected language and then to the
configured language.

> **Reference**
>
> - [_whisperx_aligner_impl.py:241](../src/lecture_analyzer/transcription/_whisperx_aligner_impl.py#L241); `load_align_model(language_code=, device=)` with `model_name=` only when `transcript_alignment_model_name` is set, and a `TypeError` retry that drops it
> - Models cached per `(language, device, model_name)`
> - Language vote: `_resolve_alignment_language`, raises `MissingAlignmentLanguageError` when nothing resolves
> - Word ids: `{audio_source_id}_aligned_word_{seg:04d}_{word:04d}`
> - Failure is per-source and non-fatal: `align_session` catches `AlignmentError` and bare `Exception`, marks that source `failed` with a reason, and continues
> - Config: `transcript_alignment_enabled` True, `transcript_alignment_model_name` None (whisperx picks its per-language wav2vec2 default), `transcript_alignment_device` `cpu`

Disabling alignment (`--disable-alignment`) removes word timings, which removes
utterances, which removes sentences - and QA extraction silently falls back to
the merged transcript layer. This is the single most consequential flag in the
pipeline.

### 2.5 Utterance building

An utterance is a run of consecutive words within one aligned segment, split
wherever the silence between two words exceeds 0.85 seconds. Aligned-segment
boundaries are always hard boundaries; utterances never span them.

Utterances are the pipeline's atomic timed unit. Sentences are built by grouping
them, never by cutting them, which is what guarantees that every sentence has
exact start and end times without any interpolation.

Utterance ids are content-addressed: a SHA-1 over the segment index, word index
range, rounded timings and text. Identical input therefore produces identical
ids across runs, which is what makes the artifact caches and the Excel
human-comment round-trip work.

> **Reference**
>
> - [_utterance_builder_impl.py:339](../src/lecture_analyzer/analysis/_utterance_builder_impl.py#L339) (`_should_split_before_word`); missing timings never split
> - Id: `_build_utterance_id` -> `{audio_source_id}_utterance_{sha1[:16]}`
> - Text joining tightens spacing before punctuation and inside brackets (`_join_word_texts`)
> - When a segment yields no word-based utterance, `_build_segment_fallback_utterance` emits one coarse utterance tagged `build_strategy: "segment_fallback"`, `timing_source: "aligned_segment"`
> - Config: `utterance_max_gap_seconds` 0.85
> - Without aligned transcripts the stage sets `utterance_build_status="skipped"`, reason `aligned_transcripts_unavailable`

### 2.6 Diarization

pyannote `speaker-diarization-3.1`, **off by default**. When enabled it produces
`DiarizationSegment` spans with anonymous speaker ids, re-aliased deterministically
to `SPEAKER_00`, `SPEAKER_01`, ... in order of first appearance.

Two workarounds are worth knowing about: PCM WAV input is preloaded into a
`{"waveform": tensor, "sample_rate": int}` dict rather than passed as a path, to
avoid the torchcodec decoder path; and a torch 2.6 `weights_only` load failure
triggers one retry with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.

> **Reference**
>
> - [_pyannote_diarizer_impl.py](../src/lecture_analyzer/transcription/_pyannote_diarizer_impl.py); `_load_waveform_input`, `_is_weights_only_load_error`, `_force_legacy_torch_load`
> - Speaker count hints: `diarization_num_speakers` wins outright; otherwise `min_speakers`/`max_speakers` are passed if set
> - Token resolution order: `diarization_auth_token` -> `HUGGINGFACE_HUB_TOKEN` -> `HF_TOKEN`
> - Track preference: `exclusive_speaker_diarization` when `diarization_prefer_exclusive`, else `speaker_diarization`
> - Same per-source catch-and-continue failure pattern as alignment

### 2.7 Speaker attribution

Assigns a diarization speaker to each utterance by temporal overlap voting.
For each utterance, overlap seconds are accumulated per speaker; the top speaker
wins only if it covers at least 60% of the utterance and the runner-up is not
too close.

```
top_overlap_ratio = top_overlap_seconds / utterance_duration

top_overlap_ratio < 0.6                  -> unassigned / low_overlap
runner_up / top   >= 0.85                -> unassigned / ambiguous_overlap
otherwise                                -> assigned to top speaker

confidence = 0.55*coverage + 0.25*margin + 0.20*evidence
             (x 0.7 when the audio window is degraded)
  coverage = clamp(top_overlap_ratio)
  margin   = 1 - runner_up/top
  evidence = min(1, top_overlap_seconds / duration)
```

A three-pass smoothing stage then repairs the characteristic diarization
artefacts: short unassigned gaps between two runs of the same speaker are
bridged; A-B-A sandwiches where B is too short or too weakly evidenced collapse
to A; and unstable runs containing degraded audio inherit the previous speaker.
Every reassignment multiplies the confidence by 0.7 and may mark the utterance
uncertain.

> **Reference**
>
> - [_speaker_attribution_impl.py:159](../src/lecture_analyzer/analysis/_speaker_attribution_impl.py#L159) (`attribute_utterance`); confidence at [:411](../src/lecture_analyzer/analysis/_speaker_attribution_impl.py#L411)
> - Smoothing: [_speaker_stability_impl.py:88](../src/lecture_analyzer/analysis/_speaker_stability_impl.py#L88) (`smooth_assignments`) - `_bridge_short_gaps`, `_smooth_short_switches`, `_preserve_previous_speaker`, `_mark_residual_uncertainty`
> - Instability test `_is_unstable_run`: duration < 1.25 s, **or** evidence < 1.0 s, **or** fully degraded and duration <= 1.75 s
> - Config: `speaker_attribution_min_overlap_ratio` 0.6, `speaker_attribution_ambiguity_ratio` 0.85, `speaker_switch_min_duration_seconds` 1.25, `speaker_switch_min_stable_evidence_seconds` 1.0, `speaker_switch_short_gap_merge_seconds` 0.5
> - Full audit trail per utterance in `metadata["speaker_attribution"]`

Because diarization is off by default, the default path through this stage marks
every utterance unassigned with reason `diarization_unavailable`. Speaker
information then only reaches QA extraction through the separate acoustic check
in section 5.10.

### 2.8 Audio quality

A deliberately minimal, dependency-free quality probe used to discount speaker
attribution confidence on bad audio. Two metrics, computed with `wave`, `array`
and `math` only:

- `rms_ratio` = window RMS / whole-file reference RMS, degraded below **0.35**
- `zero_crossing_rate` = sign changes / (N-1), degraded above **0.22**

Its only numeric effect anywhere is the `x 0.7` factor on speaker attribution
confidence.

> **Reference**
>
> - [_audio_quality_impl.py](../src/lecture_analyzer/analysis/_audio_quality_impl.py); reference RMS cached per resolved path
> - Sample widths 1, 2 and 4 bytes supported; anything else reports `audio_features_unavailable`
> - Multichannel input is strided to channel 0, not downmixed
> - Config: `speaker_attribution_low_energy_ratio_threshold` 0.35, `speaker_attribution_high_zero_crossing_rate_threshold` 0.22; either may be `None` to disable that check

### 2.9 Artifact caching

Five independent caches, each with its own version counter and its own
invalidation predicate. Cache identity is the **original media path**, not the
normalized WAV, so re-normalizing does not invalidate a transcript.

| Layer | File | Invalidated by |
|---|---|---|
| Transcription | `<stem>.txt` + `<stem>.transcription.json` | version only (see gap below) |
| Alignment | `<stem>.alignment.json` | version, `{enabled, model, device}`, transcript digest |
| Diarization | `<stem>.diarization.json` | version, all 7 diarization settings, audio size+mtime |
| Utterances | `<stem>.utterances.json` | version, `utterance_max_gap_seconds`, aligned-word digest |
| Sentences | `<stem>.sentences.json` | version, all 5 sentence settings, utterance digest |

> **Reference**
>
> - [_cache_store_impl.py](../src/lecture_analyzer/transcription/_cache_store_impl.py); directory key `{stem}_{sha1(str(parent))[:10]}`
> - `force_recompute` does not delete artifacts; it bypasses reuse and logs the skip
> - Transcription defaults to sidecars next to the source when `transcription_cache_directory` is unset

**Known gap.** `transcription_cache_require_backend_match` defaults to `False`,
so the transcription cache is the only layer that does **not** verify its
configuration snapshot: a warm run can silently reuse a transcript produced by a
different Whisper model. Additionally,
`transcription_cache_allow_text_fallback` defaults to `True`, so a missing or
incompatible manifest falls back to loading the plain `.txt` into one synthetic
chunk with `timing_source: "synthetic"` - discarding all chunk timing while the
run still appears to succeed.

---

## 3. Sentence reconstruction

### 3.1 Why this stage exists

ASR segments are not sentences. Whisper emits chunks bounded by its own decoding
window and by silence, so a single chunk routinely contains two questions, or
half of one. QA extraction needs the opposite: units that are *semantically
autonomous* (a question that can be read on its own), *precisely timed*, and
*attributed to one speaker*. This stage manufactures them.

The design constraint that shapes everything else is traceability. A sentence
must be able to name the exact utterances it came from, and its timings must be
real word timings rather than interpolations. The solution is to **never cut an
utterance**: sentence boundaries are always snapped to utterance boundaries, so a
sentence's time range is simply `min(start)` to `max(end)` over its utterances,
and its provenance is the list of their ids.

### 3.2 Algorithm

**1. Group.** Utterances are grouped into splitting units. A group breaks when
the gap exceeds 2.5 seconds, or (when
`sentence_reconstruction_respect_speaker_boundaries` is set) at a *strong speaker
boundary* - both utterances have non-empty, non-uncertain, differing speaker ids.

**2. Render with an offset table.** The group's utterance texts are joined with
single spaces while recording the cumulative character offset at which each
utterance ends. This offset table is the bridge between the text world and the
timing world.

**3. Split.** The joined text goes to `wtpsplit`'s `SaT` model `sat-3l-sm`, a
neural sentence segmenter chosen because it works on unpunctuated multilingual
text - which is exactly what a lecture transcript is. The splitter is resolved
lazily and memoized. `_call_splitter` probes four call signatures in order
(`lang_code=`, `language_code=`, `language=`, positional) on `split`, then the
same four on the object itself, to tolerate API drift across wtpsplit versions.

**4. Snap back to utterances.** For each fragment except the last,
`text.find(fragment, cursor)` gives its end offset; `_closest_boundary_index`
picks the utterance whose cumulative end offset is nearest, constrained to be
strictly greater than the previous boundary. The final boundary is forced to the
end of the group. The result is a list of half-open utterance-index spans.

**5. Refine.** Three passes run in order:
`merge_low_quality -> split_run_on -> merge_low_quality`.

The merge pass folds a span into its neighbour when it is a fragment candidate or
ends on an incomplete marker (a dangling conjunction, preposition or
determiner), provided no strong speaker boundary sits between them and the merged
result stays under 30 words / 15 seconds.

The split pass attacks over-long spans by scoring every internal boundary:

```
+3.00  previous utterance ends with . ? !
+1.25  previous utterance ends with , ; : - ( [ {
+2.00  gap >= 0.75 s
+1.00  gap >= 0.35 s
+1.00  next utterance starts with a transition marker
-1.50  next utterance starts with an incomplete marker
-2.00  previous utterance ends with an incomplete marker
-0.50  strong speaker boundary across the candidate
```

The best boundary is accepted only if it scores at least **1.0** and both halves
are non-fragments; the pass then recurses on each half.

**6. Materialize and consolidate.** Each span becomes a `Sentence` with id
`{audio_source_id}_sentence_{index:04d}`, timings from its utterances, and a
consolidation pass that computes labels, buckets and speaker resolution.

> **Reference**
>
> - [_sentence_reconstruction_impl.py](../src/lecture_analyzer/analysis/_sentence_reconstruction_impl.py): `_group_utterances`, `_render_group`, `_split_text`, `_resolve_boundary_indexes` [:1593](../src/lecture_analyzer/analysis/_sentence_reconstruction_impl.py#L1593), `_refine_spans` [:1671](../src/lecture_analyzer/analysis/_sentence_reconstruction_impl.py#L1671), `_best_run_on_boundary` [:1866](../src/lecture_analyzer/analysis/_sentence_reconstruction_impl.py#L1866)
> - Backend selection: `sentence_splitter_backend` is `wtpsplit` or `fallback_rules`; the latter splits on `(?<=[.!?;:])\s+`
> - A wtpsplit failure when `wtpsplit` was explicitly requested **aborts the whole run** rather than degrading, since silently switching to regex splitting would invalidate every downstream quality label
> - Config: `sentence_reconstruction_max_gap_seconds` 2.5, `sentence_splitter_model_name` `sat-3l-sm`, merge caps `sentence_merge_max_word_count` 30 / `sentence_merge_max_duration_seconds` 15.0, run-on triggers `sentence_run_on_max_word_count` 24 / `sentence_run_on_max_duration_seconds` 12.0, fragment triggers 3 words / 20 chars / 2.0 s. These are clamped into a monotone ladder at load time (`run_on >= fragment+1`, `merge >= run_on`)

### 3.3 Quality labels and cleanup scores

Every sentence is labelled and scored. These values are not cosmetic: QA
extraction reads them directly when scoring questions (section 5.3).

Signals are collected first - fragment signals (`very_short_word_count` <= 3,
`very_short_text` <= 20 chars, `very_short_duration` <= 2.0 s,
`starts_with_incomplete_marker`, `ends_with_incomplete_marker`,
`missing_strong_final_punctuation`) and run-on signals (`high_word_count` >= 24,
`high_duration` >= 12.0 s, `multiple_internal_clauses` >= 2 of `,;:`).

The label follows:

```
fragment    if starts/ends with an incomplete marker,
            or (very short word count AND no strong final punctuation)
run_on      else if any run-on signal
borderline  else if any fragment signal
good        otherwise
```

Three continuous scores are then derived from a fixed baseline:

```
autonomy   = 0.78     boundary_confidence = 0.76     continuation_risk = 0.08

label fragment:    -0.34 / -0.18 / +0.24
label run_on:      -0.22 / -0.24 / +0.20
label borderline:  -0.12 / -0.10 / +0.10
plus per-signal adjustments, +0.08 boundary confidence for strong final
punctuation, and +0.06 autonomy for 4 <= word_count <= 23; all clamped to [0,1]
```

A sentence scoring below 0.45 on autonomy or boundary confidence gains the
review flag `low_sentence_autonomy` or `low_boundary_confidence`.

Alongside these, each sentence gets a `merge_safety_label`
(`safe` / `borderline` / `risky`), length and duration buckets, a
`review_priority`, and a list of `sentence_review_flags`.

> **Reference**
>
> - `_semantic_fragment_signals` [:2172](../src/lecture_analyzer/analysis/_sentence_reconstruction_impl.py#L2172), `_semantic_run_on_signals` [:2195](../src/lecture_analyzer/analysis/_sentence_reconstruction_impl.py#L2195), `_sentence_semantic_cleanup_features` [:2213](../src/lecture_analyzer/analysis/_sentence_reconstruction_impl.py#L2213), `_semantic_quality_label` [:2274](../src/lecture_analyzer/analysis/_sentence_reconstruction_impl.py#L2274)
> - Incomplete markers: `sentence_semantic_incomplete_markers`, a 45-item EN+IT tuple of conjunctions, prepositions and determiners

### 3.4 Sentence-level speaker resolution

An utterance has at most one speaker; a sentence may span several utterances with
conflicting speakers. Resolution is a weighted vote in which each utterance
contributes:

```
base      1.00 stable | 0.65 uncertain | 0.50 unassigned (counted as noise)
        + 0.35 * min(duration / 1.0 s, 1)
        + 0.25 * min(word_count / 4, 1)
        x 0.50 if a short fragment (<= 0.45 s or <= 2 words)
        x 1.08 if it is the closing utterance of the sentence
```

The dominant speaker is assigned directly when it holds at least **58%** of the
weight *and* leads the runner-up by a margin of at least **0.45**, with guards
against unassigned noise dominating. The resulting status is `stable`,
`mostly_stable`, `uncertain`, `mixed`, or `unassigned`.

Two recovery passes then attempt to resolve what remains unassigned: an internal
majority rule with relaxed thresholds (share >= 0.52, margin >= 0.3), and a
neighbour-consensus rule that lets a very short sentence (<= 2.25 s, <= 6 words)
inherit the speaker of stable neighbours within 1.0 second.

> **Reference**
>
> - `_speaker_contribution_score` [:729](../src/lecture_analyzer/analysis/_sentence_reconstruction_impl.py#L729), `_resolve_direct_sentence_speaker` [:832](../src/lecture_analyzer/analysis/_sentence_reconstruction_impl.py#L832), `_recover_unassigned_sentence_speakers` [:915](../src/lecture_analyzer/analysis/_sentence_reconstruction_impl.py#L915)
> - Config: `sentence_speaker_dominant_share_threshold` 0.58, `sentence_speaker_dominance_margin_threshold` 0.45, plus the weighting constants above
> - Exported on the sentence as `speaker_resolution_status`, `speaker_confidence_label`, `speaker_stability_label`, `speaker_assignment_method`, `speaker_evidence_summary`

### 3.5 The provenance invariant

The traceability claim rests on one invariant: **every utterance belongs to
exactly one sentence, and every sentence names a non-empty, duplicate-free set of
utterances**. The merge, split and recovery passes can in principle violate it,
so it is checked rather than assumed.

`validate_sentence_structure` builds the forward and reverse maps and counts
duplicate source ids within a sentence, unknown references, sentences with empty
provenance but non-empty text, utterances claimed by more than one sentence, and
orphan utterances. The first four sum into `provenance_anomaly_count`; orphans
are reported but not counted as anomalies.

> **Reference**
>
> - Analysis-side: [_sentence_provenance_impl.py](../src/lecture_analyzer/analysis/_sentence_provenance_impl.py), consumed by `_provenance_diagnostics` into `SentenceCollection.metadata` and aggregated into `session.metadata["sentence_provenance_anomaly_count"]`
> - Output-side: [_sentence_provenance_validation_impl.py](../src/lecture_analyzer/output/_sentence_provenance_validation_impl.py) revalidates the serialized dicts; keys are source-scoped as `{audio_source_id}::{utterance_id}`
> - The flag `all_sentences_have_provenance_overlap` is a smell test: if true, the mapping logic is broken rather than merely imperfect

---

## 4. Merged transcript and segmentation

### 4.1 Transcript post-processing

Chunks from all audio sources are ordered (by source order, then start time,
then stable tie-breakers) and merged into a `MergedTranscript` of
`MergedTranscriptUnit`s, then lightly normalized: whitespace collapsed, spacing
tightened before punctuation and inside brackets, a space forced after `,;:!?`
when followed by a letter. Raw text is preserved alongside.

The merger deliberately **reports anomalies rather than repairing them**:
duplicate chunk ids are kept and flagged with a `chunk_occurrence` index,
overlapping chunks get an `ordering_warning` and are listed in
`overlapping_chunk_ids`, invalid timings are dropped into `dropped_chunk_ids`.

This layer serves two purposes: the human-readable `transcript_text`, and the
degraded fallback input for segmentation and QA extraction when sentences are
unavailable.

> **Reference**
>
> - [_transcript_merger_impl.py](../src/lecture_analyzer/transcription/_transcript_merger_impl.py), [_transcript_normalizer_impl.py](../src/lecture_analyzer/transcription/_transcript_normalizer_impl.py)
> - Config: `transcript_normalize_whitespace`, `transcript_normalize_punctuation_spacing`, `transcript_drop_empty_chunks`, `transcript_keep_raw_text`

### 4.2 What a segment is

A segment is a contiguous run of sentences that belong together topically -
roughly, one stretch of lecture before the speaker moves on. Segments do not
gate QA pairing; they *modulate* it. A question and its answer in the same
segment score +0.10, in adjacent segments +0.05, in distant segments -0.05.
Only when `allow_cross_segment_answer` is explicitly disabled does segment
membership become a hard constraint.

The segment also aggregates provenance: the union of its sentences' chunk ids,
merged unit ids, sentence ids, utterance ids, audio source ids, observed
languages and speaker labels.

Input preference mirrors QA extraction: `session.sentences` gives layer
`sentences`; falling back to merged units gives layer
`merged_transcript_fallback` with reason
`sentences_unavailable_using_merged_transcript`.

### 4.3 The three modes

**Structural** (default) follows the transcript's own rhythm. It closes a
segment on a hard boundary - audio source change, duration over 90 s, or text
over 800 characters - and otherwise on a silence gap over 2.0 s.

The interesting part is the *soft-gap override*. A long pause in the middle of a
sentence-in-progress should not end a segment. So when the gap exceeds the hard
threshold but stays under `segmentation_soft_max_gap_seconds` (7.0 s),
continuity signals are collected: whether the current text looks incomplete or
ends on a continuation marker (these are *required*), plus supporting signals
(current text short, next unit short, next unit starts lowercase, next unit
starts with a continuation marker). At least one required signal and at least
two signals total keep the segment open.

**Windowed** ignores content entirely: fixed 60-second windows with a
configurable overlap, each containing at least one unit. It exists as a
content-blind control for evaluating the other two.

**Adaptive** extends a segment until it reaches a target (55 s or 650 chars),
then looks back over the last 4 unit boundaries and picks the best-scoring one:

```
+2.00  segment text ends with . ? !
+0.50  segment text ends with , ; : - ( [ {
+0.75  segment text is strong enough to stand alone   (-0.50 if not)
+1.00  segment text looks complete                    (-1.00 if not)
-1.25  segment text ends with a continuation marker
-0.50  trailing unit is short
+0.50  gap before next unit >= 0.35 s
-0.25  gap before next unit <= 0.15 s
+1.00  next unit starts with a transition marker
-1.00  next unit starts with a continuation marker
-0.35  next unit starts lowercase
```

A boundary is accepted at **1.5** or above. If nothing scores well enough and a
hard limit (85 s or 950 chars) is reached, the segment closes anyway with
`adaptive_boundary_source="hard_limit_fallback"`.

> **Reference**
>
> - [_segmenter_impl.py](../src/lecture_analyzer/analysis/_segmenter_impl.py): `_segment_structural` [:341](../src/lecture_analyzer/analysis/_segmenter_impl.py#L341), `_segment_windowed` [:389](../src/lecture_analyzer/analysis/_segmenter_impl.py#L389), `_segment_adaptive` [:455](../src/lecture_analyzer/analysis/_segmenter_impl.py#L455), `_adaptive_candidate_score` [:755](../src/lecture_analyzer/analysis/_segmenter_impl.py#L755), `_continuity_signals` [:845](../src/lecture_analyzer/analysis/_segmenter_impl.py#L845)
> - Closing reasons: `max_gap`, `max_duration`, `max_text_length`, `source_change`, `window_end`, `adaptive_boundary`, `adaptive_hard_max_*`, `end_of_transcript`
> - Transition markers (13, EN+IT): `now, so, therefore, then, next, okay, well, allora, quindi, ora, bene, adesso, dunque`. Continuation markers: 37 EN+IT items
> - `resolved_mode` silently falls back to `structural` for any unrecognized mode; `both` is expanded by the pipeline, never seen by the segmenter
> - Debug histogram in `session.metadata["segmentation_debug"]`

**Known gap.** `_looks_incomplete` returns `True` for any text not ending in
`.?!`, so on an unpunctuated ASR transcript "incomplete" is effectively the
default verdict, and the soft-gap override fires more often than the signal list
suggests.

---

## 5. QA extraction

This is the core of the system: roughly 9,600 lines in one module, of which the
large majority is not "find the question and the answer" but "decide whether
this thing is a question at all, and whether that thing really answers it".

### 5.1 Shape of the stage

`QAPairExtractor.extract(session)` is the only public method. It returns a list
of `QAPairCandidate` and writes three run-level diagnostic blocks into
`session.metadata`: `qa_coverage`, `qa_semantic_responsiveness` and
`qa_speaker_rescue`.

The stage runs in thirteen steps:

```
 1  guard on enable_qa_extraction
 2  _prepare_input          pick the sentence layer or the fallback layer
 3  _build_segment_lookup   unit/sentence -> segment id maps
 4  _detect_questions       candidate detection, scoring, gate at 0.45
    for each surviving question:
 5    generate_candidates   answer search  (strategy object)
 6    rank_candidates       answer ranking (strategy object)
 7    _maybe_run_deferred_answer_search    far-window rescue + re-rank
 8    _select_ranked_answer + _build_qa_candidate   confidence, timings, context
 9  _apply_semantic_responsiveness_scoring   optional embedding rescoring
10  _qa_candidate_suppression_reason         emission gating
11  _dedupe_qa_candidates                    echo + near-duplicate resolution
12  _apply_speaker_assisted_rescue           acoustic rescue of gated candidates
13  _build_qa_coverage                       run-level counters
```

Steps 5 and 6 are dispatched through two protocols, `AnswerSearchStrategy` and
`AnswerRankingStrategy`. This is the intended extension point: a new retrieval or
ranking method is a new class implementing one of these, selected by the
`qa_answer_search_strategy` / `qa_answer_ranking_strategy` config values and
injected in the constructor. Both semantic strategies are wrappers that hold the
corresponding local strategy as a fallback, which is why a missing model never
breaks the stage.

> **Reference**
>
> - [_qa_extractor_impl.py:1341](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L1341) (`QAPairExtractor`), `extract` at [:1387](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L1387)
> - Protocols: `AnswerSearchStrategy` [:543](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L543), `AnswerRankingStrategy` [:559](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L559)
> - Implementations: `_LocalRuleBasedAnswerSearcher` [:582](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L582), `_SemanticAnswerSearcher` [:748](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L748), `_RuleBasedAnswerRanker` [:1121](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L1121), `_SemanticAnswerRanker` [:1177](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L1177)
> - Caller: `ProcessingPipeline.extract_qa_candidates` [:387](../src/lecture_analyzer/core/_processing_pipeline_impl.py#L387)

### 5.2 Input preparation

Sentences and merged transcript units are normalized into one internal shape,
`_ExtractionUnit`, carrying index, layer, text, timings, audio source, the
provenance id lists, and - when the source was a sentence - the full set of
speaker and quality labels from section 3.

The layer choice propagates as a penalty. A candidate built from the fallback
layer loses 0.04 confidence, gains the review flag `fallback_input_layer`
(risk weight 0.16), and carries the reason code `merged_transcript_fallback`.
Every scoring function that reads sentence metadata returns a neutral zero delta
on the fallback layer, and the semantic searcher refuses to run at all
(`sentence_layer_required`).

> **Reference**
>
> - `_prepare_input` [:3424](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L3424); `_ExtractionUnit` [:415](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L415)
> - With neither sentences nor merged units, `extract` returns `[]` with an empty coverage block

### 5.3 Question detection

Detection is **purely lexical and orthographic**. There is no classifier, no
prosody, no acoustic input, and speaker identity is not used as a detector - only
as a score adjustment. This is a deliberate tradeoff: the rules are inspectable
and every decision produces a reason code, which is what makes the human review
workflow in section 6 possible.

#### Extraction

Each unit is examined by `_extract_question_parts`, which tries three modes in
cascade over the unit's sentences:

**(a) A sentence containing `?`.** The best-scoring one wins. Within it,
`_refine_question_focus` separates a preamble from the actual question, so
*"I was reading this last night, but how does the compiler know?"* yields the
clause after the comma rather than the whole run-on. Two patterns drive this: a
conjunction pivot (`, but how ...?`) requiring a focus of at least 4 tokens, and
a wh-anchored tail requiring at least 2. Overlong originals (> 24 tokens) are
tagged `runon_question_candidate`.

Text *after* the `?` becomes a `local_answer_seed` and marks the unit
`intra_sentence_qa` - the Socratic case where a lecturer asks and immediately
answers within one sentence.

**(b) A colon split.** `prefix : suffix` where the prefix carries a strong
question signal.

**(c) An implicit cue sentence** with no `?` at all. This handles the didactic
frame that dominates lecture speech: *"la domanda e come si calcola"*,
*"the question is what happens next"*. `_refine_implicit_question_focus` strips
the frame and keeps the focus.

#### Vetoes and recovery

Three mechanisms fight the two dominant error modes.

*A causal clause after a question is an answer, not a question.* If a unit has no
`?`, matches a causal pattern (`perche ...`, `because ...`), and the previous
unit contained a question within 12 seconds, it is rejected outright.

*A declarative "what" is not a question.* `DECLARATIVE_WHAT_PATTERNS`
(`that's what ...`, `what it means ...`, `what we call ...`, `what happens is
...`) is a hard veto when there is no `?`.

*A real question may have lost its question mark.* ASR frequently drops terminal
punctuation. Recovery requires two independent gates: the preamble must look
like a question frame (ending in `:` or `?`, or a bridge tail like `cioe` /
`that is` / `i mean`, or containing `la domanda` / `the question` in its last
four tokens), **and** there must be a plausible local answer slot - an answer
seed, or a next unit within 3.5 seconds - with no competing interrogative cue in
the adjacent previous unit.

#### Span repair

If a question was split across units by the sentence splitter,
`_recompose_split_question_span` reassembles it, looking up to 2 units backward
or forward. Units are glued only when they share an audio source, sit within 3.5
seconds, agree on speaker where known, and each glued unit passes
`_is_split_question_setup_unit` (no `?`, 3-24 tokens, no autonomous
interrogative head, not procedural).

Otherwise, a *contextual* question - an anaphoric stub like *"why?"*, *"really?"*,
*"is that true?"* - gets the previous unit prepended by `_expand_question_context`
under the same locality constraints.

#### Scoring

`_evaluate_question_text` produces an additive score. Empty text, text over 220
characters, a declarative-what match, a causal declarative, or no cue of any kind
without a `?` all return `None` (rejected outright).

```
  -0.28  rhetorical check-in ("can you hear me", "ci siamo")
  -0.42  rhetorical poll ("one or two?", "uno o due?")
  -0.20  poll or backchannel noise
  -0.26  procedural request
  -0.30  declarative tag question
  -0.22  fragment question
  + intent delta          poll_or_check -0.18, rhetorical_tag -0.18,
                          fragment -0.16, subordinate_fragment -0.18,
                          embedded_statement -0.12, weak_form -0.10
  -0.18  no '?' and no interrogative head
  +0.40  contains '?'
  +min(0.36, sum of matched question cue weights)
  +min(0.30, sum of matched didactic cue weights)
  +0.18  direct definition request ("can you explain ...", "puoi spiegare ...")
  +0.12  starts with an interrogative word
  +0.10  implicit question cue        (+0.06 more if terminal-mark recovery fired)
  +0.08  3 <= token_count <= 24       (-0.25 if token_count < 2)
  -> clamp to [0,1]
```

Two adjustments follow. A **context delta** derived from the sentence labels of
section 3 - utterance grounding +/-0.03, semantic quality up to -0.08, merge
safety up to -0.06, review priority up to -0.04, autonomy below 0.45 -0.08,
boundary confidence below 0.45 -0.06, speaker resolution up to -0.05 - and a
**span integrity penalty** of -0.10 when the extracted focus starts mid-sentence
at a boundary that is not a clean break.

A second, differently weighted score is computed in parallel:

```
raw_question_score      = clamp(eval + context_delta + integrity_delta)
didactic_question_score = clamp(eval + clip(context_delta, -0.02, +0.04)
                                + 0.06 if context was expanded
                                + 0.04 if intra-sentence QA
                                - the noise penalties again
                                + 0.03 for why/how/where/difference/didactic types)
question_score          = max(raw_question_score, didactic_question_score)
```

The `max` is the point: the didactic variant is more forgiving of weak sentence
context and more rewarding of pedagogically useful question types, so a
genuinely didactic question is not killed by a poor sentence boundary. Both
scores survive into the output, and `didactic_question_score` separately gates
the deferred answer search.

The gate is `min_question_score` = **0.45**.

> **Reference**
>
> - `_detect_questions` [:3522](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L3522), `_extract_question_parts` [:4016](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L4016), `_evaluate_question_text` [:4218](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L4218), `_score_question_context` [:4598](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L4598), `_score_didactic_question_usefulness` [:4551](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L4551)
> - Focus refinement `_refine_question_focus` [:4172](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L4172), `_refine_implicit_question_focus` [:4103](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L4103)
> - Recovery `_allows_missing_terminal_preamble_recovery` [:3728](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L3728), `_has_missing_terminal_local_support` [:3800](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L3800)
> - Span repair `_recompose_split_question_span` [:3838](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L3838), `_expand_question_context` [:4471](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L4471)
> - Rule tables: [_qa_rules_impl.py](../src/lecture_analyzer/analysis/_qa_rules_impl.py) - `QUESTION_CUE_RULES` (24 weighted EN+IT patterns, 0.16-0.28), `DIDACTIC_QUESTION_RULES` (5, 0.28-0.30), `ANSWER_CUE_RULES` (17, 0.08-0.24), `INTERROGATIVE_START_WORDS`, `DECLARATIVE_WHAT_PATTERNS`
> - Output dataclass `QuestionCandidate` [:453](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L453)

### 5.4 Answer search

Three channels feed candidates, and they compose rather than compete: the local
searcher always runs, the semantic searcher adds to its results, and the
deferred searcher only engages when the first two produced nothing convincing.

#### Local rule-based search

Walks forward from the question over 1 to `answer_search_window_units` (**3**)
units, producing a growing consecutive span at each distance. Four other
candidate shapes are also emitted:

- **Same-unit seed** - text after the `?` inside the question's own sentence
  (distance 0, gap 0), the Socratic case
- **Span completion** - at the window boundary, one continuation sentence is
  appended when the answer clearly runs on
- **Competing-question handling** - when another question appears inside the
  window, the text *before* it is offered as an answer, and optionally an
  extended span including that prefix
- **Interview cluster** - skips up to 4 bridge units (a moderator prompt or an
  echoed restatement of the question) to reach the real answer, requiring at
  least one prompt/echo bridge, a gap within 45 s, and a plausible resulting
  answer of at least 5 tokens with at least 3 answer-only tokens

The walk stops with an explicit reason, which is exported:
`window_exhausted`, `competing_question`, `audio_source_boundary`,
`temporal_gap_limit` (gap > 45 s), `answer_span_limit` (> 3 units),
`answer_duration_limit` (span > 45 s), `cross_segment_block`,
`question_continuation_skipped`.

Whether two adjacent units are one continuing sentence is decided by
`_looks_like_sentence_continuation`: the previous text ending in `,;:` says yes;
the next text starting with an EN/IT connective says yes; a lowercase start says
yes; two different known speakers says no; otherwise a gap under 0.35 s.

#### Semantic retrieval

Embeds the question and each candidate answer sentence with
`intfloat/multilingual-e5-base` and ranks by cosine similarity. E5 requires
asymmetric prefixes, applied literally as `query: ` and `passage: `; embeddings
are L2-normalized at encode time, so cosine is a dot product.

The search window widens to 8 units, top 4 hits above similarity 0.2 are kept,
and each hit is expanded into spans of 1 to 3 units. Results are merged onto the
local baseline keyed by answer sentence id tuple, **baseline first** - semantic
retrieval adds candidates, it never removes them.

Each candidate is tagged with `candidate_channel` (`baseline_local` or
`semantic_retrieval`) and its similarity and rank, all of which reach the debug
workbook.

#### Deferred far-window search

The rescue path for questions answered much later. It is deliberately hard to
trigger:

- the question's `didactic_question_score` must be at least **0.62**
- rhetorical, poll, fragment and tag questions are excluded outright
- there must be no strong local answer (score >= 0.52 within 10 s with shared
  keywords counts as strong, and blocks the search)
- local relevance must be genuinely low: best score under 0.38, or the best
  answer is itself a question, or no shared keywords
- a forward scan over the next 28 units must find at least one unit with a
  signal score of at least 0.18

Surviving candidates are then filtered again by distance and gap, on the
principle that the further away an answer is, the more lexical evidence it must
carry:

```
always:        gap > 120 s and (signal < 0.34 or fewer than 3 keywords and no numbers) -> drop
               gap >  60 s and  signal < 0.28                                          -> drop
               distance > 12 and signal < 0.24 and fewer than 2 keywords               -> drop
quality_local: gap >  75 s / gap > 30 s / distance > 10, with tightened thresholds     -> drop
```

Anything that survives is merged in, re-ranked, and tagged
`deferred_answer_search`.

> **Reference**
>
> - Local: `_LocalRuleBasedAnswerSearcher.generate_candidates` [:582](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L582); builders `_build_same_unit_answer_candidate` [:4930](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L4930), `_build_answer_candidate_from_units` [:4982](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L4982), `_build_interview_cluster_answer_candidate` [:5179](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L5179)
> - Semantic: [_semantic_retrieval_impl.py](../src/lecture_analyzer/analysis/_semantic_retrieval_impl.py) (`SentenceTransformersE5Backend`), wrapper `_SemanticAnswerSearcher` [:748](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L748)
> - Deferred: `_maybe_run_deferred_answer_search` [:6329](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L6329), trigger `_deferred_search_trigger_debug` [:7817](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L7817), filter `_deferred_alignment_is_strong_enough` [:8076](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L8076)
> - Config: `answer_search_window_units` 3, `max_answer_units` 3, `max_answer_duration_seconds` 45.0, `qa_semantic_retrieval_top_k` 4, `qa_semantic_retrieval_window_units` 8, `qa_semantic_retrieval_min_similarity` 0.2, `deferred_answer_search_window_units` 28, `deferred_answer_search_min_question_score` 0.62, `deferred_answer_search_local_score_threshold` 0.38, `deferred_answer_search_min_signal_score` 0.18

There is **no hard speaker constraint on pairing anywhere**. A question and its
answer are not required to come from different speakers - lectures are full of
self-answered questions. Speaker identity only shifts the score (section 5.5)
and can trigger the acoustic re-check (section 5.10).

### 5.5 Answer scoring

`_score_answer_candidate` is an additive ledger. Every term is written into
`partial_scores` on the candidate, so any final score can be decomposed after
the fact - this is the audit trail the debug workbook and the review packet
expose. Grouped by what they measure:

**Proximity.** A base bias of +0.18; a distance term (+0.28 at distance 0,
+0.20 at 1, +0.12 at 2, +0.06 beyond); a gap term (+0.12 within 4 s, +0.06 within
10 s, -0.08 beyond); and a segment relation term (+0.10 same, +0.05 next, -0.05
distant).

**Lexical alignment.** Answer cue rules up to +0.30 ("the answer is", "la
ragione e", "because", "quindi"); shared content tokens up to +0.15; shared
numbers +0.12, or +0.10 when a quantity question meets an answer containing
numbers. Relevance is +0.10 when anything at all is shared, and negative when
nothing is - as far as **-0.28** for a quantity question whose answer has no
numbers.

**Responsiveness.** A sub-score in [-0.24, +0.06] built from topical anchoring,
quantity matching, answer-only token count, and whether the reply is thin,
socratic, unanchored or a follow-up prompt. Its raw value is also exported as a
normalized `responsiveness = clamp(0.55 + raw * 1.6)` with labels
`answer_responsiveness_weak` / `_strong`.

**Span quality.** Completeness of the answer span (+0.12 down to -0.10 for a
premise-only answer, -0.08 for spans over 80 tokens); a length bonus of +0.08
for 3-80 tokens; and a penalty of -0.20 if the answer itself contains a `?`,
compounding to **-0.35** if the answer is structurally a question.

**Quality gate.** The largest block of negative evidence, applied as one
`quality_gate` delta: moderator handoff -0.22, boilerplate -0.26, poll or
backchannel -0.34, question continuation -0.24, same-sentence answer -0.16 (a
further -0.26 without an answer cue), echo -0.28, circular echo -0.22,
low-information -0.12, incomplete span -0.10, truncated boundary -0.16, surface
answer cue -0.18, **deflection -0.42**, deferred-too-broad -0.14; with a single
positive, +0.08 for a legitimate short socratic answer.

Deflection deserves a note: it detects an answer that acknowledges the question
without addressing it ("that's a course-level topic, not for today") by requiring
a short answer with no topical anchors, at least one meta token (course, lesson,
slide, exam), at least one dismissive or negation token, and a meta-token ratio
of at least 0.20.

**Speaker evidence.** Both spans reliably attributed to *different* speakers is
+0.10 (a further +0.03 when both are dominant); same speaker is -0.02 to -0.04;
a conflicted span is -0.05. Separately, a monologue-continuation risk term
subtracts up to -0.22 when the "answer" looks like the same person simply
continuing to talk.

**Competition.** -0.08 when the search stopped on a competing question.

> **Reference**
>
> - `_score_answer_candidate` [:5305](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L5305); sub-scorers `_score_answer_quality_gate` [:5775](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L5775), `_score_speaker_pairing` [:6161](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L6161), `_score_monologue_continuation_risk` [:6236](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L6236), `_deflection_answer_debug` [:5930](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L5930)
> - Speaker reliability ladder `_speaker_reliability` [:8235](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L8235): `reliable` / `weak` / `conflict` / `missing`
> - Answer text cleanup before scoring: `_trim_answer_meta_opening` (strips "that's a good question"), `_trim_answer_trailing_tag` (strips a trailing "right?"), `_trim_internal_checkin_from_answer` (removes interior "can you hear me" clauses)

### 5.6 Ranking

The rule-based ranker sorts by `(-answer_score, distance_units, gap_seconds,
completion-support-first, span_length)`.

The semantic ranker runs that first, then rescores the top 8 with the
cross-encoder `BAAI/bge-reranker-v2-m3` - which, unlike the bi-encoder in
retrieval, sees the question and answer jointly and is therefore much better at
judging actual responsiveness. Logits are sigmoid-normalized and blended:

```
answer_score = clamp(0.7 * semantic_score + 0.3 * rule_based_score)
```

then the list is re-sorted on the same key and `rank_position` assigned.

> **Reference**
>
> - [_semantic_reranking_impl.py](../src/lecture_analyzer/analysis/_semantic_reranking_impl.py), `max_length=512`, `model.eval()` under `torch.no_grad()`
> - Blend at `_SemanticAnswerRanker.rank_candidates` [:1231](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L1231)
> - Config: `qa_semantic_reranking_max_candidates` 8, `qa_semantic_reranking_weight` 0.7
> - Candidates beyond the rerank limit keep their rule score and are marked `semantic_reranking_status="skipped"`

### 5.7 Confidence and quality

Two different numbers are produced, and they answer different questions.

**Confidence** answers "how sure are we this is a QA pair":

```
base       = 0.55 * question_score + 0.45 * answer_score
confidence = clamp(base - 0.10 if a competing question was nearby
                        - 0.04 if built from the fallback input layer)
             then clamp(confidence - 0.24) if the answer is itself a question

label: >= 0.75 high | >= 0.45 medium | else low
```

**Quality** answers "how useful is this pair to a reviewer", and is built from
four component scores plus an explicit risk subtraction:

```
question_quality    = clamp(0.70 * question_score + 0.30 * didactic_question_score)
answer_quality      = clamp(answer_score)
context_quality     = 0.90 high | 0.68 medium | 0.40 low | 0.20 no context | 0.55 otherwise
grounding_quality   = clamp(sum of provenance bonuses: sentence layer 0.18,
                            question/answer sentence ids 0.12 each,
                            utterance ids 0.14 each, timings 0.08 each,
                            segment relation +0.10/+0.06/-0.08, context ids 0.04)

base_quality  = clamp(0.34*Q + 0.38*A + 0.16*C + 0.12*G)
risk_score    = clamp(sum of matched risk weights, capped at 8 reasons)
final_quality = clamp(base_quality - 0.45 * risk_score)

quality_band: >= 0.74 high | >= 0.50 medium | else low
risk_band:    >= 0.55 high | >= 0.25 medium | else low
```

`risk_score` aggregates a weighted table of roughly 40 named risk reasons drawn
from the candidate's reason codes and review flags. The weights cluster in three
tiers: **0.26-0.32** for reasons that usually mean the pair is wrong (the answer
is a poll or backchannel, the answer is a question, low relevance, premise-only
answer, follow-up prompt); **0.18-0.24** for structural weakness (procedural or
fragment questions, monologue continuation, echo, boilerplate, weak
responsiveness); and **0.08-0.16** for provenance and boundary concerns
(fallback layer, low autonomy, truncated answer, unknown segment). Medium
confidence adds a flat 0.04.

> **Reference**
>
> - Confidence [:6419](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L6419); quality `_build_candidate_quality_features` [:7455](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L7455); risk table `_quality_risk_reasons_and_score` [:7619](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L7619); bands `_quality_band` / `_risk_band` [:7776](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L7776)
> - Both are exported: `confidence` / `confidence_label` on the candidate, the rest under `metadata["quality_features"]`

### 5.8 Emission gating

Not every built candidate is emitted. Gating happens in two tiers.

**Tier 1, always active.** `confidence < min_qa_confidence` (**0.40**) suppresses
with reason `below_min_qa_confidence`.

**Tier 2, `quality_local` only.** Roughly 35 additional rules. These are what
turn a high-recall extractor into a high-precision one, and they are worth
stating plainly: **under every profile other than `quality_local`, none of them
run** - anything above 0.40 confidence is emitted. The benchmark numbers in
[quality_evaluation.md](quality_evaluation.md) are `quality_local` numbers, and
are not representative of the default configuration.

The tier-2 rules fall into families:

- *Weak-pair reasons need overwhelming confidence.* A candidate carrying a
  rhetorical, poll, tag, fragment, subordinate-fragment or procedural reason code
  is dropped below 0.72. An `embedded_statement_question` needs 0.78.
- *Deferred answers are capped.* Dropped when the gap exceeds 45 s, the distance
  exceeds 10 units, the answer exceeds 45 tokens, or the quality gate went
  negative.
- *Band rules.* `risk_band == high` with final quality under 0.58 is dropped;
  `quality_band == low` with risk at or above 0.25 is dropped.
- *Specific risks have specific floors.* `surface_answer_cue_risk` needs 0.62,
  `incomplete_answer_span` 0.65, `monologue_continuation_risk` 0.68,
  `semantic_nonresponsive` 0.72, `unanchored_quantity` and `followup_prompt`
  0.76, `weak_implicit_quantity` 0.78.
- *Any poll or backchannel risk is fatal*, at any confidence.
- *Deflection is fatal*, checked before the confidence gate so that it is
  reported as its own suppression reason.

Against these sit five **escape hatches** - patterns that are genuinely valuable
and would otherwise be caught by the generic rules: `local_socratic_completion`
(0.44 / 0.38), `interview_cluster_completion` (0.52 / 0.38 / 0.45),
`interview_echo_completion` (0.60 / 0.48), `terminal_local_interview_question`
(0.50 / 0.34), and `extended_before_competing` (0.60 / 0.42 / 0.55).

Every suppression records a stable reason string, counted into
`qa_coverage.suppressed_by_gate_reasons`. Suppressed candidates are not
discarded immediately - they are held for the rescue stage in 5.10.

> **Reference**
>
> - `_qa_candidate_suppression_reason` [:2666](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L2666), `_should_emit_qa_candidate` [:2770](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L2770), `_infer_quality_gate_suppression_reason` [:2686](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L2686)
> - The profile check is the first line of `_should_emit_qa_candidate`: a non-`quality_local` profile returns `True` immediately

### 5.9 Deduplication

Two overlap problems are resolved after gating, both using the same metric:

```
overlap(A, B) = |tokens(A) and tokens(B)| / min(|tokens(A)|, |tokens(B)|)
```

over content tokens (stopword-filtered, longer than one character). Both passes
only consider candidates whose start times are within **90 seconds**.

**Echo pairs.** A lecturer asks a question, a student restates it, and the
extractor emits both directions of the same exchange. Detected by comparing a
candidate against its immediate predecessor: answers overlapping at >= 0.45 and
at least one cross pairing (left question vs right answer, or vice versa) at
>= 0.55. The winner is whichever question echoes the other less, provided the
echo difference is at least 0.05; otherwise a quality sort on
`(-noise_penalty, confidence, answer_token_count)`. Contextual follow-up
questions are exempt, since a genuine follow-up legitimately overlaps its
predecessor.

**Near duplicates.** A backwards scan over all kept candidates, matching at
question overlap >= 0.72 with answer overlap >= 0.50, or question overlap >= 0.86
with answer overlap >= 0.35. The better candidate by
`(confidence, -review_flag_count, answer_token_count)` replaces the other.

Losers keep the reason codes `echo_pair_deduplicated` and
`deduplicated_near_duplicate` rather than vanishing silently.

> **Reference**
>
> - `_dedupe_qa_candidates` [:3207](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L3207), `_candidates_form_echo_pair` [:3277](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L3277), `_find_near_duplicate_candidate` [:3368](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L3368), `_token_overlap_ratio` [:3412](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L3412)
> - Span-level dedup also happens earlier, in both `_merge_candidates` and `_merge_answer_candidates`, keyed on the answer sentence id tuple

### 5.10 Acoustic speaker check and rescue

This is the only place where audio re-enters the QA decision after
transcription. A SpeechBrain ECAPA-TDNN model embeds the question audio span and
the answer audio span, and their cosine similarity is interpreted as:

```
similarity >= 0.72  ->  same_speaker_suspected
similarity <= 0.45  ->  different_speaker_likely
otherwise           ->  inconclusive

same-speaker penalty ramps linearly from 0 at 0.72 to 0.25 at 0.85
different-speaker bonus: +0.04
```

Spans shorter than 1.5 seconds are widened within their utterance by up to 3.0
seconds, then padded; if still too short the check reports `unavailable`.

The penalty is waived in two cases where a same-speaker result is legitimate: a
socratic pattern (the lecturer answering their own question, already flagged
upstream), and a responsive definition (a definition question met by a
definition answer with strong responsiveness markers).

This runs twice, for two different purposes. As **stage 12** of the pipeline it
adjusts the confidence of already-emitted candidates. Inside the extractor it
drives **rescue**: candidates suppressed by the tier-2 gates are re-examined, and
those whose only real weakness was ambiguous speaker evidence are reinstated when
the audio proves the two spans came from different people.

Rescue is deliberately narrow. Only five suppression reasons are eligible
(`low_autonomy_implicit_question`, `weak_expanded_contextual_question`,
`surface_answer_cue_risk`, `below_min_qa_confidence`,
`weak_answer_responsiveness`), and `below_min_qa_confidence` additionally
requires the candidate to be within 0.08 of the threshold. Eighteen reason codes
and eight risk reasons block rescue outright. The check must return
`different_speaker_likely`, the answer must show minimal responsiveness anchoring,
and no near-duplicate may already be emitted. Rescued candidates have their
question refocused and their answer truncated at a clause boundary within 80
words. Caps: 40 checks and 8 rescues per run.

> **Reference**
>
> - [qa_speaker_check.py](../src/lecture_analyzer/analysis/qa_speaker_check.py) - `QASpeakerCheckService`, `SpeechBrainSpeakerEmbeddingBackend`, `_same_speaker_penalty_for_similarity` [:847](../src/lecture_analyzer/analysis/qa_speaker_check.py#L847), waivers `_has_socratic_pattern` [:797](../src/lecture_analyzer/analysis/qa_speaker_check.py#L797) and `_has_responsive_definition_waiver` [:821](../src/lecture_analyzer/analysis/qa_speaker_check.py#L821)
> - Rescue: `_apply_speaker_assisted_rescue` [:1565](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L1565), eligibility `_speaker_rescue_candidate_is_eligible` [:2216](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L2216)
> - Model path default: `<repo>/local_models/spkrec-ecapa-voxceleb`; `qa_speaker_check_enabled` defaults to whether that path exists
> - Config: `qa_speaker_check_min_span_seconds` 1.5, `..._same_threshold` 0.72, `..._same_full_penalty_threshold` 0.85, `..._different_threshold` 0.45, `..._same_speaker_penalty` 0.25, `..._different_speaker_bonus` 0.04, `qa_speaker_rescue_max_checks_per_run` 40, `..._max_candidates_per_run` 8, `..._min_confidence_margin` 0.08, `..._min_text_quality_score` 0.50, `..._answer_word_cap` 80

### 5.11 Semantic responsiveness

An optional post-hoc rescoring of already-emitted candidates, off by default.
It does not generate candidates and cannot change which answer was chosen - it
only annotates, and optionally penalizes.

```
qa_sim   = clamp01(cos(question, answer))
cont_sim = clamp01(cos(answer, continuation))     # continuation = context text

score = clamp(0, 1, 0.18 + 0.82 * qa_sim - echo_penalty - continuation_penalty)

echo_penalty:          0.24 if qa_sim >= 0.88 and token overlap >= 0.72
                       0.14 if qa_sim >= 0.82 and token overlap >= 0.62
continuation_penalty:  0.22 if (cont_sim - qa_sim) >= 0.12
                       0.12 if (cont_sim - qa_sim) >= 0.06
```

The two penalties encode the failure modes a raw similarity score cannot see:
an answer that merely *repeats* the question scores high on similarity but is
worthless, and an answer that resembles the surrounding narration more than the
question is probably just continuing narration.

When the gate is enabled, a score below 0.45 subtracts 0.12 from confidence and
final quality, adds the same to risk, and appends the review flag
`semantic_nonresponsive`.

Unlike the other two models, this one is **strictly offline**: the model name is
first treated as a filesystem path, then resolved through
`snapshot_download(..., local_files_only=True)`. It can never download at
runtime.

> **Reference**
>
> - [_semantic_responsiveness_impl.py](../src/lecture_analyzer/analysis/_semantic_responsiveness_impl.py); model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, device `cpu`
> - Applied at `_apply_semantic_responsiveness_scoring` [:2340](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L2340); gate at [:2562](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L2562)
> - Config: `qa_semantic_responsiveness_enabled` False, `..._max_candidates` 15, `..._gate_enabled` False, `..._gate_min_score` 0.45, `..._gate_penalty` 0.12
> - Run metrics in `session.metadata["qa_semantic_responsiveness"]`, including `load_seconds` and `model_footprint_bytes`
> - This layer appears in no profile; it is reachable only through CLI flags or programmatic config

### 5.12 Bilingual handling

The system processes Italian and English lectures, including code-switched ones,
and does so **without any language identification step**. There is no language
branch anywhere in the QA extractor.

Instead, every rule table carries both languages side by side, and
`normalize_rule_text` applies NFKD decomposition with ASCII stripping before
matching - so `perché` becomes `perche`, `cioè` becomes `cioe`, `può` becomes
`puo`, and Italian patterns are written unaccented once rather than twice.

Concretely this covers: 9 Italian question cues alongside the English ones; the
Italian didactic frames (`la domanda e`, `ci chiediamo`, `ci possiamo
chiedere`); 8 Italian answer cues (`la risposta e`, `perche`, `quindi`,
`infatti`, `cioe`, `allora`); Italian interrogative heads and auxiliaries
(`c'e`, `esiste`, `puo`, `possiamo`); the classroom check-ins `ci siamo` and
`mi state sentendo`; Italian follow-up prompts (`racconta`, `spiega`, `dicci`);
Italian poll options (`uno` ... `cinque`); and Italian dangling-word markers
(`al`, `alla`, `di`, `per`, `se`). All three embedding models are multilingual
by choice for the same reason.

> **Reference**
>
> - `normalize_rule_text`, `count_tokens`, `collect_rule_matches`: [_qa_rules_impl.py:226](../src/lecture_analyzer/analysis/_qa_rules_impl.py#L226)
> - The same bilingual-table approach is used in the segmenter's marker lists and the sentence reconstructor's incomplete markers

**Known gap.** `_summarize_context_text` contains five hardcoded English
summaries tied to specific content in the development corpus (the imperial
measurement system, inches versus centimetres on a ruler, "five tomatoes",
5,280 feet, "take an L"). These are dataset-fitted strings, not general rules,
and they will produce wrong summaries on unrelated material.

### 5.13 Degradation matrix

Every ML and upstream dependency in this stage fails open. This table is the
practical reference for interpreting a run that produced fewer candidates than
expected.

| Missing | Behaviour | Emitted marker |
|---|---|---|
| `session.sentences` | Build units from merged transcript | `input_layer="merged_transcript_fallback"`, -0.04 confidence, flag `fallback_input_layer` |
| Sentences and merged transcript | Return `[]` | empty `qa_coverage` |
| Non-sentence layer | All sentence-metadata deltas become 0 | `sentence_metadata_available: False` |
| Semantic retrieval model | Local baseline only | `semantic_backend_status="fallback"`, `search_fallback_reason` |
| Semantic reranker model | Rule-based order kept | `semantic_reranking_status="fallback"`, `ranking_fallback_reason` |
| Responsiveness model | No scores, no gate | `semantic_responsiveness_status="fallback"` |
| Utterance grounding | Timings fall back to unit clock | `timing_source` = layer name, -0.03/-0.05, grounding score loses 0.14/0.08 |
| Segments | Neutral relation | `segment_relation="segment_unknown"`, flag `segment_unknown` |
| Speaker labels | No pairing bonus or penalty | reliability `missing` |
| Speaker-check model | Rescue skipped | `model_available: False`, note `model_not_configured_or_missing` |
| Speaker spans under 1.5 s | Widen, pad, else give up | status `unavailable` |
| Local answer | Deferred far-window search | `deferred_answer_search` |
| Any answer at all | Question silently dropped | counted in `qa_coverage` as uncovered |
| Usable context | Cascade to preamble, previous unit, then none | `fallback_previous_context`, `weak_context_risk` |

> **Reference**
>
> - The one exception to fail-open is transcription: a missing faster-whisper raises `BackendInitializationError`
> - Coverage counters: `_build_qa_coverage` [:1487](../src/lecture_analyzer/analysis/_qa_extractor_impl.py#L1487); the denominator is `_is_interrogative_sentence_proxy`, a cheap `?`-or-cue test over all units

---

## 6. Output

Four exporters, serving four different consumers.

### 6.1 JSON export

The production artifact. The exporter itself has no schema logic - it is
`json.dumps(session.to_dict(), indent=..., ensure_ascii=False)`. Its real
responsibility is deterministic naming: `{base_name}_{mode}.json`, where the
base is the single input's file stem or the session id, sanitized to lowercase
alphanumerics and underscores. If `output_path` ends in `.json`, only its parent
directory is used - the filename is always regenerated, so a mode is never
silently written over another mode's file.

Top-level keys: `session_metadata`, `input_sources`, `audio_sources`,
`transcript` (derived counters), `merged_transcript`, `transcript_chunks`,
`aligned_transcripts`, `diarization_segments`, `utterances`, `sentences`,
`segments`, `speaker_role_estimates`, `qa_candidates`, `pipeline_timing`.

A `QAPairCandidate` carries 34 fields, in four groups:

- **Content**: `qa_candidate_id`, `question_text`, `answer_text`, `context_text`
- **Provenance**: six id lists (question/answer/context x sentence/utterance),
  `question_unit_ids`, `answer_unit_ids`, `question_segment_id`,
  `answer_segment_id`, `source_segment_ids`
- **Timing and role**: `start_seconds`, `end_seconds`, `question_timing`,
  `answer_timing` (each a `TimeRange` with local and session clocks),
  `question_speaker_role`, `answer_speaker_role`
- **Judgement**: `confidence`, `confidence_label`, `confidence_score`,
  `question_type`, `didactic_question_score`, `answer_is_question`,
  `context_strategy`, `context_confidence`, `reason_codes`, `review_flags`,
  `extraction_notes`, and `metadata`

`metadata` is the audit bag: `quality_features`, `question_debug`,
`answer_debug` (including `partial_scores` and `search_signals`),
`pairing_debug`, `grounding_debug`, `context_debug`, `confidence_debug`,
`ranking_debug`, `search_debug`, `speaker_check`,
`semantic_responsiveness_debug`, `rank_position`.

> **Reference**
>
> - [_json_exporter_impl.py](../src/lecture_analyzer/output/_json_exporter_impl.py); `QAPairCandidate` at [_models_impl.py:423](../src/lecture_analyzer/core/_models_impl.py#L423); `schema_version = "0.6.0"`
> - Mode normalization silently falls back to `structural` for an unrecognized mode

**Known gap.** `_serialize_value` drops every `None` from dicts and lists
([_models_impl.py:811](../src/lecture_analyzer/core/_models_impl.py#L811)).
Optional fields are therefore **absent** from the JSON rather than null: a
candidate with no answer has no `answer_text` key at all. Consumers must use
`.get()` and must not assume a fixed key set.

### 6.2 Debug Excel workbook

The human review surface, generated from the exported JSON rather than from
memory so that it always reflects what was actually written. Six sheets:

| Sheet | Purpose |
|---|---|
| `Utterances` | The timed base layer, with speaker colour banding and change borders |
| `Sentences` | Reconstruction quality: labels, buckets, speaker evidence, provenance |
| `QACandidates` | One row per candidate, core columns first, then the full technical trail |
| `ReviewCandidates` | Everything flagged for human attention, with a suggested action |
| `SentenceMetrics` | Aggregate sentence-layer metrics |
| `run_summary` | Run-level counters |

Each reviewable sheet has a `human_comment` column. On regeneration the exporter
**reopens the existing workbook first and re-injects those comments** by row key,
matching several header aliases (`human_comment`, `manual_comment`, `comment`,
`commento`, `human_note`). This is what makes iterative review possible: rerun
the pipeline, keep the annotations. Note that any exception during that reload is
swallowed, and the comments are then lost silently.

> **Reference**
>
> - [_debug_excel_exporter_impl.py](../src/lecture_analyzer/output/_debug_excel_exporter_impl.py); `_load_existing_human_comments` [:663](../src/lecture_analyzer/output/_debug_excel_exporter_impl.py#L663)
> - `QACandidates` technical columns expose the entire semantic path: requested vs effective search and ranking strategy, stop and fallback reasons, backend status and errors, model names, `semantic_similarity`, `candidate_channel`, `semantic_relevance_score`, `combined_rank_score`, `rank_position`, `speaker_influence`, timing sources, `deferred_answer_search_used`, `segment_relation`, `input_layer`

### 6.3 AI review packet

A single Markdown file designed to be pasted into an external reviewer - human
or LLM. It contains the reviewer instructions, run metadata, a timing summary,
the coverage summary, the transcript, and one block per candidate with its
question, answer, context and full judgement trail.

It asks for 1-5 scores on `question_quality`, `answer_quality`,
`context_quality` and `grounding_quality`, plus a
`keep` / `revise` / `reject` decision, and embeds the exact expected JSON
envelope so the response can be pasted straight back into `ai_review.json`.

A "Runtime Review Signals" section explicitly flags stages that took under a
millisecond while reporting cache or artifact reuse, with a warning not to
compare profiles across different cache states - the most common way to
misread the timing numbers.

> **Reference**
>
> - [_ai_review_packet_exporter_impl.py](../src/lecture_analyzer/output/_ai_review_packet_exporter_impl.py)

### 6.4 Evaluation runs

Each evaluation run gets an immutable directory
`<evaluation_root>/<input_label>/runs/<run_label>/` holding `session.json`,
`review_packet.md`, `metrics.json` and an empty `ai_review.json` template. The
directory is created with `exist_ok=False` after the label is made unique with a
`_2`, `_3` suffix, so a run never overwrites another.

`metrics.json` records run identity, a privacy block, a **code snapshot**, the
full (secret-redacted) pipeline configuration, objective counters, the QA quality
distribution, coverage, speaker-check metrics, and timing.

The code snapshot is worth highlighting: besides the git commit and branch it
stores a SHA-256 over `git status --short`, the binary diff against HEAD, and the
bytes of every untracked non-ignored file. A dirty working tree therefore cannot
masquerade as an earlier clean run.

The run-level quality signal is a single diagnostic number:

```
useful_yield    = min(1, (high_quality_count + 0.4 * medium_quality_count) / 10)
high_risk_ratio = high_risk_count / max(1, available_candidate_count)
risk_adjustment = max(0, 1 - 0.8*avg_risk - 0.2*high_risk_ratio)

score = 0.45 * median_final_quality + 0.35 * useful_yield + 0.20 * risk_adjustment
band  = high >= 0.72 | medium >= 0.50 | low
```

It never filters or ranks candidates; it exists to compare runs.

The comparison tool (`lecture-analyzer-compare-evaluations`) reconstructs a
`cold_equivalent_runtime_seconds` alongside the observed runtime by borrowing
real cold durations for cache-reused stages from sibling runs, preferring a
reference from the same profile and segmentation mode. When no reference exists
the value is `null` and the missing stages are named.

> **Reference**
>
> - [_evaluation_run_exporter_impl.py](../src/lecture_analyzer/output/_evaluation_run_exporter_impl.py); `run_quality_signal` at [:600](../src/lecture_analyzer/output/_evaluation_run_exporter_impl.py#L600)
> - [_evaluation_comparison_exporter_impl.py](../src/lecture_analyzer/output/_evaluation_comparison_exporter_impl.py); protocol in [quality_evaluation.md](quality_evaluation.md)
> - Secret redaction: any config key containing `token`, `secret`, `password`, `api_key` or `auth`

**There is no automated accuracy metric anywhere in this repository.** Quality is
measured by external human or LLM review of the packet, pasted back as
`ai_review.json` keep/revise/reject decisions. Every number in `metrics.json` is
a distribution of the system's own self-assessment, not ground truth.

---

## 7. Configuration reference

### 7.1 QA-relevant fields

All in [_pipeline_config_impl.py](../src/lecture_analyzer/core/_pipeline_config_impl.py).

| Field | Default | Effect |
|---|---|---|
| `enable_qa_extraction` | `True` | Master switch |
| `min_question_score` | `0.45` | Question detection gate |
| `min_qa_confidence` | `0.4` | Universal emission gate |
| `max_question_length_chars` | `220` | Hard reject above this |
| `answer_search_window_units` | `3` | Local forward search width |
| `max_answer_units` | `3` | Max answer span length |
| `max_answer_duration_seconds` | `45.0` | Max answer span and gap |
| `allow_cross_segment_answer` | `True` | When False, segments hard-gate pairing |
| `question_context_expansion_enabled` | `True` | Allow anaphoric question expansion |
| `question_context_max_gap_seconds` | `3.5` | Locality bound for expansion and recomposition |
| `question_context_short_token_threshold` | `5` | Contextual-question size bound |
| `deferred_answer_search_enabled` | `True` | Far-window rescue |
| `deferred_answer_search_window_units` | `28` | Far-window width |
| `deferred_answer_search_min_question_score` | `0.62` | Didactic score required to trigger |
| `deferred_answer_search_local_score_threshold` | `0.38` | Below this, local is "not good enough" |
| `deferred_answer_search_min_signal_score` | `0.18` | Minimum recurrence signal |
| `qa_answer_search_strategy` | `semantic_retrieval` | `local_rule_based` or `semantic_retrieval` |
| `qa_semantic_retrieval_model_name` | `intfloat/multilingual-e5-base` | Bi-encoder |
| `qa_semantic_retrieval_top_k` / `_window_units` / `_min_similarity` | `4` / `8` / `0.2` | Retrieval bounds |
| `qa_answer_ranking_strategy` | `semantic_reranker` | `rule_based` or `semantic_reranker` |
| `qa_semantic_reranking_model_name` | `BAAI/bge-reranker-v2-m3` | Cross-encoder |
| `qa_semantic_reranking_max_candidates` / `_weight` | `8` / `0.7` | Rerank depth and blend |
| `qa_semantic_responsiveness_enabled` | `False` | Post-hoc rescoring |
| `qa_semantic_responsiveness_gate_*` | `False` / `0.45` / `0.12` | Optional penalty gate |
| `qa_speaker_check_enabled` | model path exists | Acoustic check |
| `qa_speaker_check_same_threshold` / `_same_full_penalty_threshold` / `_different_threshold` | `0.72` / `0.85` / `0.45` | Similarity bands |
| `qa_speaker_check_same_speaker_penalty` / `_different_speaker_bonus` | `0.25` / `0.04` | Adjustments |
| `qa_speaker_rescue_*` | `40` / `8` / `0.08` / `0.50` / `80` | Rescue caps |
| `utterance_max_gap_seconds` | `0.85` | Utterance split |
| `sentence_reconstruction_max_gap_seconds` | `2.5` | Sentence group break |
| `sentence_splitter_model_name` | `sat-3l-sm` | wtpsplit SaT model |
| `segmentation_mode` | `structural` | `structural`, `windowed`, `adaptive`, `both` |
| `pipeline_profile` | `current` | Bundle of the above |

Validation and clamping happen in `_normalize_configurable_values`: strategy
names are checked against their allowed sets, ratios are clamped to [0,1], the
sentence quality thresholds are forced into a monotone ladder, the retrieval
window is forced at least as wide as the local window, and the speaker-check
thresholds are ordered (`same_full_penalty >= same >= different`).

### 7.2 CLI flags that reach QA

| Flag | Field |
|---|---|
| `--pipeline-profile` | `pipeline_profile` (and everything it bundles) |
| `--segmentation-mode` | `segmentation_mode` (controls the QA fan-out) |
| `--enable-qa-speaker-check`, `--qa-speaker-model-path` | speaker check on/off and model |
| `--qa-speaker-min-span-seconds` | `qa_speaker_check_min_span_seconds` |
| `--qa-speaker-same-threshold`, `--qa-speaker-same-full-penalty-threshold`, `--qa-speaker-different-threshold`, `--qa-speaker-same-penalty` | similarity bands and penalty |
| `--qa-speaker-rescue-max-checks`, `--qa-speaker-rescue-max-candidates`, `--qa-speaker-rescue-confidence-margin`, `--qa-speaker-rescue-min-text-quality` | rescue caps |
| `--enable-qa-semantic-responsiveness`, `--qa-semantic-responsiveness-model`, `--qa-semantic-responsiveness-max-candidates` | responsiveness layer |
| `--enable-qa-semantic-responsiveness-gate`, `--qa-semantic-responsiveness-gate-min-score`, `--qa-semantic-responsiveness-gate-penalty` | responsiveness gate |

Indirectly upstream: `--disable-alignment` (removes word timings, and with them
utterances and sentences), `--enable-diarization` and the speaker-count hints,
`--from-scratch`, `--transcription-compute-type`, `--transcription-cache-dir`,
`--disable-transcription-cache`. Export side: `--export-ai-review-packet`,
`--export-evaluation-run`, `--evaluation-root`, `--evaluation-input-label`,
`--evaluation-run-label`.

**Known gap.** The core QA thresholds - `min_qa_confidence`,
`min_question_score`, `max_answer_units`, the deferred-search knobs, the
retrieval and reranking models - have **no CLI flags**. They are reachable only
by editing the dataclass defaults, choosing a profile, or calling
`apply_overrides` programmatically.
[scripts/run_evaluation_batch.py](../scripts/run_evaluation_batch.py) does not
work around this - it shells out to the CLI once per input and profile, so it is
limited to the same flag surface. Separately, `should_use_root_pipeline` in
[main.py:416](../src/lecture_analyzer/main.py#L416) decides between the real
pipeline and the smoke placeholder by comparing nine QA-speaker arguments against
hardcoded duplicates of their defaults, and `run_root_pipeline` repeats the same
nine literals a third time. Passing a flag at exactly its default value is
indistinguishable from not passing it, and changing any default requires editing
three places.

---

## 8. Known limitations

Collected from the inline notes above, with location and practical impact.

**1. Speaker roles are never assigned.**
`SpeakerRoleEstimator`
([_speaker_role_impl.py:10](../src/lecture_analyzer/analysis/_speaker_role_impl.py#L10))
is a placeholder that would label the first-seen speaker `teacher` and everyone
else `student` at confidence 0.25. It is never instantiated outside its own
module and one compatibility test. Consequently
`LectureSession.speaker_role_estimates` stays empty,
`Segment.estimated_speaker_roles` is populated only from
`TranscriptChunk.estimated_speaker_role` which nothing ever sets, and
`_resolve_speaker_role` returns `SpeakerRole.UNKNOWN` for essentially every
candidate. **Teacher versus student is not decided anywhere in the pipeline.**
The speaker layer is anonymous diarization ids plus the same-or-different ECAPA
check.

**2. Default configuration and recommended profile disagree.** The dataclass
defaults enable semantic retrieval and reranking; `quality_local`, the profile
the benchmark uses, disables both and enables a different set of gates. A run
without `--pipeline-profile` exercises an untested-against-benchmark path.

**3. Tier-2 gates are profile-scoped.** Every precision gate in section 5.8 is
inactive outside `quality_local`. A user comparing candidate counts across
profiles is comparing two different algorithms, not two settings.

**4. QA stage status can contradict the candidate list.** Status is derived from
the merged transcript's presence; the extractor consumes sentences.

**5. Two of three models can reach the network.** E5 and BGE load through
`SentenceTransformer(...)` and `from_pretrained(...)` and will download on first
use. Only the responsiveness model is pinned to `local_files_only=True`. This
sits awkwardly against the local-first framing in
[quality_evaluation.md](quality_evaluation.md).

**6. Transcription cache asymmetry.**
`transcription_cache_require_backend_match` defaults `False` while every other
cache layer verifies its config snapshot, so a warm run can reuse a transcript
from a different Whisper model. The text-only fallback can additionally
substitute a timing-free transcript without failing.

**7. Dataset-fitted strings in context summarization.** Five hardcoded English
summaries in `_summarize_context_text` tied to development-corpus content.

**8. No VAD, no loudness normalization.** Both are standard expectations for a
Whisper front end and neither is present.

**9. No automated accuracy metric.** All quality numbers are the system's
self-assessment. Ground truth comes only from external review. The frozen
benchmark (7 inputs, `quality_local` + `structural`, speaker check and rescue on)
is 46 emitted candidates at 52% keep, 30% revise, 17% reject - see
[quality_evaluation.md](quality_evaluation.md).

**10. Duplicated default literals in CLI routing.** Nine QA-speaker defaults are
written out in three separate places in `main.py`.
