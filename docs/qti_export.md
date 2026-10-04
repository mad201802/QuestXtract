# QTI export

QuestXtract builds a version-independent assessment model before serialization. The
shared model contains response declarations, interactions and options, answer keys,
outcome declarations, response processing, feedback, provenance and metadata. XML
details are isolated in `questxtract.qti_v21` and `questxtract.qti_v3`.

QuestXtract can export records in `questions.json` as a QTI 3.0
package:

```console
questxtract qti questions.json -o questions.qti.zip
```

Python callers can use `questxtract.qti_v21.write_qti_package` for QTI 2.1 or
`questxtract.qti_v3.write_qti_package` for QTI 3.0.

By default, the command exports every question, including `unjudged`, `keep`,
`revise`, and `reject`. It excludes only questions whose reviewer explicitly
set `review.status` to `deleted` or `false_positive`. Use `--only-keep` to
export only questions with `verdict == "keep"`; reviewer rejections remain
excluded in that mode.

The current profile supports:

- extended text interactions;
- single- and multiple-choice interactions;
- response declarations and optional answer keys;
- outcome declarations and response processing;
- feedback and QuestXtract provenance metadata.

`question_type` remains the linguistic QuestXtract classification
(`what`, `why`, `how`, `yes_no`, or `other`). It is stored in the package
metadata and is not interpreted as a QTI interaction type. In particular,
`yes_no` does not imply that a correct answer is known.

The extracted answer, confidence, provenance, and optional LaTeX are preserved
in `metadata/questxtract.json`. The normalized `correct_response` field, when
present, is also represented as a QTI `correctResponse`.

The generated ZIP contains:

```text
imsmanifest.xml
assessmentTest.xml
items/
  qti-q_<stable-id>.xml
metadata/
  questxtract.json
```

The public Python entry points are `export_qti`, `questions_to_qti`, and
`write_qti_package` in `questxtract.qti`.

Both versioned serializers produce the same complete package layout:

```text
imsmanifest.xml
assessmentTest.xml
items/
  qti-q_<stable-id>.xml
metadata/
  questxtract.json
```
