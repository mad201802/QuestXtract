# QTI 3.0 export

QuestXtract can export the approved records in `questions.json` as a QTI 3.0
package:

```console
questxtract qti questions.json -o questions.qti.zip
```

The command exports only records whose `verdict` is `keep`. Records with
`revise` or `reject` are not exported. `unjudged` records can be included
explicitly with `--include-unjudged`.

The initial profile supports one interaction only:

- `qti-extended-text-interaction`;
- one `string` response with `single` cardinality;
- no automatic scoring;
- no `correctResponse`, mapping, or response processing.

`question_type` remains the linguistic QuestXtract classification
(`what`, `why`, `how`, `yes_no`, or `other`). It is stored in the package
metadata and is not interpreted as a QTI interaction type. In particular,
`yes_no` does not imply that a correct answer is known.

The extracted answer, confidence, timestamps, provenance, and optional LaTeX
are preserved in `metadata/questxtract.json`; they are not represented as a
QTI `correctResponse`. This is intentional because the current extraction
pipeline does not provide a normalized answer key or scoring rules.

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
