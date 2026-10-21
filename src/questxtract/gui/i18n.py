"""The graphical app's words in English, Norsk (bokmål) and Italiano (issue #124).

One table, every key in every language: ``tests/test_gui_i18n.py`` fails when one is missing or
when a ``{placeholder}`` differs between languages. Messages from the CLI itself (the one-line
errors of ``process`` and ``transcribe``) are not here: the CLI owns them and they stay English.

Nothing here imports tkinter, so it is tested headless.
"""

from __future__ import annotations

__all__ = ["DEFAULT_LANGUAGE", "LANGUAGES", "has", "language", "set_language", "tr"]

#: The languages the app can show, by code, with each one's own name for the switch.
LANGUAGES: dict[str, str] = {"en": "English", "nb": "Norsk", "it": "Italiano"}
DEFAULT_LANGUAGE = "en"

_MESSAGES: dict[str, dict[str, str]] = {
    # --- the window's groups, buttons and labels ---
    "drop.help": {
        "en": "Drop an .mp4, .mp3 or .json file here",
        "nb": "Slipp en .mp4-, .mp3- eller .json-fil her",
        "it": "Trascina qui un file .mp4, .mp3 o .json",
    },
    "source.none": {
        "en": "No file chosen.",
        "nb": "Ingen fil valgt.",
        "it": "Nessun file scelto.",
    },
    "button.browse": {"en": "Browse...", "nb": "Bla gjennom...", "it": "Sfoglia..."},
    "group.source": {
        "en": "Recording or transcript",
        "nb": "Opptak eller transkripsjon",
        "it": "Registrazione o trascrizione",
    },
    "group.output": {"en": "Output", "nb": "Resultat", "it": "Output"},
    "group.options": {
        "en": "Advanced settings (a setting left at its default is not passed)",
        "nb": "Avanserte innstillinger (en innstilling som står på standard sendes ikke)",
        "it": "Impostazioni avanzate (un'impostazione al valore predefinito non viene passata)",
    },
    "button.advanced_show": {
        "en": "Advanced settings >>",
        "nb": "Avanserte innstillinger >>",
        "it": "Impostazioni avanzate >>",
    },
    "button.advanced_hide": {
        "en": "Advanced settings <<",
        "nb": "Avanserte innstillinger <<",
        "it": "Impostazioni avanzate <<",
    },
    "output.files": {
        "en": "Files: <name>.questions.json, <name>.transcript.json, <name>.quiz.json "
        "(beside the questions) and <name>.qti.zip, next to the input unless given below.",
        "nb": "Filer: <navn>.questions.json, <navn>.transcript.json, <navn>.quiz.json "
        "(ved siden av spørsmålene) og <navn>.qti.zip, ved siden av inndatafilen med mindre "
        "et annet er valgt under.",
        "it": "File: <nome>.questions.json, <nome>.transcript.json, <nome>.quiz.json "
        "(accanto alle domande) e <nome>.qti.zip, accanto al file di input se non indicato sotto.",
    },
    "output.questions": {"en": "Questions", "nb": "Spørsmål", "it": "Domande"},
    "output.transcript": {"en": "Transcript", "nb": "Transkripsjon", "it": "Trascrizione"},
    "output.quiz": {"en": "Quiz", "nb": "Quiz", "it": "Quiz"},
    "output.qti": {"en": "QTI package", "nb": "QTI-pakke", "it": "Pacchetto QTI"},
    "button.run": {"en": "Run", "nb": "Kjør", "it": "Avvia"},
    "button.cancel": {"en": "Cancel", "nb": "Avbryt", "it": "Annulla"},
    "button.open_folder": {"en": "Open folder", "nb": "Åpne mappe", "it": "Apri cartella"},
    "control.label": {
        "en": "{label} (default: {default})",
        "nb": "{label} (standard: {default})",
        "it": "{label} (predefinito: {default})",
    },
    "hint.next_input": {
        "en": "next to the input",
        "nb": "ved siden av inndata",
        "it": "accanto all'input",
    },
    "hint.on": {"en": "on", "nb": "på", "it": "attivo"},
    "hint.off": {"en": "off", "nb": "av", "it": "disattivo"},
    "hint.unset": {"en": "unset", "nb": "ikke satt", "it": "non impostato"},
    "tristate.default": {"en": "Default", "nb": "Standard", "it": "Predefinito"},
    "tristate.on": {"en": "On", "nb": "På", "it": "Sì"},
    "tristate.off": {"en": "Off", "nb": "Av", "it": "No"},
    # --- the options' own labels (the CLI's flags are not translated) ---
    "opt.config": {"en": "Settings file", "nb": "Innstillingsfil", "it": "File delle impostazioni"},
    "opt.output": {"en": "Output file", "nb": "Utdatafil", "it": "File di output"},
    "opt.model": {"en": "Whisper model", "nb": "Whisper-modell", "it": "Modello Whisper"},
    "opt.from_scratch": {
        "en": "Ignore cached results",
        "nb": "Ignorer mellomlagrede resultater",
        "it": "Ignora i risultati in cache",
    },
    "opt.threshold": {
        "en": "Threshold (0 to 1)",
        "nb": "Terskel (0 til 1)",
        "it": "Soglia (da 0 a 1)",
    },
    "opt.detector": {"en": "Detector", "nb": "Detektor", "it": "Rilevatore"},
    "opt.llm_model": {"en": "LLM model", "nb": "Språkmodell", "it": "Modello LLM"},
    "opt.llm_url": {"en": "LLM endpoint", "nb": "LLM-endepunkt", "it": "Endpoint LLM"},
    "opt.llm_reasoning": {"en": "Reasoning", "nb": "Resonnering", "it": "Ragionamento"},
    "opt.vision": {
        "en": "Read the board (vision)",
        "nb": "Les tavlen (syn)",
        "it": "Leggi la lavagna (visione)",
    },
    "opt.vision_model": {"en": "Vision model", "nb": "Synsmodell", "it": "Modello di visione"},
    "opt.vision_reader": {"en": "Board reader", "nb": "Tavleleser", "it": "Lettore della lavagna"},
    "opt.strictness": {"en": "Quiz questions", "nb": "Quiz-spørsmål", "it": "Domande del quiz"},
    "opt.options": {
        "en": "Options per quiz item (3 to 5)",
        "nb": "Alternativer per quiz-oppgave (3 til 5)",
        "it": "Opzioni per domanda del quiz (da 3 a 5)",
    },
    "opt.quiz_model": {"en": "Quiz model", "nb": "Quiz-modell", "it": "Modello del quiz"},
    # --- drop, dialogs and messages ---
    "drop.unavailable": {
        "en": "Drag and drop is not available on this system: use Browse.",
        "nb": "Dra og slipp er ikke tilgjengelig på dette systemet: bruk Bla gjennom.",
        "it": "Trascinamento non disponibile su questo sistema: usa Sfoglia.",
    },
    "drop.one_file": {
        "en": "One file at a time: the first one is used.",
        "nb": "Én fil om gangen: den første brukes.",
        "it": "Un file alla volta: viene usato il primo.",
    },
    "hint.drop_first": {
        "en": "Drop a recording or transcript here, or click Browse.",
        "nb": "Slipp et opptak eller en transkripsjon her, eller klikk Bla gjennom.",
        "it": "Trascina qui una registrazione o una trascrizione, oppure clicca Sfoglia.",
    },
    "dialog.source": {
        "en": "Choose a recording or transcript",
        "nb": "Velg et opptak eller en transkripsjon",
        "it": "Scegli una registrazione o una trascrizione",
    },
    "dialog.output": {
        "en": "Where to write the output",
        "nb": "Hvor resultatet skal skrives",
        "it": "Dove scrivere il risultato",
    },
    "dialog.settings": {
        "en": "Settings file",
        "nb": "Innstillingsfil",
        "it": "File delle impostazioni",
    },
    "filetype.recordings": {
        "en": "Recordings and transcripts",
        "nb": "Opptak og transkripsjoner",
        "it": "Registrazioni e trascrizioni",
    },
    "filetype.all": {"en": "All files", "nb": "Alle filer", "it": "Tutti i file"},
    "filetype.settings": {"en": "Settings", "nb": "Innstillinger", "it": "Impostazioni"},
    "status.long_run": {
        "en": "A run can take as long as the recording. Cancel stops it.",
        "nb": "En kjøring kan ta like lang tid som opptaket. Avbryt stopper den.",
        "it": "Una esecuzione può durare quanto la registrazione. Annulla la ferma.",
    },
    "status.starting": {"en": "Starting...", "nb": "Starter...", "it": "Avvio..."},
    "status.stopping": {
        "en": "Stopping after the current step...",
        "nb": "Stopper etter det pågående trinnet...",
        "it": "Arresto dopo il passaggio in corso...",
    },
    "status.closing": {
        "en": "Stopping after the current step, then the window closes...",
        "nb": "Stopper etter det pågående trinnet, deretter lukkes vinduet...",
        "it": "Arresto dopo il passaggio in corso, poi la finestra si chiude...",
    },
    "status.cancelled": {
        "en": "Cancelled. No partial file was written.",
        "nb": "Avbrutt. Ingen ufullstendig fil ble skrevet.",
        "it": "Annullato. Nessun file parziale è stato scritto.",
    },
    "status.written": {
        "en": "Written: {path}",
        "nb": "Skrevet: {path}",
        "it": "Scritto: {path}",
    },
    "status.open_failed": {
        "en": "Could not open the folder: {error}",
        "nb": "Kunne ikke åpne mappen: {error}",
        "it": "Impossibile aprire la cartella: {error}",
    },
    "status.unexpected": {
        "en": "unexpected {name}; see the terminal",
        "nb": "uventet {name}; se terminalen",
        "it": "errore imprevisto {name}; vedi il terminale",
    },
    "status.failed": {
        "en": "the run failed",
        "nb": "kjøringen mislyktes",
        "it": "l'esecuzione non è riuscita",
    },
    # --- progress: the pipeline's stage names, then the three ends of a run ---
    "stage.ingest": {
        "en": "Checking recording",
        "nb": "Sjekker opptaket",
        "it": "Verifica della registrazione",
    },
    "stage.transcribe": {
        "en": "Transcribing",
        "nb": "Transkriberer",
        "it": "Trascrizione in corso",
    },
    "stage.sentences": {
        "en": "Splitting sentences",
        "nb": "Deler opp setninger",
        "it": "Suddivisione in frasi",
    },
    "stage.segments": {
        "en": "Grouping segments",
        "nb": "Grupperer segmenter",
        "it": "Raggruppamento dei segmenti",
    },
    "stage.layout": {"en": "Reading layout", "nb": "Leser oppsett", "it": "Lettura del layout"},
    "stage.vision": {
        "en": "Reading the board",
        "nb": "Leser tavlen",
        "it": "Lettura della lavagna",
    },
    "stage.attach_visual": {
        "en": "Joining visuals",
        "nb": "Kobler sammen bilder",
        "it": "Unione dei contenuti visivi",
    },
    "stage.detect": {
        "en": "Detecting questions",
        "nb": "Finner spørsmål",
        "it": "Ricerca delle domande",
    },
    "stage.answer": {
        "en": "Finding answers",
        "nb": "Finner svar",
        "it": "Ricerca delle risposte",
    },
    "stage.judge": {
        "en": "LLM detection",
        "nb": "Språkmodellsøk",
        "it": "Rilevamento con LLM",
    },
    "stage.score": {"en": "Scoring", "nb": "Poengsetter", "it": "Valutazione"},
    "stage.quiz": {
        "en": "Writing the quiz",
        "nb": "Skriver quizen",
        "it": "Scrittura del quiz",
    },
    "stage.done": {"en": "Done", "nb": "Ferdig", "it": "Fatto"},
    "stage.cancelled": {"en": "Cancelled", "nb": "Avbrutt", "it": "Annullato"},
    "stage.stopped": {"en": "Stopped", "nb": "Stoppet", "it": "Interrotto"},
}

_language = DEFAULT_LANGUAGE


def language() -> str:
    """Return the code of the language the app is showing."""
    return _language


def set_language(code: str) -> None:
    """Show the app in the language ``code`` (a key of :data:`LANGUAGES`).

    Raises:
        ValueError: If the language is not one the app has.

    """
    global _language
    if code not in LANGUAGES:
        raise ValueError(f"no such language: {code!r}")
    _language = code


def has(key: str) -> bool:
    """Return whether the table has a message under ``key``."""
    return key in _MESSAGES


def tr(key: str, **values: object) -> str:
    """Return the message ``key`` in the current language, with its placeholders filled in.

    Raises:
        KeyError: If there is no message under ``key``.

    """
    translations = _MESSAGES[key]
    return translations.get(_language, translations[DEFAULT_LANGUAGE]).format(**values)
