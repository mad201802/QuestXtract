"""Tests for the direct export to ExerPlaza (issue #37), against a fake ExerPlaza on 127.0.0.1."""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
import urllib.parse
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from questxtract import (
    ExerPlazaExportError,
    ExerPlazaImport,
    ExerPlazaItem,
    export_exerplaza,
    validate_quiz,
)
from questxtract import exerplaza as exerplaza_module
from questxtract.cli import main
from questxtract.exerplaza import _base_url

MakeQuiz = Callable[..., dict[str, Any]]

EMAIL = "teacher@example.org"
PASSWORD = "Pa55-word-never-shown"  # the fake's; no output may ever hold it
UNITS = ("Sorting", "Stable sorts")

# ExerPlaza's login form as its server sends it (main 0a2f7fa7, Flask-WTF hidden_tag()), with
# the token's value replaced. The export depends on these field names.
LOGIN_FORM = """<form method="post" action="" class="p-6 space-y-5">
      <input id="csrf_token" name="csrf_token" type="hidden" value="{token}">
  <div class="flex flex-col ">
    <label
      for="email"
      class="block text-sm font-medium text-slate-700 "
    >
      Email
    </label>
  <input class="mt-1 block w-full rounded-lg bg-white px-3 py-2 text-gray-900 placeholder-gray-400 shadow-sm border border-blue-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-500" id="email" name="email" required type="text" value="">
</div>
  <div class="flex flex-col ">
    <label
      for="password"
      class="block text-sm font-medium text-slate-700 "
    >
      Password
    </label>
  <input class="mt-1 block w-full rounded-lg bg-white px-3 py-2 text-gray-900 placeholder-gray-400 shadow-sm border border-blue-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-500" id="password" name="password" required type="password" value="">
</div>
      <div>
        <input class="w-full inline-flex items-center justify-center rounded-lg bg-blue-600 px-4 py-2 text-white font-semibold shadow hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500" id="submit" name="submit" type="submit" value="Log In">
      </div>
    </form>"""  # noqa: E501 - as the server sends it

# The contract with ExerPlaza: the import request for _quiz(), the answer to it, and ExerPlaza's
# problem for that request with questions[1].options[2].text.text empty. ExerPlaza's tests hold a
# copy in tests/backend/test_files/questxtract_quiz_import_contract.json; change both together.
CONTRACT_QUERY = "course_session=42&content_unit=Sorting&content_unit=Stable+sorts"
CONTRACT_REQUEST = """{
  "schema_version": "1.0", "id": "quiz_lecture", "title": "Sorting", "language": "en",
  "status": "draft",
  "source": {"questions_path": "lecture.questions.json", "questions_schema_version": "2.1",
             "questions_hash": "9f2a"},
  "instructions": {"shuffle_questions": false, "shuffle_options": false, "show_feedback": true},
  "generation": {"method": "llm", "model": "gpt-5.6-terra", "prompt_version": "v0"},
  "questions": [
    {"id": "quiz_q_aaa",
     "prompt": {"text": "Which sort is stable: 2 < 3 & 4 > 1?", "origin": "spoken"},
     "interaction_type": "choice_single", "points": 2,
     "options": [{"id": "a", "text": {"text": "Merge sort", "origin": "generated"}},
                 {"id": "b", "text": {"text": "Quicksort", "origin": "generated"}},
                 {"id": "c", "text": {"text": "Heapsort", "origin": "generated"}}],
     "correct_response": {"option_ids": ["a"]},
     "solution": {
       "explanation": {"text": "A stable sort keeps equal keys in order.", "origin": "generated"},
       "option_explanations": {
         "a": {"text": "Merging never swaps equal keys.", "origin": "generated"},
         "b": {"text": "Partitioning can reorder equal keys.", "origin": "generated"},
         "c": {"text": "Sifting can reorder equal keys.", "origin": "generated"}}},
     "provenance": {"source_question_id": "q_aaa", "start": 1.5, "end": 4.0},
     "review": {"status": "pending", "note": null, "metadata": {}}},
    {"id": "quiz_q_bbb",
     "prompt": {"text": "Which sort is stable: 2 < 3 & 4 > 1?", "origin": "spoken"},
     "interaction_type": "choice_single", "points": 2,
     "options": [{"id": "a", "text": {"text": "Merge sort", "origin": "generated"}},
                 {"id": "b", "text": {"text": "Quicksort", "origin": "generated"}},
                 {"id": "c", "text": {"text": "Heapsort", "origin": "generated"}}],
     "correct_response": {"option_ids": ["a"]},
     "solution": {
       "explanation": {"text": "A stable sort keeps equal keys in order.", "origin": "generated"},
       "option_explanations": {
         "a": {"text": "Merging never swaps equal keys.", "origin": "generated"},
         "b": {"text": "Partitioning can reorder equal keys.", "origin": "generated"},
         "c": {"text": "Sifting can reorder equal keys.", "origin": "generated"}}},
     "provenance": {"source_question_id": "q_bbb", "start": 1.5, "end": 4.0},
     "review": {"status": "accepted", "note": null, "metadata": {}}}
  ]
}"""
CONTRACT_RESPONSE = """{
  "quiz": {"uid": 7, "outcome": "created"},
  "questions": [
    {"ref": "quiz_lecture/quiz_q_aaa", "uid": 123, "outcome": "created", "reason": null},
    {"ref": "quiz_lecture/quiz_q_bbb", "uid": 123, "outcome": "skipped",
     "reason": "same text as questions[0]"}
  ]
}"""
CONTRACT_PROBLEM = """{
  "title": "Invalid quiz", "status": 422, "detail": "1 error",
  "errors": [{"path": "questions[1].options[2].text.text",
              "detail": "String should have at least 1 character"}]
}"""


@dataclass(frozen=True, kw_only=True)
class Seen:
    """One request the fake received."""

    method: str
    path: str
    query: str
    headers: dict[str, str]
    body: bytes

    def form(self) -> dict[str, str]:
        return dict(urllib.parse.parse_qsl(self.body.decode("utf-8")))


class FakeExerPlaza(ThreadingHTTPServer):
    """ExerPlaza's login and the import endpoint of the contract, recording every request.

    A session is a cookie, as in Flask: ``GET /login`` sets one holding the form's CSRF token,
    and a login with that token, the email and the password replaces it with a logged-in one.
    The import works as ExerPlaza's: the quiz is the course session's quiz with the title, and an
    item whose text ExerPlaza already has, or an earlier item has, is skipped, never changed.
    """

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.seen: list[Seen] = []
        self.sessions: dict[str, dict[str, str | None]] = {}
        self.password = PASSWORD
        self.login_page = LOGIN_FORM
        self.wrong_token = False  # the page shows a token the session does not hold
        self.course_sessions = {42}
        self.quizzes: dict[tuple[int, str], int] = {}  # (course session, title) -> uid
        self.questions: dict[str, int] = {}  # text -> uid
        self.import_reply: tuple[int, dict[str, str], str] | None = None  # a canned answer
        self.delay = 0.0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"

    def routes(self) -> list[str]:
        return [f"{seen.method} {seen.path}" for seen in self.seen]

    def last_import(self) -> Seen:
        return next(s for s in reversed(self.seen) if s.path == "/api/v1/quizzes/import")


class _Handler(BaseHTTPRequestHandler):
    server: FakeExerPlaza

    def log_message(self, format: str, *args: Any) -> None:
        """Keep the server's access log out of the captured output."""

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def _handle(self) -> None:
        fake = self.server
        split = urllib.parse.urlsplit(self.path)
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        headers = {name.lower(): value for name, value in self.headers.items()}
        fake.seen.append(
            Seen(
                method=self.command, path=split.path, query=split.query, headers=headers, body=body
            )
        )
        time.sleep(fake.delay)
        cookie = SimpleCookie(self.headers.get("Cookie") or "")
        session_id = cookie["session"].value if "session" in cookie else ""
        session = fake.sessions.get(session_id)
        route = f"{self.command} {split.path}"
        if route == "GET /login":
            if session is None:
                session_id = f"s{len(fake.sessions)}"
                session = fake.sessions[session_id] = {"csrf": f"token-{session_id}", "user": None}
            token = "forged" if fake.wrong_token else str(session["csrf"])
            page = f"<html><body>{fake.login_page.replace('{token}', token)}</body></html>"
            self._reply(200, {"Content-Type": "text/html; charset=utf-8"}, page, session_id)
        elif route == "POST /login":
            form = dict(urllib.parse.parse_qsl(body.decode("utf-8")))
            if session is None or form.get("csrf_token") != session["csrf"]:
                self._reply(200, {"Content-Type": "text/html"}, "<form>again</form>")
            elif form.get("email") == EMAIL and form.get("password") == fake.password:
                new_id = f"s{len(fake.sessions)}"
                fake.sessions[new_id] = {"csrf": session["csrf"], "user": EMAIL}
                self._reply(302, {"Location": "/course/"}, "", new_id)
            else:
                self._reply(302, {"Location": "/login"}, "")
        elif route == "POST /api/v1/quizzes/import":
            if fake.import_reply is not None:
                self._reply(*fake.import_reply)
            elif session is None or session["user"] is None:
                self._problem(401, "Unauthorized", "log in first")
            else:
                self._import(split.query, body)
        else:
            self._reply(404, {"Content-Type": "text/html"}, "<h1>Not Found</h1>")

    def _import(self, query: str, body: bytes) -> None:
        fake = self.server
        if self.headers.get("Content-Type") != "application/json":
            self._problem(415, "Unsupported Media Type", "expected application/json")
            return
        course = dict(urllib.parse.parse_qsl(query)).get("course_session", "")
        if not course.isdigit() or int(course) not in fake.course_sessions:
            self._problem(404, "Not Found", f"course session {course} does not exist")
            return
        document = json.loads(body)
        key = (int(course), document["title"])
        quiz_outcome = "updated" if key in fake.quizzes else "created"
        quiz_uid = fake.quizzes.setdefault(key, 7 + len(fake.quizzes))
        answer: list[dict[str, Any]] = []
        first_with_text: dict[str, int] = {}
        for index, item in enumerate(document["questions"]):
            text = item["prompt"]["text"]
            reason = None
            if text in first_with_text:
                reason = f"same text as questions[{first_with_text[text]}]"
            elif text in fake.questions:
                reason = "a question with the same text already exists"
            else:
                fake.questions[text] = 123 + len(fake.questions)
            first_with_text.setdefault(text, index)
            answer.append(
                {
                    "ref": f"{document['id']}/{item['id']}",
                    "uid": fake.questions[text],
                    "outcome": "created" if reason is None else "skipped",
                    "reason": reason,
                }
            )
        reply = {"quiz": {"uid": quiz_uid, "outcome": quiz_outcome}, "questions": answer}
        self._reply(200, {"Content-Type": "application/json"}, json.dumps(reply))

    def _problem(self, status: int, title: str, detail: str) -> None:
        problem = {"title": title, "status": status, "detail": detail}
        self._reply(status, {"Content-Type": "application/problem+json"}, json.dumps(problem))

    def _reply(self, status: int, headers: dict[str, str], body: str, session: str = "") -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        if session:
            self.send_header("Set-Cookie", f"session={session}; HttpOnly; Path=/")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(autouse=True)
def _loopback_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that would connect anywhere but 127.0.0.1, and keep proxies out."""
    connect = socket.create_connection

    def guarded(address: tuple[str, int], *args: Any, **kwargs: Any) -> socket.socket:
        assert address[0] == "127.0.0.1", f"a test connected to {address[0]}"
        return connect(address, *args, **kwargs)

    monkeypatch.setattr(socket, "create_connection", guarded)
    for name in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1")  # also stops urllib reading the system's proxies


@pytest.fixture
def exerplaza(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeExerPlaza]:
    """A running fake ExerPlaza, and the teacher's login in the environment."""
    fake = FakeExerPlaza()
    thread = threading.Thread(target=fake.serve_forever, args=(0.01,), daemon=True)
    thread.start()
    monkeypatch.setenv("QUESTXTRACT_EXERPLAZA_EMAIL", EMAIL)
    monkeypatch.setenv("QUESTXTRACT_EXERPLAZA_PASSWORD", PASSWORD)
    yield fake
    fake.shutdown()
    fake.server_close()


def _quiz(make_quiz: MakeQuiz) -> dict[str, Any]:
    """A pending, an accepted and a rejected item."""
    quiz = make_quiz("q_aaa", "q_bbb", "q_ccc")
    quiz["questions"][1]["review"]["status"] = "accepted"
    quiz["questions"][2]["review"]["status"] = "rejected"
    return quiz


def _file(tmp_path: Path, quiz: dict[str, Any]) -> Path:
    path = tmp_path / "lecture.quiz.json"
    path.write_text(json.dumps(quiz), encoding="utf-8")
    return path


def _args(url: str, quiz_path: Path, course_session: str = "42") -> list[str]:
    units = [arg for unit in UNITS for arg in ("--content-unit", unit)]
    return ["exerplaza", str(quiz_path), "--url", url, "--course-session", course_session, *units]


def _cli(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, str]:
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def _export(url: str, quiz_path: Path) -> ExerPlazaImport:
    return export_exerplaza(quiz_path, url=url, course_session=42, content_units=UNITS)


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_a_quiz_goes_in_one_login_and_one_import(
    tmp_path: Path,
    make_quiz: MakeQuiz,
    exerplaza: FakeExerPlaza,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, output = _cli(capsys, _args(exerplaza.url, _file(tmp_path, _quiz(make_quiz))))

    assert code == 0
    assert exerplaza.routes() == ["GET /login", "POST /login", "POST /api/v1/quizzes/import"]
    assert output == (
        "ExerPlaza quiz 7 (created): 1 created, 0 updated, 1 skipped\n"
        "  skipped quiz_lecture/quiz_q_bbb: same text as questions[0]\n"
    )


def test_the_body_is_the_quiz_without_rejected_items(
    tmp_path: Path, make_quiz: MakeQuiz, exerplaza: FakeExerPlaza
) -> None:
    _export(exerplaza.url, _file(tmp_path, _quiz(make_quiz)))

    sent = exerplaza.last_import()
    body = json.loads(sent.body)
    validate_quiz(body)
    assert [item["id"] for item in body["questions"]] == ["quiz_q_aaa", "quiz_q_bbb"]
    assert body == json.loads(CONTRACT_REQUEST)
    assert sent.query == CONTRACT_QUERY
    assert sent.headers["content-type"] == "application/json"


def test_the_contract_answer_is_read_item_by_item(
    tmp_path: Path, make_quiz: MakeQuiz, exerplaza: FakeExerPlaza
) -> None:
    exerplaza.import_reply = (200, {"Content-Type": "application/json"}, CONTRACT_RESPONSE)

    result = _export(exerplaza.url, _file(tmp_path, _quiz(make_quiz)))

    assert result == ExerPlazaImport(
        quiz_uid=7,
        quiz_outcome="created",
        items=(
            ExerPlazaItem(ref="quiz_lecture/quiz_q_aaa", uid=123, outcome="created", reason=None),
            ExerPlazaItem(
                ref="quiz_lecture/quiz_q_bbb",
                uid=123,
                outcome="skipped",
                reason="same text as questions[0]",
            ),
        ),
    )


def test_the_login_sends_the_pages_token_with_its_cookie(
    tmp_path: Path, make_quiz: MakeQuiz, exerplaza: FakeExerPlaza
) -> None:
    _export(exerplaza.url, _file(tmp_path, _quiz(make_quiz)))

    page, login, upload = exerplaza.seen
    assert "cookie" not in page.headers
    assert login.headers["cookie"] == "session=s0"
    assert login.form() == {"csrf_token": "token-s0", "email": EMAIL, "password": PASSWORD}
    assert upload.headers["cookie"] == "session=s1"  # the logged-in session, not the first


def test_a_login_page_without_a_token_fails_and_sends_no_password(
    tmp_path: Path, make_quiz: MakeQuiz, exerplaza: FakeExerPlaza
) -> None:
    exerplaza.login_page = LOGIN_FORM.replace('name="csrf_token"', 'name="token"')

    with pytest.raises(ExerPlazaExportError, match="GET http://127.0.0.1:.*/login: .*csrf_token"):
        _export(exerplaza.url, _file(tmp_path, _quiz(make_quiz)))
    assert exerplaza.routes() == ["GET /login"]


@pytest.mark.parametrize("missing", ["--course-session", "--content-unit"])
def test_a_missing_option_is_a_usage_error(
    tmp_path: Path, make_quiz: MakeQuiz, exerplaza: FakeExerPlaza, missing: str
) -> None:
    argv = _args(exerplaza.url, _file(tmp_path, _quiz(make_quiz)))
    while missing in argv:
        del argv[argv.index(missing) : argv.index(missing) + 2]

    with pytest.raises(SystemExit) as stopped:
        main(argv)
    assert stopped.value.code == 2
    assert exerplaza.seen == []


def test_an_unknown_course_session_fails_with_exerplazas_detail(
    tmp_path: Path,
    make_quiz: MakeQuiz,
    exerplaza: FakeExerPlaza,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, output = _cli(capsys, _args(exerplaza.url, _file(tmp_path, _quiz(make_quiz)), "99"))

    assert code == 1
    assert output.startswith("error: exerplaza: POST http://127.0.0.1:")
    assert "refused the import: HTTP 404 Not Found: course session 99 does not exist" in output
    assert exerplaza.questions == {}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda q: q["questions"][0].update(correct_response={"option_ids": ["z"]}), r"\[0\]"),
        (lambda q: q["questions"][1]["solution"]["option_explanations"].pop("b"), r"\[1\]"),
        (lambda q: q["questions"][0].update(interaction_type="choice_multiple"), "interaction"),
        (lambda q: q.update(schema_version="2.0"), "schema"),
    ],
)
def test_an_invalid_quiz_fails_naming_the_field_and_sends_nothing(
    tmp_path: Path,
    make_quiz: MakeQuiz,
    exerplaza: FakeExerPlaza,
    change: Callable[[dict[str, Any]], object],
    message: str,
) -> None:
    quiz = _quiz(make_quiz)
    change(quiz)

    with pytest.raises(ExerPlazaExportError, match=message):
        _export(exerplaza.url, _file(tmp_path, quiz))
    assert exerplaza.seen == []


def test_a_quiz_with_no_item_left_fails_and_sends_nothing(
    tmp_path: Path, make_quiz: MakeQuiz, exerplaza: FakeExerPlaza
) -> None:
    quiz = make_quiz("q_aaa")
    quiz["questions"][0]["review"]["status"] = "rejected"

    with pytest.raises(ExerPlazaExportError, match="no question to export"):
        _export(exerplaza.url, _file(tmp_path, quiz))
    assert exerplaza.seen == []


@pytest.mark.parametrize("unset", ["QUESTXTRACT_EXERPLAZA_EMAIL", "QUESTXTRACT_EXERPLAZA_PASSWORD"])
def test_a_missing_credential_fails_naming_it_and_sends_nothing(
    tmp_path: Path,
    make_quiz: MakeQuiz,
    exerplaza: FakeExerPlaza,
    monkeypatch: pytest.MonkeyPatch,
    unset: str,
) -> None:
    monkeypatch.delenv(unset)

    with pytest.raises(ExerPlazaExportError, match=f"set {unset} in the environment or in .env"):
        _export(exerplaza.url, _file(tmp_path, _quiz(make_quiz)))
    assert exerplaza.seen == []


def test_the_credentials_can_come_from_dotenv(
    tmp_path: Path, make_quiz: MakeQuiz, exerplaza: FakeExerPlaza, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("QUESTXTRACT_EXERPLAZA_EMAIL")
    monkeypatch.delenv("QUESTXTRACT_EXERPLAZA_PASSWORD")
    (tmp_path / ".env").write_text(
        f'QUESTXTRACT_EXERPLAZA_EMAIL={EMAIL}\nQUESTXTRACT_EXERPLAZA_PASSWORD="{PASSWORD}"\n',
        encoding="utf-8",
    )

    result = _export(exerplaza.url, _file(tmp_path, _quiz(make_quiz)))

    assert [item.outcome for item in result.items] == ["created", "skipped"]


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("http://exerplaza.example.org", "not encrypted"),
        ("https://user:secret@exerplaza.example.org", "without credentials"),
        ("https://exerplaza.example.org/?next=x", "without credentials, query"),
        ("exerplaza.example.org", "expected ExerPlaza's address"),
        ("https://exerplaza.example.org:port", "expected ExerPlaza's address"),
    ],
)
def test_an_unsafe_or_malformed_url_fails_before_any_request(
    tmp_path: Path, make_quiz: MakeQuiz, url: str, message: str
) -> None:
    with pytest.raises(ExerPlazaExportError, match=message) as failed:
        _export(url, _file(tmp_path, _quiz(make_quiz)))
    assert "secret" not in str(failed.value)


@pytest.mark.parametrize(
    ("url", "base"),
    [
        ("https://exerplaza.example.org/", "https://exerplaza.example.org"),
        ("https://example.org/exerplaza/", "https://example.org/exerplaza"),
        ("http://localhost:54321", "http://localhost:54321"),
        ("http://127.0.0.1:5000/", "http://127.0.0.1:5000"),
        ("http://[::1]:5000", "http://[::1]:5000"),
    ],
)
def test_https_or_a_local_http_address_is_accepted(url: str, base: str) -> None:
    assert _base_url(url) == base


@pytest.mark.parametrize(
    ("units", "course_session", "message"),
    [
        ((), 42, "at least one"),
        (("Sorting", " "), 42, "non-empty"),
        (("Sorting\nHeaps",), 42, "one line"),
        (UNITS, 0, "positive"),
    ],
)
def test_bad_arguments_fail_before_any_request(
    tmp_path: Path,
    make_quiz: MakeQuiz,
    exerplaza: FakeExerPlaza,
    units: tuple[str, ...],
    course_session: int,
    message: str,
) -> None:
    with pytest.raises(ExerPlazaExportError, match=message):
        export_exerplaza(
            _file(tmp_path, _quiz(make_quiz)),
            url=exerplaza.url,
            course_session=course_session,
            content_units=units,
        )
    assert exerplaza.seen == []


def _wrong_password(fake: FakeExerPlaza) -> None:
    fake.password = "something else"


def _form_refused(fake: FakeExerPlaza) -> None:
    fake.wrong_token = True


def _invalid_quiz(fake: FakeExerPlaza) -> None:
    fake.import_reply = (422, {"Content-Type": "application/problem+json"}, CONTRACT_PROBLEM)


def _redirected(fake: FakeExerPlaza) -> None:
    fake.import_reply = (302, {"Location": "/login"}, "")


def _no_endpoint(fake: FakeExerPlaza) -> None:
    fake.import_reply = (404, {"Content-Type": "text/html"}, "<h1>Not Found</h1>")


def _rate_limited(fake: FakeExerPlaza) -> None:
    error = '{"success": false, "error": "Rate limit exceeded", "retry_after": 60}'
    fake.import_reply = (429, {"Content-Type": "application/json"}, error)


def _out_of_shape(fake: FakeExerPlaza) -> None:
    answer = '{"quiz": {"uid": 7, "outcome": "created"}, "questions": [{"ref": "x", "uid": "1"}]}'
    fake.import_reply = (200, {"Content-Type": "application/json"}, answer)


def _not_json(fake: FakeExerPlaza) -> None:
    fake.import_reply = (200, {"Content-Type": "text/html"}, "<html>ok</html>")


FAILURES: dict[str, tuple[Callable[[FakeExerPlaza], None], str]] = {
    "refused login": (_wrong_password, "refused the login: wrong email or password"),
    "refused form": (_form_refused, "showed the login form again"),
    "invalid quiz": (
        _invalid_quiz,
        "refused the import: HTTP 422 Invalid quiz: 1 error\n"
        "  questions[1].options[2].text.text: String should have at least 1 character\n",
    ),
    "redirect": (_redirected, "redirected to /login: the session was not accepted"),
    "no endpoint": (_no_endpoint, "HTTP 404, not a JSON answer: is this ExerPlaza"),
    "rate limited": (_rate_limited, "HTTP 429 Rate limit exceeded"),
    "answer out of shape": (
        _out_of_shape,
        "unexpected answer from ExerPlaza: questions[0].outcome",
    ),
    "answer not json": (_not_json, "expected a JSON answer"),
}


@pytest.mark.parametrize("failure", list(FAILURES))
def test_a_failure_exits_with_1_and_says_what_went_wrong(
    tmp_path: Path,
    make_quiz: MakeQuiz,
    exerplaza: FakeExerPlaza,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    arrange, message = FAILURES[failure]
    arrange(exerplaza)

    code, output = _cli(capsys, _args(exerplaza.url, _file(tmp_path, _quiz(make_quiz))))

    assert code == 1
    assert output.startswith("error: exerplaza: ")
    assert message in output


def test_an_unreachable_server_exits_with_1(
    tmp_path: Path,
    make_quiz: MakeQuiz,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QUESTXTRACT_EXERPLAZA_EMAIL", EMAIL)
    monkeypatch.setenv("QUESTXTRACT_EXERPLAZA_PASSWORD", PASSWORD)
    url = f"http://127.0.0.1:{_closed_port()}"

    code, output = _cli(capsys, _args(url, _file(tmp_path, _quiz(make_quiz))))

    assert code == 1
    assert f"error: exerplaza: GET {url}/login: cannot reach ExerPlaza" in output


def test_a_server_that_does_not_answer_times_out(
    tmp_path: Path, make_quiz: MakeQuiz, exerplaza: FakeExerPlaza, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exerplaza_module, "_TIMEOUT_SECONDS", 0.2)
    exerplaza.delay = 1.0

    with pytest.raises(ExerPlazaExportError, match="/login: no answer within 0.2 s"):
        _export(exerplaza.url, _file(tmp_path, _quiz(make_quiz)))


def test_sending_twice_skips_what_exerplaza_already_has(
    tmp_path: Path,
    make_quiz: MakeQuiz,
    exerplaza: FakeExerPlaza,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = _args(exerplaza.url, _file(tmp_path, _quiz(make_quiz)))
    assert _cli(capsys, argv)[0] == 0

    code, output = _cli(capsys, argv)

    assert code == 0
    assert output == (
        "ExerPlaza quiz 7 (updated): 0 created, 0 updated, 2 skipped\n"
        "  skipped quiz_lecture/quiz_q_aaa: a question with the same text already exists\n"
        "  skipped quiz_lecture/quiz_q_bbb: same text as questions[0]\n"
    )
    assert len(exerplaza.questions) == 1


def test_control_characters_from_the_server_are_not_printed(
    tmp_path: Path, make_quiz: MakeQuiz, exerplaza: FakeExerPlaza
) -> None:
    problem = {"title": "Bad\x1b[31m", "status": 422, "errors": [{"path": "id", "detail": "\x07"}]}
    exerplaza.import_reply = (422, {"Content-Type": "application/json"}, json.dumps(problem))

    with pytest.raises(ExerPlazaExportError) as failed:
        _export(exerplaza.url, _file(tmp_path, _quiz(make_quiz)))
    assert "HTTP 422 Bad?[31m\n  id: ?" in str(failed.value)


@pytest.mark.parametrize("failure", ["success", "unreachable", *FAILURES])
def test_the_password_never_shows(
    tmp_path: Path,
    make_quiz: MakeQuiz,
    exerplaza: FakeExerPlaza,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    failure: str,
) -> None:
    caplog.set_level(logging.DEBUG)
    url = f"http://127.0.0.1:{_closed_port()}" if failure == "unreachable" else exerplaza.url
    if failure in FAILURES:
        FAILURES[failure][0](exerplaza)
    quiz_path = _file(tmp_path, _quiz(make_quiz))

    _, output = _cli(capsys, _args(url, quiz_path))
    try:
        _export(url, quiz_path)
    except ExerPlazaExportError as exc:
        chain: BaseException | None = exc
        while chain is not None:
            output += f"{chain!s} {chain!r}"
            chain = chain.__cause__ or chain.__context__

    assert PASSWORD not in output + caplog.text
