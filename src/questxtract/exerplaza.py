"""Direct export to ExerPlaza: a ``quiz.json`` sent to its import endpoint (12.6).

ExerPlaza authenticates with Flask sessions only, so this logs in through its web form as a
browser does: ``GET /login`` gives the form's ``csrf_token`` and the session cookie it belongs
to, ``POST /login`` sends them with the teacher's email and password, and the session cookie that
comes back goes with one ``POST /api/v1/quizzes/import``. The cookie lives in memory for that one
call. ExerPlaza checks CSRF per form, not for the whole site, so the JSON import needs only the
cookie.

The body is the ``quiz.json`` without its rejected items, still a valid ``quiz.json``.
ExerPlaza saves the quiz in one transaction and answers, per item, whether it created or
skipped it, and why it skipped it. Redirects are never followed: on the login a redirect is the
answer, and on the import it is an error.
"""

from __future__ import annotations

import http.client
import http.cookiejar
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from email.message import Message
from html.parser import HTMLParser
from pathlib import Path
from typing import IO, Any

from .config import EXERPLAZA_EMAIL_ENV, EXERPLAZA_PASSWORD_ENV, exerplaza_credentials
from .models import SchemaError
from .qti import QtiExportError, quiz_to_qti
from .quiz_schema import read_quiz, validate_quiz

LOGIN_PATH = "/login"
IMPORT_PATH = "/api/v1/quizzes/import"
_TIMEOUT_SECONDS = 30
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}  # where plain http is allowed
_QUIZ_OUTCOMES = {"created", "updated"}
_ITEM_OUTCOMES = {"created", "updated", "skipped"}


class ExerPlazaExportError(SchemaError):
    """Raised when a quiz cannot be sent to ExerPlaza, or ExerPlaza refuses it."""


@dataclass(frozen=True, kw_only=True)
class ExerPlazaItem:
    """What ExerPlaza did with one quiz item.

    ``ref`` is ``<quiz id>/<item id>``, naming the item, and ``uid`` the id of its question in
    ExerPlaza: for a skipped item, the question ExerPlaza already had. ``outcome`` is ``created``,
    ``updated`` or ``skipped``, though ExerPlaza never updates a question today; ``reason`` says
    why an item was skipped (``a question with the same text already exists``), else it is
    ``None``.
    """

    ref: str
    uid: int
    outcome: str
    reason: str | None


@dataclass(frozen=True, kw_only=True)
class ExerPlazaImport:
    """ExerPlaza's answer to an import: its quiz, ``created`` or ``updated``, and every item."""

    quiz_uid: int
    quiz_outcome: str
    items: tuple[ExerPlazaItem, ...]


@dataclass(frozen=True, kw_only=True)
class _Reply:
    status: int
    headers: Message
    body: bytes


def export_exerplaza(
    quiz_path: str | Path, *, url: str, course_session: int, content_units: Sequence[str]
) -> ExerPlazaImport:
    """Send a ``quiz.json`` to ExerPlaza, logged in as the teacher whose credentials are set.

    ExerPlaza adds one multiple-choice question per item to the course session's quiz with the
    quiz's title, creating that quiz if there is none. Items a reviewer marked ``rejected`` are
    not sent; ``pending`` items are visible only to their author in ExerPlaza, ``accepted`` ones
    to everybody. An item whose text ExerPlaza already has, in that quiz or among the teacher's
    questions, is not saved again: it is reported ``skipped`` and its question is left as it is.
    Every check that needs no network runs before the first request.

    The login is ExerPlaza's own form, with the email and password from
    ``QUESTXTRACT_EXERPLAZA_EMAIL`` and ``QUESTXTRACT_EXERPLAZA_PASSWORD``, in the environment or
    in ``.env`` in the working directory. They are never printed, logged or stored.

    Args:
        quiz_path: A ``quiz.json``, schema version 1.x.
        url: ExerPlaza's address, such as ``https://exerplaza.example.org``. Plain ``http`` is
            refused except for ``localhost``, since the password goes with the login.
        course_session: The id of the course session the quiz belongs to: the number in the
            address of its page, ``/course/session/<id>``. You must teach it.
        content_units: ExerPlaza content units for every item, at least one, each spelled
            exactly as in ExerPlaza: only an exact match links to an existing unit.

    Returns:
        ExerPlaza's quiz and what it did with each item, in the order it answered.

    Raises:
        ExerPlazaExportError: If an argument is wrong, the quiz file cannot be read or breaks a
            rule of ``quiz.json`` (naming the field), no item is left to send, or the email or
            password is not set (naming the variable), all before any request; if ExerPlaza
            cannot be reached, refuses the login or the import (with the explanation it sent,
            item paths included), or answers in a shape this module does not know.

    """
    base = _base_url(url)
    session = _course_session(course_session)
    units = _content_units(content_units)
    body = _body(quiz_path)
    email, password = _credentials()
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar), _NoRedirect())
    _login(opener, base, email, password)
    return _import(opener, base, session, units, body)


def _base_url(url: str) -> str:
    """Return ``url`` without a trailing slash, after the checks that keep the password safe."""
    parts = urllib.parse.urlsplit(url.strip())
    try:
        host = (parts.hostname or "").lower()
        parts.port  # noqa: B018 - raises ValueError for a port that is not a number
    except ValueError:
        host = ""
    if parts.scheme not in {"http", "https"} or not host:
        raise ExerPlazaExportError(
            "url: expected ExerPlaza's address, such as https://exerplaza.example.org"
        )
    if parts.username is not None or parts.query or parts.fragment:
        raise ExerPlazaExportError("url: expected an address without credentials, query or #")
    if parts.scheme == "http" and host not in _LOCAL_HOSTS:
        raise ExerPlazaExportError(
            f"url: http://{host} is not encrypted and the login sends your password; "
            "use https (plain http is allowed for localhost only)"
        )
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def _course_session(course_session: int) -> int:
    if isinstance(course_session, bool) or not isinstance(course_session, int):
        raise ExerPlazaExportError(f"course session: expected a number, got {course_session!r}")
    if course_session < 1:
        raise ExerPlazaExportError(f"course session: expected a positive id, got {course_session}")
    return course_session


def _content_units(content_units: Sequence[str]) -> tuple[str, ...]:
    """Return the units, stripped, each checked: non-empty and on one line."""
    if isinstance(content_units, str) or not content_units:
        raise ExerPlazaExportError("content units: expected at least one name")
    units: list[str] = []
    for unit in content_units:
        name = unit.strip() if isinstance(unit, str) else ""
        if not name:
            raise ExerPlazaExportError(f"content unit {unit!r}: expected a non-empty name")
        if len(name.splitlines()) > 1:
            raise ExerPlazaExportError(f"content unit {unit!r}: expected one line")
        units.append(name)
    return tuple(units)


def _body(quiz_path: str | Path) -> dict[str, Any]:
    """Read the quiz and return it without its rejected items, still a valid ``quiz.json``.

    The items kept are those of :func:`questxtract.qti.quiz_to_qti`, so ``qti``, ``tex`` and
    this export send the same items, after the same field checks.
    """
    try:
        document = read_quiz(quiz_path)
        kept = {item.identifier for item in quiz_to_qti(document).items}
        body = dict(document)
        body["questions"] = [
            item
            for item in document["questions"]
            if isinstance(item, Mapping) and item.get("id") in kept
        ]
        validate_quiz(body)
    except OSError as exc:
        raise ExerPlazaExportError(f"{quiz_path}: {exc.strerror or exc}") from exc
    except (SchemaError, QtiExportError) as exc:
        raise ExerPlazaExportError(str(exc)) from exc
    if not kept:
        raise ExerPlazaExportError(
            "no question to export: every item is rejected or the quiz is empty"
        )
    return body


def _credentials() -> tuple[str, str]:
    try:
        email, password = exerplaza_credentials()
    except ValueError as exc:  # a .env line it cannot read; the message holds no value
        raise ExerPlazaExportError(str(exc)) from exc
    if not email:
        raise _unset(EXERPLAZA_EMAIL_ENV)
    if not password:
        raise _unset(EXERPLAZA_PASSWORD_ENV)
    return email, password


def _unset(name: str) -> ExerPlazaExportError:
    return ExerPlazaExportError(
        f"set {name} in the environment or in .env in this folder: "
        "the ExerPlaza login of the teacher who imports"
    )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Hand every redirect back as an answer: the login reads it, the import refuses it."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: http.client.HTTPMessage,
        newurl: str,
    ) -> None:
        """Follow no redirect."""
        return None


class _CsrfToken(HTMLParser):
    """Find the value of the first ``<input name="csrf_token">``, as Flask-WTF writes it."""

    def __init__(self) -> None:
        super().__init__()
        self.token: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if self.token is None and tag == "input" and values.get("name") == "csrf_token":
            self.token = values.get("value") or None


def _login(opener: urllib.request.OpenerDirector, base: str, email: str, password: str) -> None:
    """Log in through ExerPlaza's form; the opener's cookie jar keeps the session."""
    url = base + LOGIN_PATH
    page = _send(opener, urllib.request.Request(url), f"GET {url}")  # noqa: S310 - see _base_url
    if page.status != 200:
        raise ExerPlazaExportError(f"GET {url}: expected ExerPlaza's login page, got {_said(page)}")
    parser = _CsrfToken()
    parser.feed(page.body.decode("utf-8", "replace"))
    if parser.token is None:
        raise ExerPlazaExportError(
            f"GET {url}: the page has no csrf_token field; is this ExerPlaza's login page?"
        )
    form = {"csrf_token": parser.token, "email": email, "password": password}
    request = urllib.request.Request(  # noqa: S310 - _base_url allows http(s) only
        url,
        data=urllib.parse.urlencode(form).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    reply = _send(opener, request, f"POST {url}")
    location = reply.headers.get("Location")
    if 300 <= reply.status < 400 and location:
        target = urllib.parse.urlsplit(urllib.parse.urljoin(url, location)).path.rstrip("/")
        if target != urllib.parse.urlsplit(url).path.rstrip("/"):
            return  # logged in: ExerPlaza sends a new user to their courses
        raise ExerPlazaExportError(
            f"POST {url}: ExerPlaza refused the login: wrong email or password, or a disabled "
            f"account (check {EXERPLAZA_EMAIL_ENV} and {EXERPLAZA_PASSWORD_ENV})"
        )
    if reply.status == 200:
        raise ExerPlazaExportError(
            f"POST {url}: ExerPlaza showed the login form again: it refused the form, for "
            f"example because {EXERPLAZA_EMAIL_ENV} is not an email address"
        )
    raise ExerPlazaExportError(
        f"POST {url}: expected a redirect after the login, got {_said(reply)}"
    )


def _import(
    opener: urllib.request.OpenerDirector,
    base: str,
    course_session: int,
    units: tuple[str, ...],
    body: Mapping[str, Any],
) -> ExerPlazaImport:
    query = urllib.parse.urlencode(
        [("course_session", str(course_session)), *(("content_unit", unit) for unit in units)]
    )
    what = f"POST {base}{IMPORT_PATH}"
    request = urllib.request.Request(  # noqa: S310 - _base_url allows http(s) only
        f"{base}{IMPORT_PATH}?{query}",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    reply = _send(opener, request, what)
    if reply.status == 200:
        try:
            return _parse_import(json.loads(reply.body.decode("utf-8")))
        except (UnicodeError, json.JSONDecodeError):
            raise ExerPlazaExportError(f"{what}: expected a JSON answer from ExerPlaza") from None
        except ValueError as exc:
            raise ExerPlazaExportError(f"{what}: unexpected answer from ExerPlaza: {exc}") from None
    if 300 <= reply.status < 400:
        raise ExerPlazaExportError(
            f"{what}: ExerPlaza redirected to {_printable(reply.headers.get('Location') or '?')}: "
            "the session was not accepted, or this ExerPlaza has no import endpoint"
        )
    raise ExerPlazaExportError(f"{what}: ExerPlaza refused the import: {_said(reply)}")


def _parse_import(data: object) -> ExerPlazaImport:
    """Read ExerPlaza's answer, raising ``ValueError`` naming the first field out of shape."""
    if not isinstance(data, Mapping):
        raise ValueError("expected a JSON object")
    quiz = data.get("quiz")
    if not isinstance(quiz, Mapping):
        raise ValueError("quiz: expected an object")
    quiz_uid = _uid(quiz.get("uid"), "quiz.uid")
    quiz_outcome = quiz.get("outcome")
    if quiz_outcome not in _QUIZ_OUTCOMES:
        raise ValueError(f"quiz.outcome: expected 'created' or 'updated', got {quiz_outcome!r}")
    questions = data.get("questions")
    if not isinstance(questions, list):
        raise ValueError("questions: expected a list")
    items: list[ExerPlazaItem] = []
    for index, entry in enumerate(questions):
        path = f"questions[{index}]"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{path}: expected an object")
        ref, outcome, reason = entry.get("ref"), entry.get("outcome"), entry.get("reason")
        if not isinstance(ref, str) or not ref:
            raise ValueError(f"{path}.ref: expected a non-empty string")
        if outcome not in _ITEM_OUTCOMES:
            raise ValueError(f"{path}.outcome: expected one of {sorted(_ITEM_OUTCOMES)}")
        if reason is not None and not isinstance(reason, str):
            raise ValueError(f"{path}.reason: expected a string or null")
        uid = _uid(entry.get("uid"), f"{path}.uid")
        items.append(ExerPlazaItem(ref=ref, uid=uid, outcome=outcome, reason=reason))
    return ExerPlazaImport(quiz_uid=quiz_uid, quiz_outcome=quiz_outcome, items=tuple(items))


def _uid(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path}: expected a positive id, got {value!r}")
    return value


def _send(
    opener: urllib.request.OpenerDirector, request: urllib.request.Request, what: str
) -> _Reply:
    """Send ``request`` and return the answer whatever its status; fail only without one."""
    try:
        with opener.open(request, timeout=_TIMEOUT_SECONDS) as response:
            return _Reply(status=response.status, headers=response.headers, body=response.read())
    except urllib.error.HTTPError as error:  # a status >= 300 is an answer here, not a failure
        with error:
            return _Reply(status=error.code, headers=error.headers, body=error.read())
    except TimeoutError:
        raise ExerPlazaExportError(f"{what}: no answer within {_TIMEOUT_SECONDS} s") from None
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, TimeoutError):
            raise ExerPlazaExportError(f"{what}: no answer within {_TIMEOUT_SECONDS} s") from None
        raise ExerPlazaExportError(f"{what}: cannot reach ExerPlaza ({reason})") from None
    except (OSError, http.client.HTTPException) as exc:
        raise ExerPlazaExportError(f"{what}: the connection failed ({exc!r})") from None


def _said(reply: _Reply) -> str:
    """Return the status and ExerPlaza's explanation as sent: an RFC 9457 problem or its own."""
    status = f"HTTP {reply.status}"
    try:
        data = json.loads(reply.body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        data = None
    if not isinstance(data, Mapping):
        if reply.status == 404:
            return f"{status}, not a JSON answer: is this ExerPlaza, with the import endpoint?"
        return status
    head = [data[key] for key in ("title", "detail") if isinstance(data.get(key), str)]
    if not head:  # ExerPlaza's rate limiter and older routes answer {"error": ...}
        head = [data[key] for key in ("error", "message") if isinstance(data.get(key), str)][:1]
    errors = data.get("errors")
    lines = [
        f"\n  {entry['path']}: {entry['detail']}"
        for entry in (errors if isinstance(errors, list) else [])
        if isinstance(entry, Mapping)
        and isinstance(entry.get("path"), str)
        and isinstance(entry.get("detail"), str)
    ]
    said = " ".join([status, ": ".join(head)]).strip() + "".join(lines)
    return _printable(said)


def _printable(text: str) -> str:
    """Keep the server's text from steering the terminal: control characters become ``?``."""
    return "".join(char if char.isprintable() or char == "\n" else "?" for char in text)
