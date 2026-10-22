from pathlib import Path

import pytest

from questxtract.config import DEEPSEEK, Config, load_config


def test_missing_implicit_file_keeps_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_config() == Config()


@pytest.mark.parametrize("bom", ["", "\ufeff"])
def test_reads_cwd_file_over_defaults_with_utf8_and_optional_bom(tmp_path, monkeypatch, bom):
    monkeypatch.chdir(tmp_path)
    Path("questxtract.toml").write_text(
        bom + 'model_size = "small"\nmin_confidence = 0.6\ninitial_prompt = "Spørsmål"\n',
        encoding="utf-8",
    )
    assert load_config() == Config().replace(
        model_size="small", min_confidence=0.6, initial_prompt="Spørsmål"
    )


def test_explicit_file_replaces_default_file_selection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("questxtract.toml").write_text('model_size = "small"', encoding="utf-8")
    other = tmp_path / "other.toml"
    other.write_text('detector = "llm"\ncache_dir = "cache"', encoding="utf-8")
    assert load_config(other) == Config().replace(detector="llm", cache_dir="cache")


def test_missing_explicit_file_does_not_fall_back(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("questxtract.toml").write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing.toml"):
        load_config(tmp_path / "missing.toml")


def test_accepts_field_types_including_optional_string_and_int_for_float(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text(
        'base_url = "http://localhost:11434/v1"\n'
        "beam_size = 3\nvision_fps = 2\nfrom_scratch = true\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config == Config().replace(
        base_url="http://localhost:11434/v1", beam_size=3, vision_fps=2.0, from_scratch=True
    )
    assert type(config.vision_fps) is float


@pytest.mark.parametrize(
    ("key", "value", "expected", "actual"),
    [
        ("min_confidence", '"high"', "float", "string"),
        ("beam_size", "5.5", "int", "float"),
        ("beam_size", "true", "int", "bool"),
        ("vision_fps", "true", "float", "bool"),
        ("from_scratch", "1", "bool", "int"),
        ("base_url", "5", "str", "int"),
        ("model_size", '["small"]', "str", "array"),
        ("model_size", '{name = "small"}', "str", "table"),
    ],
)
def test_wrong_types_name_file_key_and_types(tmp_path, key, value, expected, actual):
    path = tmp_path / "settings.toml"
    path.write_text(f"{key} = {value}", encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_config(path)
    assert str(error.value) == f"{path}: {key}: expected {expected}, got {actual}"


def test_unknown_key_reports_typo_without_echoing_value(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text('min_confidance = "do-not-echo"', encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_config(path)
    assert str(path) in str(error.value)
    assert "unknown config field(s): min_confidance" in str(error.value)
    assert "do-not-echo" not in str(error.value)


@pytest.mark.parametrize("key", ["api_key", "openai_token", "my_secret", "PASSWORD", "Key"])
def test_secrets_are_rejected_without_echoing_values(tmp_path, key):
    path = tmp_path / "settings.toml"
    path.write_text(f'{key} = "do-not-echo"', encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_config(path)
    message = str(error.value)
    assert str(path) in message and key in message and "OPENAI_API_KEY" in message
    assert "do-not-echo" not in message


@pytest.mark.parametrize("content", ['model_size = "small"\nbeam_size = @', "beam_size = "])
def test_malformed_toml_reports_file_and_line(tmp_path, content):
    path = tmp_path / "settings.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_config(path)
    assert str(error.value) == f"{path}: invalid TOML at line {content.count(chr(10)) + 1}"


def test_parser_errors_do_not_echo_input(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text('"do-not-echo" = 1\n"do-not-echo" = 2', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid TOML") as error:
        load_config(path)
    assert "do-not-echo" not in str(error.value)


def test_invalid_encoding_reports_path_without_echoing_input(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_bytes(b'model_size = "\xff"')
    with pytest.raises(ValueError, match="expected UTF-8") as error:
        load_config(path)
    assert str(path) in str(error.value)


@pytest.mark.parametrize("value", ["sk-test", "'sk-test'", '"sk-test"'])
def test_dotenv_reads_only_api_key_with_comments_and_quotes(tmp_path, monkeypatch, value):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("UNRELATED_SETTING", raising=False)
    Path(".env").write_text(
        f"\ufeff# local settings\n\nUNRELATED_SETTING=ignored\nOPENAI_API_KEY={value} # note\n",
        encoding="utf-8",
    )
    config = Config(base_url=None)  # OpenAI: OPENAI_API_KEY
    assert config.api_key == "sk-test"
    assert "sk-test" not in repr(config)
    import os

    assert "UNRELATED_SETTING" not in os.environ
    assert "OPENAI_API_KEY" not in os.environ


@pytest.mark.parametrize("value", ["sk-environment", ""])
def test_environment_key_wins_even_over_invalid_dotenv(tmp_path, monkeypatch, value):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", value)
    Path(".env").write_text('OPENAI_API_KEY="unterminated', encoding="utf-8")
    assert Config(base_url=None).api_key == value


def test_dotenv_never_searches_parent_directories(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-parent", encoding="utf-8")
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert Config().api_key is None


@pytest.mark.parametrize(
    "value", ['"sk-do-not-echo', "sk-do-not-echo extra", '"$(do-not-echo)"', '"${do-not-echo}"']
)
def test_bad_dotenv_values_are_reported_without_exposure(tmp_path, monkeypatch, value):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    Path(".env").write_text(f"OPENAI_API_KEY={value}", encoding="utf-8")
    with pytest.raises(ValueError) as error:
        _ = Config(base_url=None).api_key
    assert ".env: line 1:" in str(error.value)
    assert "do-not-echo" not in str(error.value)


def test_tests_see_neither_the_developers_key_nor_their_checkout():
    # conftest's autouse fixture: a README-style .env or a questxtract.toml in the checkout,
    # or OPENAI_API_KEY in the shell, must not change any test's result (issue #112).
    import os

    assert "OPENAI_API_KEY" not in os.environ
    assert Path.cwd().resolve() != Path(__file__).resolve().parents[1]
    assert list(Path.cwd().iterdir()) == []
    assert Config().api_key is None
    assert load_config() == Config()


def test_everything_defaults_to_deepseek_on_ntnu_with_one_key(tmp_path, monkeypatch):
    """#173: vision, the judge and the quiz on one model at NTNU's service, with its key."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    Path(".env").write_text("IDUN_API_KEY=ntnu-dotenv\n", encoding="utf-8")

    config = Config()

    assert config.vision_model == config.judge_model == config.quiz_model == DEEPSEEK
    assert config.base_url == config.vision_url == "https://llm.hpc.ntnu.no/v1"
    assert config.api_key == config.vision_api_key == "ntnu-dotenv"  # never the OpenAI key
    assert config.reasoning_effort == "none"
    assert "ntnu-dotenv" not in repr(config)


def test_each_key_goes_to_one_service_only(monkeypatch):
    """#173 review: the vision key is never NTNU's, so an OpenAI key stored there for #162's
    setup never reaches NTNU, nor an NTNU key a third-party vision host."""
    monkeypatch.setenv("QUESTXTRACT_VISION_API_KEY", "vision-key")
    assert Config().api_key is None and Config().vision_api_key is None

    third_party = Config(vision_base_url="https://openrouter.ai/api/v1")
    monkeypatch.setenv("IDUN_API_KEY", "ntnu-key")
    assert (third_party.api_key, third_party.vision_api_key) == ("ntnu-key", "vision-key")


def test_openai_takes_the_openai_key_and_never_the_ntnu_one(monkeypatch):
    """#173 AC2: the key goes with the endpoint."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("IDUN_API_KEY", "ntnu-key")

    openai = Config(base_url="https://api.openai.com/v1")

    assert (openai.api_key, openai.vision_url, openai.vision_api_key) == (
        "sk-openai",
        "https://api.openai.com/v1",
        "sk-openai",
    )
    split = openai.replace(vision_base_url="https://llm.hpc.ntnu.no/v1/")
    assert (split.api_key, split.vision_api_key) == ("sk-openai", "ntnu-key")


def test_the_ntnu_key_in_the_environment_wins_over_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IDUN_API_KEY", "ntnu-environment")
    Path(".env").write_text("IDUN_API_KEY=ntnu-dotenv\n", encoding="utf-8")

    assert Config().vision_api_key == "ntnu-environment"


def test_the_former_ntnu_key_name_is_still_read_for_ntnu_only(tmp_path, monkeypatch):
    """The key's old name, QUESTXTRACT_NTNU_API_KEY, still works, in the environment and .env."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("IDUN_API_KEY", raising=False)
    monkeypatch.setenv("QUESTXTRACT_NTNU_API_KEY", "old-environment")
    assert Config().api_key == Config().vision_api_key == "old-environment"

    monkeypatch.delenv("QUESTXTRACT_NTNU_API_KEY")
    Path(".env").write_text("QUESTXTRACT_NTNU_API_KEY=old-dotenv\n", encoding="utf-8")
    assert Config().api_key == "old-dotenv"


def test_the_new_ntnu_key_name_wins_over_the_old_one(tmp_path, monkeypatch):
    """IDUN_API_KEY is read first, from the environment or .env, whatever the old name holds."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QUESTXTRACT_NTNU_API_KEY", "old-environment")
    Path(".env").write_text("IDUN_API_KEY=new-dotenv\n", encoding="utf-8")
    assert Config().api_key == "new-dotenv"

    monkeypatch.setenv("IDUN_API_KEY", "new-environment")
    assert Config().api_key == "new-environment"


def test_the_former_ntnu_key_never_goes_to_another_service(monkeypatch):
    """The old name is NTNU's key still: OpenAI and a third-party vision host never get it."""
    monkeypatch.delenv("IDUN_API_KEY", raising=False)
    monkeypatch.setenv("QUESTXTRACT_NTNU_API_KEY", "old-ntnu")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    openai = Config(base_url="https://api.openai.com/v1")
    third_party = Config(vision_base_url="https://openrouter.ai/api/v1")

    assert openai.api_key is None
    assert third_party.vision_api_key is None
    assert third_party.api_key == "old-ntnu"  # the judge still goes to NTNU


def test_vision_reasoning_follows_reasoning_effort_unless_set():
    """#173 review: vision may need another setting than the judge (gpt-4o judge, DeepSeek
    vision; or a local vision model and DeepSeek judging)."""
    assert Config(reasoning_effort="low").vision_effort == "low"
    split = Config(reasoning_effort="", vision_reasoning_effort="none")
    assert (split.reasoning_effort, split.vision_effort) == ("", "none")


def test_vision_follows_base_url_unless_it_has_its_own(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-local")
    monkeypatch.setenv("QUESTXTRACT_VISION_API_KEY", "vision-key")

    local = Config(base_url="http://localhost:11434/v1")
    own = Config(base_url="http://localhost:11434/v1", vision_base_url="http://vlm/v1")

    assert (local.vision_url, local.vision_api_key) == ("http://localhost:11434/v1", "sk-local")
    # a host of its own for vision keeps its own key: the OpenAI key never goes there (#173)
    assert (own.vision_url, own.vision_api_key) == ("http://vlm/v1", "vision-key")


def test_the_old_reasoning_name_in_the_settings_file_still_applies(tmp_path, monkeypatch):
    """#173 AC5: judge_reasoning_effort is reasoning_effort's old name."""
    monkeypatch.chdir(tmp_path)
    Path("questxtract.toml").write_text('judge_reasoning_effort = "low"\n', encoding="utf-8")
    assert load_config().reasoning_effort == "low"

    Path("questxtract.toml").write_text(
        'judge_reasoning_effort = "low"\nreasoning_effort = "none"\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="set reasoning_effort only"):
        load_config()


@pytest.mark.parametrize(
    "url",
    [
        "https://LLM.hpc.ntnu.no/v1",
        "https://llm.hpc.ntnu.no:443/v1/",
        "http://llm.hpc.ntnu.no/v1",
        "https://llm.hpc.ntnu.no",
        "https://llm.hpc.ntnu.no/api/v1",
    ],
)
def test_any_spelling_of_ntnus_url_gets_the_ntnu_key(monkeypatch, url):
    """#173 review: the key goes with the service, however its URL is written."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("IDUN_API_KEY", "ntnu-key")
    assert Config(base_url=url).api_key == "ntnu-key"
    assert Config(base_url="https://api.openai.com/v1").api_key == "sk-openai"
