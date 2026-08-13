import json
import os
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest


SITE = Path(__file__).parents[1] / "site"
HARNESS = Path(__file__).parent / "customer_site_harness.cjs"


class TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.text: list[str] = []

    def handle_data(self, data: str) -> None:
        cleaned = data.strip()
        if cleaned:
            self.text.append(cleaned)


def page_text() -> str:
    parser = TextParser()
    parser.feed((SITE / "index.html").read_text(encoding="utf-8"))
    return "\n".join(parser.text)


def test_customer_site_has_product_flow() -> None:
    text = page_text()

    assert "Video Translator" in text
    assert "Translate video" in text
    assert "Upload video" in text
    assert "Start translation" in text
    assert "Advanced" in text
    assert "Retry" not in text
    assert "Subtitles" in text
    assert "stages complete" in text
    assert "utterances voiced" in text
    assert "attempts" in text
    assert "Run next stage" not in text


def test_customer_site_references_assets() -> None:
    html = (SITE / "index.html").read_text(encoding="utf-8")

    assert "source-panel" in html
    assert 'href="./styles.css"' in html
    assert 'src="./app.js"' in html
    assert (SITE / "styles.css").is_file()
    assert (SITE / "app.js").is_file()


def test_customer_site_uses_responsive_glass_system() -> None:
    css = (SITE / "styles.css").read_text(encoding="utf-8")

    assert "backdrop-filter: blur" in css
    assert "@supports not ((backdrop-filter" in css
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert "@media (max-width: 700px)" in css


def test_customer_site_exposes_only_release_enabled_hindi() -> None:
    html = (SITE / "index.html").read_text(encoding="utf-8")

    assert '<option value="hi">Hindi</option>' in html
    assert '<option value="es">' not in html
    assert '<option value="fr">' not in html
    assert '<option value="de">' not in html
    assert '<option value="ja">' not in html


def test_customer_site_interaction_uses_video_upload_and_progress() -> None:
    script = (SITE / "app.js").read_text(encoding="utf-8")

    assert "URL.createObjectURL(file)" in script
    assert 'postForm("/api/jobs", form)' in script
    assert "autoAdvance" not in script
    assert "/stages/${stage}" not in script
    assert "localStorage" in script
    assert "let polling = false" in script
    assert "setTimeout(async () =>" in script
    assert "setInterval(pollJob" not in script
    assert 'searchParams.set("job"' in script
    assert 'form.append("glossary"' in script
    assert 'form.append("voice_reference"' in script
    assert 'form.append("end", "90")' not in script
    assert "customerStatus" in script
    assert "setDownload" in script
    assert "startRun" in script
    assert "finishRun" in script
    assert "progressBar" in script
    assert "renderDurableProgress" in script


def test_customer_site_releases_a_job_that_no_longer_exists() -> None:
    script = (SITE / "app.js").read_text(encoding="utf-8")

    # A remembered run may belong to a cleaned runs directory or another
    # machine. The page must release it instead of polling forever.
    assert "discardMissingJob" in script
    assert "handlePollFailure" in script
    assert "error.status = response.status" in script
    assert "stopPolling();" in script
    assert "forgetJob();" in script


def test_customer_site_uses_simple_customer_statuses() -> None:
    script = (SITE / "app.js").read_text(encoding="utf-8")

    assert "Uploading" in script
    assert "Transcribing" in script
    assert "Translating" in script
    assert "Generating voice" in script
    assert "Rendering" in script
    assert "Queued" in script
    assert "Ready" in script


def node_is_missing() -> bool:
    """Decide whether the browser-behaviour harness can run.

    These scenarios are the only executable coverage of `site/app.js`, so a
    machine without Node silently drops them and the suite still reports
    success. Set VIDEO_TRANSLATOR_REQUIRE_NODE=1 in CI to fail instead of skip.
    """
    if shutil.which("node") is not None:
        return False
    if os.getenv("VIDEO_TRANSLATOR_REQUIRE_NODE") == "1":
        raise RuntimeError(
            "Node.js is required for the customer-site behaviour tests but was "
            "not found, and VIDEO_TRANSLATOR_REQUIRE_NODE=1 is set."
        )
    return True


def run_browser_state_scenario(name: str) -> dict[str, object]:
    completed = subprocess.run(
        ["node", str(HARNESS), name],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


@pytest.mark.skipif(node_is_missing(), reason="Node.js is required")
def test_restored_terminal_jobs_do_not_continue_polling() -> None:
    result = run_browser_state_scenario("rendered")

    assert result["fetches"] == ["/api/jobs/run-rendered"]
    assert result["timer_count"] == 0
    assert result["run_status"] == "Ready"
    assert result["stored_job"] == "run-rendered"


@pytest.mark.skipif(node_is_missing(), reason="Node.js is required")
def test_restored_active_job_polls_until_it_becomes_terminal() -> None:
    result = run_browser_state_scenario("active_then_rendered")

    assert result["initial_timer_count"] == 1
    assert result["fetches"] == [
        "/api/jobs/run-active",
        "/api/jobs/run-active",
    ]
    assert result["timer_count"] == 0
    assert result["run_status"] == "Ready"


@pytest.mark.skipif(node_is_missing(), reason="Node.js is required")
def test_explicit_url_job_wins_and_missing_job_clears_stale_state() -> None:
    precedence = run_browser_state_scenario("url_precedence")
    missing = run_browser_state_scenario("missing_url")

    assert precedence["fetches"] == ["/api/jobs/url-run"]
    assert precedence["stored_job"] == "url-run"
    assert "campaign=demo" in precedence["href"]
    assert missing["stored_job"] is None
    assert missing["state_job_id"] is None
    assert "job=" not in missing["href"]
    assert "campaign=demo" in missing["href"]
    assert missing["timer_count"] == 0


@pytest.mark.skipif(node_is_missing(), reason="Node.js is required")
def test_reset_preserves_unrelated_query_parameters() -> None:
    result = run_browser_state_scenario("reset")

    assert result["stored_job"] is None
    assert result["state_job_id"] is None
    assert "job=" not in result["href"]
    assert "campaign=demo" in result["href"]
    assert result["timer_count"] == 0


@pytest.mark.skipif(node_is_missing(), reason="Node.js is required")
def test_restored_failed_job_cannot_offer_browser_retry() -> None:
    result = run_browser_state_scenario("failed")

    assert result["run_status"] == "Failed"
    assert result["start_disabled"] is True
    assert result["retry_element_requested"] is False
    assert result["timer_count"] == 0


@pytest.mark.skipif(node_is_missing(), reason="Node.js is required")
def test_restored_cancelled_job_stops_with_cancelled_status() -> None:
    result = run_browser_state_scenario("cancelled")

    assert result["run_status"] == "Cancelled"
    assert result["upload_status"] == "Translation cancelled."
    assert result["timer_count"] == 0


@pytest.mark.skipif(node_is_missing(), reason="Node.js is required")
def test_reset_ignores_an_in_flight_stale_job_response() -> None:
    result = run_browser_state_scenario("reset_while_fetching")

    assert result["state_job_id"] is None
    assert result["run_status"] == "Ready"
    assert result["upload_status"] == "No file"
    assert result["timer_count"] == 0


@pytest.mark.skipif(node_is_missing(), reason="Node.js is required")
def test_non_json_submission_response_has_actionable_error() -> None:
    result = run_browser_state_scenario("non_json_submit")

    assert result["run_status"] == "Failed"
    assert result["upload_status"] == "Not submitted"
    assert result["start_disabled"] is False
    assert result["error_visible"] is True
    assert result["error_message"] == (
        "The translation API is unavailable on this server. "
        "Start it with: uv run dub-mvp web --no-open --port 8787"
    )
