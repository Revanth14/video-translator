from html.parser import HTMLParser
from pathlib import Path


SITE = Path(__file__).parents[1] / "site"


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
    assert "Retry" in text
    assert "Subtitles" in text
    assert "Run next stage" not in text


def test_customer_site_references_assets() -> None:
    html = (SITE / "index.html").read_text(encoding="utf-8")

    assert "source-panel" in html
    assert 'href="./styles.css"' in html
    assert 'src="./app.js"' in html
    assert (SITE / "styles.css").is_file()
    assert (SITE / "app.js").is_file()


def test_customer_site_interaction_uses_video_upload_and_progress() -> None:
    script = (SITE / "app.js").read_text(encoding="utf-8")

    assert "URL.createObjectURL(file)" in script
    assert 'postForm("/api/jobs", form)' in script
    assert "autoAdvance" not in script
    assert "/stages/${stage}" not in script
    assert "localStorage" in script
    assert "let polling = false" in script
    assert 'searchParams.set("job"' in script
    assert 'form.append("glossary"' in script
    assert 'form.append("voice_reference"' in script
    assert "customerStatus" in script
    assert "setDownload" in script
    assert "startRun" in script
    assert "finishRun" in script
    assert "progressBar" in script


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
