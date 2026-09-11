import pytest

from questxtract import TranscriptSegment


@pytest.fixture
def segments() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(
            text="Today we cover sorting. What is a stable sort?", start=0.0, end=6.0
        ),
        TranscriptSegment(text="It keeps equal elements in order.", start=6.0, end=9.0),
        TranscriptSegment(text="Does anyone know why that matters", start=9.0, end=12.0),
    ]
