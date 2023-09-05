# pylint: disable=missing-docstring

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import soundfile

from nengo_edge import validate


def save_audio_file(
    filepath: Path,
    audio_type: str,
    sample_rate: int,
    duration: float,
) -> None:
    # note: creating 2 channel data, second channel will be ignored
    sample_data = np.random.uniform(
        -32000, 32000, size=(int(sample_rate * duration), 2)
    ).astype("int16")
    soundfile.write(filepath, sample_data, sample_rate, format=audio_type)


def save_transcript_file(
    filepath: Path, fileIDs: List[str], sentences: List[str]
) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        for fileID, sentence in zip(fileIDs, sentences):
            f.write(f"{fileID} {sentence}\n")


def mock_asr_dataset(
    tmp_path: Path,
    audio_type: str = "wav",
    sample_rate: int = 16000,
    duration: float = 0.5,  # in seconds
    how_many: int = 10,
) -> None:
    rng = random.Random(0)
    for split in ["train", "validation", "test"]:
        split_dir = tmp_path / split
        split_dir.mkdir()
        fileIDs = []
        sentences = []
        for i in range(how_many):
            fileIDs.append(f"some_audio_{i}.{audio_type}")
            sentences.append(f"{''.join(rng.choices('abc ' , k=100))}")
            save_audio_file(
                split_dir / f"some_audio_{i}.{audio_type}",
                audio_type=audio_type,
                sample_rate=sample_rate,
                duration=duration,
            )
        save_transcript_file(split_dir / "transcription.txt", fileIDs, sentences)


def mock_kws_dataset(
    tmp_path: Path,
    audio_type: str = "wav",
    sample_rate: int = 16000,
    duration: float = 0.5,  # in seconds
    keywords: Tuple[str, ...] = ("on", "off"),
    how_many: int = 10,
) -> None:
    for split in ["train", "validation", "test"]:
        split_dir = tmp_path / split
        for word in keywords:
            keyword_dir = split_dir / word
            keyword_dir.mkdir(parents=True)
            for i in range(how_many):
                save_audio_file(
                    keyword_dir / f"some_audio_{i}.{audio_type}",
                    audio_type=audio_type,
                    sample_rate=sample_rate,
                    duration=duration,
                )

    background_dir = tmp_path / "background_noise"
    background_dir.mkdir()
    for i in range(2):
        save_audio_file(
            background_dir / f"some_audio_{i}.{audio_type}",
            audio_type=audio_type,
            sample_rate=sample_rate,
            duration=duration,
        )


@pytest.mark.parametrize(
    "split",
    ("train", "validation", "test", "unknown"),
)
def test_infer_split(split: str) -> None:
    path = Path(f"{split}/path/to/file.txt")
    if split == "unknown":
        with pytest.raises(ValueError, match=f"{path} should be"):
            validate.infer_split(path)
    else:
        assert split == validate.infer_split(path)


@pytest.mark.parametrize("audio_format", ("wav", "flac", "mp3"))
def test_infer_audio_properties(audio_format: str, tmp_path: Path) -> None:
    sample_rate = 16000
    duration = 0.5
    save_audio_file(
        tmp_path / f"audio.{audio_format}",
        audio_type=audio_format,
        sample_rate=sample_rate,
        duration=duration,
    )

    with open(tmp_path / f"audio.{audio_format}", "rb") as f:
        audio_bytes = f.read()
        audio_properties = validate.infer_audio_properties(audio_bytes)

    assert audio_properties["sample_rate"] == sample_rate

    # validate.infer_audio_properties returns clip duration in ms
    assert audio_properties["clip_duration"] == duration * 1e3


def test_validate_audio(tmp_path: Path) -> None:
    sample_rate = 16000
    duration = 0.5

    # test clip duration warning
    data_dir = tmp_path / "kws0"
    data_dir.mkdir()
    mock_kws_dataset(data_dir, sample_rate=sample_rate, duration=duration)
    save_audio_file(
        data_dir / "audio.wav",
        audio_type="wav",
        sample_rate=sample_rate,
        duration=duration + 10,
    )

    audio_files = sorted(list(data_dir.rglob("*.wav")))
    with pytest.warns(match="Clip durations"):
        validate.validate_audio(audio_files)

    # test sample rate error
    data_dir = tmp_path / "kws1"
    data_dir.mkdir()
    mock_kws_dataset(data_dir, sample_rate=sample_rate, duration=duration)
    save_audio_file(
        data_dir / "audio.wav",
        audio_type="wav",
        sample_rate=sample_rate + 1,
        duration=duration,
    )

    audio_files = list(data_dir.rglob("*.wav"))
    with pytest.raises(ValueError, match="Found sample rates"):
        validate.validate_audio(audio_files)


def test_audio_map_fn(tmp_path: Path) -> None:
    save_audio_file(
        tmp_path / "audio.wav", audio_type="wav", sample_rate=16000, duration=1
    )

    props = validate.audio_map_fn(tmp_path / "audio.wav")
    assert props["sample_rate"] == 16000
    assert props["clip_duration"] == 1000


def test_validate_transcriptions(tmp_path: Path) -> None:
    data_dir = tmp_path / "asr0"
    data_dir.mkdir()
    mock_asr_dataset(data_dir)

    with open(data_dir / "train" / "transcription.txt", "a+", encoding="utf-8") as f:
        f.write("dummy_id.audio dummy sentence\n")

    audio_files = list(data_dir.rglob("*.wav"))

    with open(data_dir / "train" / "transcription.txt", "r", encoding="utf-8") as f:
        transcription_lines = [
            f"{data_dir / 'train'}/{line}" for line in f.read().splitlines()
        ]

    with pytest.raises(
        FileNotFoundError,
        match=f"reference to {data_dir / 'train' / 'dummy_id.audio'}",
    ):
        validate.validate_transcriptions(transcription_lines, audio_files)


def test_validate_kws_formatting(tmp_path: Path) -> None:
    # test incompatible audio format
    with pytest.raises(FileNotFoundError, match="Found no compatible audio"):
        validate.validate_kws_formatting([])

    # test missing split
    fpaths = [
        tmp_path / f"{split}" / "keyword" / "some_audio.wav"
        for split in ("train", "validation")
    ]
    with pytest.raises(FileNotFoundError, match="Did not find train/"):
        validate.validate_kws_formatting(fpaths)

    # test warning when no background_noise dir present
    fpaths = [
        tmp_path / f"{split}" / "keyword" / "some_audio.wav"
        for split in ("train", "validation", "test")
    ]
    with pytest.warns(match="No background_noise"):
        validate.validate_kws_formatting(fpaths)

    # test warning when extra top level directory is found with
    # _dirname_ format
    fpaths = [
        tmp_path / f"{split}" / "keyword" / "some_audio.wav"
        for split in ("train", "validation", "test", "_invalid_")
    ]
    with pytest.warns(match="Found directory _invalid_"):
        validate.validate_kws_formatting(fpaths)

    # test invalid/no keywords
    fpaths = [
        tmp_path / f"{split}" / "_invalid_" / "some_audio.wav"
        for split in ("train", "validation", "test")
    ]
    with pytest.raises(FileNotFoundError, match="keyword directories"):
        validate.validate_kws_formatting(fpaths)

    # test keyword returns
    fpaths = [
        tmp_path / f"{split}" / f"keyword_{i}" / "some_audio.wav"
        for i, split in enumerate(("train", "validation", "test"))
    ]
    props = validate.validate_kws_formatting(fpaths)
    assert props["allKeywords"] == [f"keyword_{i}" for i in range(3)]


def test_validate_asr_formatting(tmp_path: Path) -> None:
    # test incompatible audio format
    with pytest.raises(FileNotFoundError, match="Found no compatible audio"):
        validate.validate_asr_formatting([], [tmp_path / "some.txt"])

    with pytest.raises(FileNotFoundError, match="Found no compatible transcription"):
        validate.validate_asr_formatting([tmp_path / "some_audio.wav"], [])

    audio = [
        tmp_path / f"{split}" / "some_audio.wav" for split in ("train", "validation")
    ]
    txt = [f.with_suffix(".txt") for f in audio]
    # test missing split
    with pytest.raises(FileNotFoundError, match="Did not find train/"):
        validate.validate_asr_formatting(audio, txt)


def test_validate_dataset_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # test empty directory error
    with pytest.raises(FileNotFoundError, match="Found empty"):
        validate.validate_dataset("kws", tmp_path, validation_samples=None)

    # test really large dataset warning
    def mock_kws_dataset_validation(audio_paths: List[Path]) -> Dict[str, List[str]]:
        """fast kws validation."""
        return {"allKeywords": []}

    def mock_audio_validation(audio_paths: List[Path], dataset_type: str) -> None:
        """fast audio validation."""

    def rglob(*_: Any) -> List[Path]:
        return [Path(f"audio_{i}.wav") for i in range(int(1e5 + 1))]

    def is_file(*_: Any) -> bool:
        return True

    monkeypatch.setattr(
        validate, "validate_kws_formatting", mock_kws_dataset_validation
    )
    monkeypatch.setattr(validate, "validate_audio", mock_audio_validation)
    monkeypatch.setattr(Path, "rglob", rglob)
    monkeypatch.setattr(Path, "is_file", is_file)

    with pytest.warns(match=f"Validating {int(1e5) + 1} audio files"):
        validate.validate_dataset("kws", tmp_path, validation_samples=None)


@pytest.mark.parametrize("dataset_type", ("kws", "asr"))
def test_validate_dataset(
    dataset_type: str, tmp_path: Path, capfd: pytest.CaptureFixture
) -> None:
    data_dir = tmp_path / dataset_type
    data_dir.mkdir()

    if dataset_type == "kws":
        mock_kws_dataset(data_dir)
    else:
        mock_asr_dataset(data_dir)

    validated_fpaths = validate.validate_dataset(
        dataset_type, data_dir, validation_samples=10
    )

    out = capfd.readouterr().out
    assert "Successfully validated 10 audio files" in out
    if dataset_type == "asr":
        assert "Successfully validated 30 transcription entries" in out
    assert validated_fpaths == [f for f in data_dir.rglob("*") if f.is_file()]
