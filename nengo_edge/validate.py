"""Various helper functions and tools for validating NengoEdge datasets."""

import io
import multiprocessing
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import soundfile
from rich import progress


def infer_split(relative_path: Path) -> str:
    """
    Infer the training split from a Path object.

    The training split is inferred from the base part
    of a file path, where the filepath should be a relative path,
    with respect to its parent data directory. If the inferred split
    is not one of train, validation or test, an error is thrown.

    Parameters
    ----------
    relative_path : Path
        File path, relative to its data directory.

    Returns
    -------
    split : str
        Returns one of train, validation or test.
    """
    for split in ["train", "validation", "test"]:
        if relative_path.parts[0].startswith(split):
            return split

    # raise error if function does not return
    raise ValueError(f"{relative_path} should be in one of train, validation or test")


def infer_audio_properties(audio_bytes: bytes) -> Dict[str, Union[int, float]]:
    """
    Infer clip duration and sample rate from audio bytes.

    Parameters
    ----------
    audio_bytes : bytes
        Bytes like object containing audio to be read

    Returns:
    --------
    props: Dict[str, Union[int, float]]
        Dictionary containing the audio `sample_rate` as an int and
        audio `clip_duration` as a float.
    """
    data, rate = soundfile.read(io.BytesIO(audio_bytes))
    duration = len(data) / rate * 1e3  # in ms
    props = {"clip_duration": duration, "sample_rate": int(rate)}
    return props


def audio_map_fn(file_path: Path) -> Dict[str, Union[int, float]]:
    """Map function for multiprocessing audio files."""
    return infer_audio_properties(file_path.read_bytes())


def validate_audio(
    audio_paths: List[Path], dataset_type: Optional[str] = "kws"
) -> None:
    """
    Validate a portion of the audio files in the full dataset.

    Validation will warn users if clip durations are inconsistent, as well as raise
    an error if audio sample rates differ between clips in the dataset.

    Parameters
    ----------
    audio_paths : List[Path]
        A list of path objects pointing to audio files in the dataset.
    dataset_type : str
        One of `kws` or `asr`, to toggle relevant warnings/errors.
    """
    clip_durations = []
    sample_rates = []

    with progress.Progress() as p:
        task_id = p.add_task(
            f"Reading {len(audio_paths)} audio samples", total=len(audio_paths)
        )
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            for result in pool.imap_unordered(audio_map_fn, audio_paths):
                clip_durations.append(result["clip_duration"])
                sample_rates.append(result["sample_rate"])
                p.advance(task_id)

    if dataset_type == "kws" and clip_durations.count(clip_durations[0]) != len(
        clip_durations
    ):
        # this isn't necessarily an error, clips will just be truncated/padded
        # to the desired length
        warnings.warn(
            "Clip durations are inconsistent. They will be cropped "
            "or padded at runtime."
        )
    if sample_rates.count(sample_rates[0]) != len(sample_rates):
        raise ValueError(
            f"Found sample rates in the range ({min(sample_rates)}, "
            f"{max(sample_rates)}) Hz. Audio files require consistent sample "
            "rates before uploading."
        )

    print(f"Successfully validated {len(audio_paths)} audio files.")


def validate_transcriptions(
    transcription_lines: List[str],
    audio_paths: List[Path],
) -> None:
    """
    Validate transcriptions for ASR datasets.

    Each line in a transcription file should have the format:
        `<filename>.<audio_format> <sentence>`
    where the first column is the filename in the same directory
    as the transcription file and the sentence portion is a space delimited
    string of words corresponding to the audio file.

    Validation will throw an error if transcription entries do not contain
    filepaths that point to existing audio files in the dataset.

    Parameters
    ----------
    transcription_lines : List[str]
        A list of strings corresponding to the lines in each of the
        transcription files, with each line prepended with the absolute file path
        to the corresponding transcription file.
    audio_paths : List[Path]
        A list of path objects pointing to audio files in the dataset.
    """
    expected_set = set()
    audio_set = set(audio_paths)
    for line in progress.track(
        transcription_lines,
        description=f"Reading {len(transcription_lines)} transcription entries",
    ):
        expected_set.add(Path(line.split()[0]))

    if not expected_set.issubset(audio_set):
        missing = expected_set - audio_set
        raise FileNotFoundError(
            f"Found reference to {missing.pop()} in transcription, "
            "but file does not exist."
        )

    print(f"Successfully validated {len(transcription_lines)} transcription entries.")


def validate_kws_formatting(audio_paths: List[Path]) -> Dict[str, List[str]]:
    """
    Validate KWS datasets for proper directory structure and audio consistency.

    Parameters
    ----------
    audio_paths : List[Path]
        A list of path objects pointing to all audio files inside the data directory.

    Returns
    -------
    dataset_properties : Dict[str, List[str]]
        Dictionary containing a list of keywords that are inferred from
        the dataset.
    """

    # check validation failure conditions
    if len(audio_paths) == 0:
        raise FileNotFoundError(
            "Found no compatible audio files for kws datasets. "
            "Ensure audio is one of mp3, wav or flac."
        )

    all_splits = set()
    all_keywords = set()
    common_path = os.path.commonpath(audio_paths)
    for fpath in progress.track(
        audio_paths, description="Validating dataset formatting"
    ):
        rel_path = fpath.relative_to(common_path) if len(common_path) > 1 else fpath
        if rel_path.parts[0].startswith(("train", "validation", "test")):
            all_splits.add(infer_split(rel_path))
            if not rel_path.parts[1].startswith("_"):
                all_keywords.add(rel_path.parts[1])
        else:
            all_splits.add(rel_path.parts[0])

    # check split conditions are met
    if not set(("test", "train", "validation")).issubset(all_splits):
        raise FileNotFoundError("Did not find train/validation/test folders in dataset")

    # make sure there are directories in the dataset archive with
    # names corresponding to viable keywords
    if len(all_keywords) == 0:
        raise FileNotFoundError("Did not find any keyword directories.")

    # warn if no background noise directory
    if "background_noise" not in all_splits:
        warnings.warn(
            "No background_noise directory found. "
            "Consider adding for optimal kws performance in NengoEdge."
        )

    # warn about unused directories
    for split in all_splits:
        if split not in ["train", "validation", "test", "background_noise"]:
            warnings.warn(
                f"Found directory {split}, which will be unused by NengoEdge."
            )

    dataset_properties = {"allKeywords": sorted(list(all_keywords))}
    return dataset_properties


def validate_asr_formatting(
    audio_paths: List[Path], transcription_paths: List[Path]
) -> None:
    """
    Validate ASR datasets for proper directory structure, audio consistency, and
    transcription formatting.

    Parameters
    ----------
    audio_paths : List[Path]
        A list of path objects pointing to all audio
        files inside the data directory.
    transcription_paths : List[Path]
        A list of path objects pointing to all transcription
        files inside the data directory.
    """

    if len(audio_paths) == 0:
        raise FileNotFoundError(
            "Found no compatible audio files for asr datasets. "
            "Ensure audio is one of mp3, wav or flac."
        )

    if len(transcription_paths) == 0:
        raise FileNotFoundError(
            "Found no compatible transcription files for asr datasets. "
            "Ensure transcriptions are saved in txt files."
        )

    all_splits = set()
    file_paths = audio_paths + transcription_paths
    common_path = os.path.commonpath(file_paths)
    for fpath in progress.track(
        file_paths, description="Validating dataset formatting"
    ):
        all_splits.add(
            infer_split(
                fpath.relative_to(common_path) if len(common_path) > 1 else fpath
            )
        )

    # check split conditions are met
    for split in ["train", "validation", "test"]:
        if split not in all_splits:
            raise FileNotFoundError(
                "Did not find train/validation/test folders in dataset"
            )


def validate_dataset(
    dataset_type: str, data_directory: Path, validation_samples: Optional[int]
) -> List[Path]:
    """
    Validate a dataset based on a given type and directory for that dataset.

    Parameters
    ----------
    dataset_type : str
        Type of dataset. One of kws or asr.
    data_directory : Path
        The directory in which the dataset resides.
    validation_samples : Optional[int]
        Maximum number of audio files to validate before packaging the dataset.
    """
    # ensure we are only inspecting file paths, not directories
    file_paths = [fpath for fpath in data_directory.rglob("*") if fpath.is_file()]

    if len(file_paths) == 0:
        raise FileNotFoundError(f"Found empty directory {data_directory}")

    audio_paths = [f for f in file_paths if f.suffix in [".mp3", ".wav", ".flac"]]
    if validation_samples is None:
        validation_samples = len(audio_paths)
        if len(audio_paths) > 1e5:
            warnings.warn(
                f"Validating {len(audio_paths)} audio files may take many hours. "
                f"Run with --validation-steps 100000 to choose a random subset."
            )

    assert isinstance(validation_samples, int)
    audio_subset = random.sample(audio_paths, validation_samples)

    if dataset_type == "kws":
        # validate formatting over all paths
        validate_kws_formatting(audio_paths)

        # validate audio over subset if necessary
        validate_audio(audio_subset, dataset_type="kws")

    elif dataset_type == "asr":
        txt_paths = [f for f in file_paths if f.suffix in [".txt"]]
        validate_asr_formatting(audio_paths, txt_paths)
        validate_audio(audio_subset, dataset_type="asr")

        # add absolute file path to transcription entries in order
        # to search for corresponding audio files
        transcription_lines = [
            f"{f.parent}/{line}"
            for f in txt_paths
            for line in f.read_text().splitlines()
        ]
        validate_transcriptions(transcription_lines, audio_paths)

    return file_paths
