"""Command-line interface for nengo-edge."""

import tarfile
from pathlib import Path
from typing import Optional

import click
from rich import progress

from nengo_edge import validate
from nengo_edge.version import version


@click.group()
def cli() -> None:
    """CLI for nengo-edge."""


@cli.command()
@click.argument(
    "dataset_type",
    type=click.Choice(["kws", "asr"], case_sensitive=False),
)
@click.argument(
    "data_directory",
    type=click.Path(exists=True, dir_okay=True),
)
@click.option(
    "--validation-samples",
    default=None,
    type=int,
    help=(
        "Set the number of audio files to be validated before the dataset is packaged."
    ),
)
def package_dataset(
    dataset_type: str,
    data_directory: str,
    validation_samples: Optional[int],
) -> None:
    """
    Validate and package a dataset for a given model type.

    Parameters
    ----------
    dataset_type : str
        One of kws or asr
    data_directory : str
        Directory containing a dataset split into train, validation and test
        subdirectories.
    validation_samples : Optional[int]
        Maximum number of audio files to validate before packaging the dataset.
    """
    # use resolved data_directory to avoid "../" or symlinks
    data_directory_path = Path(data_directory).resolve()
    tar_fname = (
        Path.cwd() / f"{dataset_type}-v{version}-{data_directory_path.name}.tar.gz"
    )
    # throw error if file already exists in cwd
    if tar_fname.exists():
        raise FileExistsError(f"{tar_fname} already exists.")

    # get validated filepaths
    file_paths = validate.validate_dataset(
        dataset_type.lower(), data_directory_path, validation_samples
    )

    with tarfile.open(tar_fname, mode="w:gz") as f:
        for path in progress.track(
            file_paths,
            description=f"Compressing dataset ({len(file_paths)} entries)",
        ):
            # use arcname to set filepath inside the archive to the
            # relative path with respect to data_directory_path
            f.add(str(path), arcname=str(path.relative_to(data_directory_path)))

    print(f"Saved {dataset_type} dataset archive to {tar_fname}")
