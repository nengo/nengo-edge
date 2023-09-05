# pylint: disable=missing-docstring

from pathlib import Path

import pytest
from click.testing import CliRunner

from nengo_edge import cli
from nengo_edge.tests.test_validate import mock_asr_dataset, mock_kws_dataset
from nengo_edge.version import version


@pytest.mark.parametrize("dataset_type", ("kws", "asr"))
def test_package_dataset(
    dataset_type: str,
    tmp_path: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_directory = tmp_path / dataset_type
    data_directory.mkdir()

    if dataset_type == "kws":
        mock_kws_dataset(data_directory)

    elif dataset_type == "asr":
        mock_asr_dataset(data_directory)

    monkeypatch.chdir(tmp_path)
    result = cli_runner.invoke(
        cli.cli,
        f"package-dataset {dataset_type} {data_directory}".split(),
    )
    assert result.exit_code == 0, result.exception

    output_tar = Path.cwd() / f"{dataset_type}-v{version}-{data_directory.name}.tar.gz"
    assert output_tar.exists()

    with pytest.raises(FileExistsError, match="already exists"):
        result = cli_runner.invoke(
            cli.cli,
            f"package-dataset {dataset_type} {data_directory}".split(),
            catch_exceptions=False,
        )
