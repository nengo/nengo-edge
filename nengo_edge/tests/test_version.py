# pylint: disable=missing-docstring

import runpy

from nengo_edge import version


def test_version_string() -> None:
    version_string = runpy.run_path(version.__file__)["version"]

    assert ("dev" in version_string) == (version.dev is not None)
    assert version_string == version.version
