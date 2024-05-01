# Automatically generated by nengo-bones, do not edit this file directly

import io
import pathlib
import runpy

try:
    from setuptools import find_packages, setup
except ImportError:
    raise ImportError(
        "'setuptools' is required but not installed. To install it, "
        "follow the instructions at "
        "https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py"
    )


def read(*filenames, **kwargs):
    encoding = kwargs.get("encoding", "utf-8")
    sep = kwargs.get("sep", "\n")
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


root = pathlib.Path(__file__).parent
version = runpy.run_path(str(root / "nengo_edge" / "version.py"))["version"]

install_req = [
    "click>=8.1.3",
    "packaging>=20.9",
    "pyserial>=3.5",
    "rich>=13.3.1",
    "soundfile>=0.12.1",
    "numpy>=1.23.0",
    "gql>=3.5.0",
]
docs_req = [
    "jupyter>=1.0.0",
    "nbsphinx>=0.8.11",
    "nengo-sphinx-theme>=20.9",
    "numpydoc>=1.4.0",
    "sounddevice>=0.4.5",
    "sphinx-click>=5.0.0",
    "sphinx-tabs>=3.2.0",
]
optional_req = [
    "tensorflow>=2.10.0",
    "tensorflow-text>=2.13.0",
]
tests_req = [
    "mypy>=0.901",
    "pytest>=7.1.1",
    "pytest-rng>=1.0.0",
    "types-click>=7.1.0",
]

setup(
    name="nengo-edge",
    version=version,
    author="Applied Brain Research",
    author_email="edge-info@appliedbrainresearch.com",
    packages=find_packages(),
    url="https://github.com/nengo/nengo-edge",
    include_package_data=True,
    license="MIT license",
    description="Tools for working with NengoEdge",
    long_description=read("README.rst", "CHANGES.rst"),
    zip_safe=False,
    install_requires=install_req,
    extras_require={
        "all": docs_req + optional_req + tests_req,
        "docs": docs_req,
        "optional": optional_req,
        "tests": tests_req,
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Framework :: Nengo",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "nengo-edge=nengo_edge.cli:cli",
        ],
    },
)
