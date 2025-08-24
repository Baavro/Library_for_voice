# setup.py
from pathlib import Path
import re
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
INIT = ROOT  / "__init__.py"
README = ROOT / "README.md"

def read_version():
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', INIT.read_text(encoding="utf-8"), re.M)
    if not m:
        raise RuntimeError("Cannot find __version__ in stt_sdk/__init__.py")
    return m.group(1)

long_description = README.read_text(encoding="utf-8") if README.exists() else "Async SDK for the STT service."

setup(
    name="stt-sdk",
    version=read_version(),
    description="Async Python SDK for the STT service (auth, retries, pooling, telemetry).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Team",
    license="MIT",
    url="https://example.com/stt-sdk",
    packages=find_packages(include=["stt_sdk", "stt_sdk.*"]),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "httpx>=0.27,<0.29",
    ],
    extras_require={
        "dev": ["pytest>=7", "ruff>=0.4", "mypy>=1.6", "types-requests"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries",
        "Topic :: Multimedia :: Sound/Audio",
    ],
)
