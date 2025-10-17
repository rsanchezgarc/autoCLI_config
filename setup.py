"""Setup script for autoCLI_config package."""

from setuptools import setup, find_packages
from pathlib import Path
import re

# Read the version from __init__.py
def get_version():
    """Extract version from __init__.py."""
    init_file = Path(__file__).parent / "autoCLI_config" / "__init__.py"
    with open(init_file, 'r') as f:
        content = f.read()
    match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content, re.MULTILINE)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string in __init__.py")

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
def read_requirements(filename):
    """Read requirements from a file."""
    req_file = Path(__file__).parent / filename
    if req_file.exists():
        with open(req_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#') and not line.startswith('-r')]
    return []

setup(
    name="autoCLI_config",
    version=get_version(),
    author="Ruben Sanchez-Garcia",
    author_email="rsanchezgarcia@faculty.ie.edu",
    description="Automatic CLI and configuration management system with documentation generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rsanchezgarc/autoCLI_config",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements_dev.txt"),
    },
    keywords="configuration cli documentation argparse dataclass autocli",
    project_urls={
        "Bug Reports": "https://github.com/rsanchezgarc/autoCLI_config/issues",
        "Source": "https://github.com/rsanchezgarc/autoCLI_config",
    },
)
