# setup.py
# Minimal setup for editable install: pip install -e .
# No build/distribution needed — this is a local dev install only.

from setuptools import setup, find_packages

setup(
    name="tinylm",
    version="0.1.0",
    description="GPT-style transformer trained from scratch on H.P. Lovecraft",
    author="Your Name",
    python_requires=">=3.10",
    packages=find_packages(),  # auto-discovers tinylm/ and sub-packages
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.64.0",
    ],
)