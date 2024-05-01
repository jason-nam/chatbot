"""
Setup package file
"""

from setuptools import setup

setup(
    name="chatbot",
    version="0.1",
    install_requires=[
        "numpy",
        "matplotlib",
        "pyyaml",
    ],
    extras_require={
        "tensorflow": ["tensorflow"],
        "convokit": ["convokit"],
        "tensorflow with gpu": ["tensorflow-gpu"],
    },
)