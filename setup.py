from setuptools import setup, find_packages


setup(
    name="qwergpt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "aiohttp>=3.10.2",
        "loguru>=0.7.2",
        "python-dotenv>=1.0.1",
        "pydantic>=2.8.2",
        "requests>=2.32.3",
    ],
    author="Leo Peng",
    author_email="leo@promptcn.com",
    description="QwerGPT: A Lightweight LLM Framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/promptcn/qwergpt",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
