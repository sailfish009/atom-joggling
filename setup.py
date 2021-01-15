from setuptools import find_namespace_packages, setup

setup(
    name="supercon",
    version="0.0.1",
    author="Janosh Riebesell",
    author_email="janosh.riebesell@gmail.com",
    long_description=open("readme.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/janosh/supercon",
    packages=find_namespace_packages(include=["supercon*"]),
)
