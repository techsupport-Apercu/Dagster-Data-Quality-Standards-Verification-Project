from setuptools import find_packages, setup

setup(
    name="data_quality_checker",
    packages=find_packages(exclude=["data_quality_checker_tests"]),
    install_requires=[
        "dagster",
        "dagster-webserver",
        "pandas",
    ],
    extras_require={"dev": ["pytest"]},
)
