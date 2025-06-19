from setuptools import find_packages, setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# TODO: add version requirements and checks


setup(
    name="uniphy",
    version="0.1.0",
    author="University of California, Los Angeles",
    author_email="contact@uniphy.org",
    description="UniPhy: A Foundation Model for Physics Simulations",
    long_description=(here / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/UniPhy",
    license="Apache-2.0",
    packages=find_packages(exclude=["configs", "scripts", "tests*"]),
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "numpy>=1.23.0",
        "PyYAML>=5.4",
        "omegaconf>=2.1.0",
        "hydra-core>=1.2.0",
        "pytorch-lightning>=1.8.0",
        "einops>=0.6.0",
        "tqdm>=4.0.0",
        "accelerate>=0.14.0",
        "nemo_toolkit[all]>=1.8.0"
    ],
    include_package_data=True,
    python_requires=">=3.8.0",
)
