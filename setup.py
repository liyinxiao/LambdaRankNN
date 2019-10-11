import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LambdaRankNN",
    version="0.1.1",
    author="Yinxiao Li",
    author_email="liyinxiao1227@gmail.com",
    description="LambdaRank Neural Netwrok model using Keras.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/liyinxiao/LambdaRankNN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
