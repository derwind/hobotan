import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("hobotan/_version.py", "r", encoding="utf-8") as f:
    exec(f.read())

with open("requirements.txt", "r", encoding="utf-8") as f:
    install_requires = list(map(str.strip, f))

setuptools.setup(
    name = "hobotan",
    version=__version__,
    author="Shoya Yasuda",
    author_email="yasuda@vigne-cla.com",
    description="HOBO Annealing package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShoyaYasuda/hobotan",
    license="Apache 2",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 3 - Alpha",
    ]
)
