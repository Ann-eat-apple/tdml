import setuptools

__version__ = '0.1.2'

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tdml",
    version=__version__,
    author="Zecheng Zhang",
    author_email="zecheng@cs.stanford.edu",
    keywords=['pandas', 'pyspark', 'numpy', 'pytorch', 'tensorflow', 'machine learning'],
    description="Transform Dataframe for Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zechengz/tdml",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'pandas',
    ],
    classifiers=[
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
