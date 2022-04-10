import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DLT",
    version="0.0.1",
    author="Tony Stark",
    author_email="manthirajak@gmail.com",
    description="Dynamic learning technique allows the user to train a model in batch wise manner",
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TONYSTARK-EDITH/Dynamic-Learning-Technique",
    keywords="DLT,Dynamic Learning Technique,python",
    install_requires=[
        'numpy',
        'scikit-learn',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
)
