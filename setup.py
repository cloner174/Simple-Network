from setuptools import setup, find_packages

setup(
    name='Simple-Network',
    version='2.1.1',
    packages=find_packages(),
    author="Hamed Hajipour",
    author_email="cloner174.org@gmail.com",
    description="A basic tool designed for the construction and visualization of complex, multilayer networks.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cloner174/Simple-Network",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
