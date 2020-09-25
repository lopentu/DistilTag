from setuptools import setup

setup(
    name='DistilTag',
    version='0.2.1',
    author="NTUGIL LOPE Lab",
    url="https://github.com/lopentu/DistilTag",
    packages=['DistilTag'],
    setup_requires=["wheel"],
    install_requires=["gdown", "torch>=1.6", "transformers>=3.2", "numpy"],
    license='GNU GPLv3',
    long_description=open('README.md').read()
)