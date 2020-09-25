from setuptools import setup

setup(
    name='DistilTag',
    version='0.2.0',
    author="NTUGIL LOPE Lab",
    url="https://github.com/lopentu/DistilTag",
    packages=['DistilTag'],
    setup_requires=["wheel"],
    install_requires=["gdown", "torch", "transformers", "numpy"],
    license='GNU GPLv3',
    long_description=open('README.md').read()
)