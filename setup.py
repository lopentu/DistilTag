from distutils.core import setup

setup(
    name='DistilTag',
    version='0.1',
    author="NTUGIL LOPE Lab",
    url="https://github.com/lopentu/DistilTag",
    packages=['distiltag'],
    install_requires=["torch", "transformers", "numpy"],
    license='GNU GPLv3',
    long_description=open('README.md').read()
)