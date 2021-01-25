from setuptools import setup, find_packages

setup(
    name='CAI-Classification -- Surgery Tool Recognition',
    version='1.0',
    description='A project for classifying tools in a surgical video.',
    url='https://github.com/amrane99/CAI-classification',
    keywords='python setuptools',
    packages=find_packages(include=['cai', 'cai.*']),
)
