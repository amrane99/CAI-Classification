from setuptools import setup, find_packages

setup(
    name='CAI-classification',
    version='0.1',
    description='A project for classifying tools in a surgical video.',
    url='https://github.com/amrane99/CAI-classification',
    keywords='python setuptools',
    packages=find_packages(include=['cai', 'cai.*']),
)
