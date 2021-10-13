from os.path import exists
from setuptools import setup

setup(name='heft',
      version='0.1.1',
      description='A static scheduling heuristic',
      url='http://github.com/mroclin/heft',
      author='Matthew Rocklin',
      author_email='mrocklin@gmail.com',
      license='BSD',
      packages=['heft'],
      long_description=open('README.md').read() if exists("README.md") else "",
      zip_safe=False)
