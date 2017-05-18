from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
  'tensorflow==1.0.1',
]

setup(
    name='app',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    requires=[]
)

if __name__ == '__main__':
  setup(name='app', packages=['app'])
