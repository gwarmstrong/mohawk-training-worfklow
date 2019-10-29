from setuptools import setup, find_packages

setup(
    name='clippers',
    version='0.1dev',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
    ],
    author="George Armstrong",
    license='BSD-3-Clause',
    author_email="garmstro@eng.ucsd.edu",
    url="https://github.com/gwarmstrong/mohawk-training-workflow",
    description="Helps set up jobs for training mohawk models"
)
