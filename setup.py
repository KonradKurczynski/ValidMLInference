from setuptools import setup, find_packages

setup(
    name="ValidMLInference",
    version="0.0.1",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=["numpy", "scipy", "numdifftools", "jax", "jaxopt"],
)