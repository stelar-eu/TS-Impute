from setuptools import setup

setup(
    name="stelarImputation",
    version="0.2.0",
    description=".",
    author="Panagiotis Betchavas",
    author_email="pbetchavas@athenarc.gr",
    python_requires='>=3.8, <4.0',
    url="",
    packages=["stelarImputation"],
    include_package_data=True,
    install_requires=[
        "jupyter==1.0.0",
        "numpy==1.24.4",
        "pandas==2.0.3",
        "scipy==1.10.1",
        "scikit-learn==1.3.2",
        "xgboost==2.0.3",
        "minio==7.2.5",
        "tqdm==4.64.1",
        "torch-geometric==2.4.0",
        "pypots @ git+https://github.com/PanosBet/PyPOTS.git",
        "notebook==6.5.2",
        "traitlets==5.9.0"
    ],
)

