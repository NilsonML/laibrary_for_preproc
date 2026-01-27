# setup.py
from setuptools import setup

setup(
    name="mpiigaze-processor",
    version="0.1.0",
    description="Библиотека для обработки данных MPIIGaze",
    author="Nilson",
    py_modules=["mpiigaze_processor"],  # Указываем единственный модуль
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
    ],
)