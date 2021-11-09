from setuptools import find_packages, setup
setup(
    name="likecobaya",
    version="0.0",
    author= "Gabriela Marques",
    author_email = 'gmarques@fsu.edu', 
    description="clgg and clkg likelihood using pyccl and cobaya",
    zip_safe=True,
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=["cobaya>=3.0", "sacc>=0.4.5","pyccl>=2.3.0"],
    package_data={"likecobaya": ["likecobaya.py"]},
)
