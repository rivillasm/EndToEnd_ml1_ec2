from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT="-e ."
def get_requirements(file_path:str)->List[str]:
    """ will function will return the list of requirements
        read lines of the object and replaces the \n character located in each read row"""
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        [req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)  # removes the -e. from the requirements file


setup(
    name="mlproject",
    version='0.0.1',
    author="Mario",
    author_email="rivillasm@gmail.com", 
    packages=find_packages(),  # folders with init.py are treated as a package
    install_requires=get_requirements("requirements.txt")

)