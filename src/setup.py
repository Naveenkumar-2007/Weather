
from setuptools import find_packages,setup
from typing import List

hyper_meter="-e ."
def get_requirements(file_path:str)->List[str]:

    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]
        if hyper_meter in requirements:
         requirements=requirements.remove(hyper_meter)
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='naveen',
    author_email='naveenkumarchapala123@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
