from setuptools import setup, find_packages

setup(
    name='phantoms',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'pyyaml',
        'torch',
        'pytorch-lightning',
        'torch-geometric',
        'rdkit==2024.03.5',
        'wandb',
        # 'massspecgym @ git+ssh://git@github.com/Jozefov/MassSpecGymMSn.git@main#egg=massspecgym'
    ],
    author='Filip Jozefov',
    author_email='your.email@example.com',
    description='Description of the phantoms package',
    url='https://github.com/Jozefov/PhantoMS',
)