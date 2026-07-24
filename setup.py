from setuptools import setup, find_packages

setup(
    name='tscore',
    version='0.1.0',
    description='Tscore: transformer-based protein-protein docking scoring',
    packages=find_packages(include=['tscore', 'tscore.*']),
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.12.0',
        'torch-geometric>=2.0.0',
        'numpy',
        'scipy',
        'scikit-learn',
        'plyfile',
        'biopython',
        'open3d',
        'pandas',
        'requests',
    ],
    extras_require={
        'dev': ['pytest', 'tqdm'],
    },
)
