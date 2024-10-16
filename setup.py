from setuptools import setup

setup(
    name='deepshool_hackathon',
    description='',
    author='Nick Ira',
    version="1.0.0",
    packages=[''],
    package_dir={'': 'src/max_work_dir'},
    install_requires=[
        'ultralytics==8.3.0',
        'torch_pruning',
        'cuda-python==12.3.0'
    ],
    entry_points={
        'console_scripts': [
            'run_pruner = run_pruner:run_pr',
        ]}
)