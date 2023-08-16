from setuptools import setup

setup(
    name='lenio-ai-prompt-engineer',
    version='1.3.1',
    packages=['cli', 'cli.evals', 'cli.promptChange', 'cli.cost', 'cli.approximate_cost', 'cli.validation_yaml'],
    install_requires=[
        'openai>=0.27.0',
        'PyYAML>=5.4',
        'matplotlib',
        'numpy',
        'marshmallow>=3.20.1',
        'tiktoken>=0.4.0'
    ],
    entry_points={
        'console_scripts': [
            'lenio-ai-prompt-engineer=cli.main:main',
        ],
    },
)