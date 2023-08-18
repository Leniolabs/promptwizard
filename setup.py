from setuptools import setup

setup(
    name='lenio-ai-prompt-engineer',
    version='1.4.7',
    packages=['cli', 'cli.evals', 'cli.promptChange', 'cli.cost', 'cli.approximate_cost', 'cli.validation_yaml'],
    python_requires='>=3.7',
    install_requires=[
        'openai>=0.27.0',
        'PyYAML>=5.4',
        'matplotlib',
        'numpy',
        'marshmallow',
        'tiktoken>=0.4.0',
        'tenacity'
    ],
    entry_points={
        'console_scripts': [
            'lenio-ai-prompt-engineer=cli.main:main',
        ],
    },
)