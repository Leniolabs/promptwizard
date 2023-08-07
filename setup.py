from setuptools import setup

setup(
    name='lenio-ai-prompt-engineer',
    version='1.2.0',
    packages=['cli', 'cli.evals', 'cli.promptChange'],
    install_requires=[
        'openai>=0.27.0',
        'PyYAML>=5.4',
        'matplotlib',
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'lenio-ai-prompt-engineer=cli.main:main',
        ],
    },
)