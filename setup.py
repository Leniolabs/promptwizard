from setuptools import setup, find_packages

setup(
    name='lenio-ai-prompt-engineer',
    version='0.1.0',
    packages=['cli', 'cli.evals'],
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