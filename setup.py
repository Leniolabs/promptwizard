from setuptools import setup

with open('/Users/camilo.basualdo/Documents/prompt-engineer/lenio-ai-prompt-engineer/doc/doc.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='lenio-ai-prompt-engineer',
    version='1.4.10',
    packages=['cli', 'cli.evals', 'cli.cost', 'cli.approximate_cost', 'cli.validation_yaml'],
    python_requires='>=3.7',
    long_description=long_description,
    long_description_content_type='text/markdown',
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