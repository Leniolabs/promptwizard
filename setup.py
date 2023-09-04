from setuptools import setup

with open('cli/doc/doc.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='lenio-ai-prompt-engineer',
    version='1.4.14',
    packages=['cli', 'cli.evals', 'cli.cost', 'cli.approximate_cost', 'cli.validation_yaml', 'cli.prompt_generation'],
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
        'tenacity',
        'zipp',
        'six',
        'aiohttp',
        'urllib3',
        'certifi',
        'tqdm',
        'python-dotenv'
    ],
    entry_points={
        'console_scripts': [
            'lenio-ai-prompt-engineer=cli.main:main',
        ],
    },
    package_data={'': ['doc/doc.md']},
    include_package_data=True,
)