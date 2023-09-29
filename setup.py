from setuptools import setup

with open('cli/doc/doc.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='lenio-ai-prompt-engineer',
    version='1.5.6',
    packages=['cli', 'cli.evals', 'cli.cost', 'cli.approximate_cost', 'cli.validation_yaml', 'cli.prompt_generation', 'cli.openai_calls'],
    python_requires='>=3.8',
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
        'python-dotenv',
        'backoff',
        'js2py'
    ],
    entry_points={
        'console_scripts': [
            'lenio-ai-prompt-engineer=cli.main:main',
        ],
    },
    package_data={'': ['doc/doc.md']},
    include_package_data=True,
    description='Prompt Engineer (lenio-ai-prompt-engineer) is a package for evaluating custom prompts using various evaluation methods. It allows you to provide your own prompts or generate them automatically and then obtain the results in a JSON file.',
)