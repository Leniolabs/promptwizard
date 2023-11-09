from setuptools import setup

with open('promptwizard/doc/doc.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='promptwizard',
    version='0.0.6',
    packages=['promptwizard', 'promptwizard.evals', 'promptwizard.cost', 'promptwizard.approximate_cost', 'promptwizard.validation_yaml', 'promptwizard.prompt_generation', 'promptwizard.openai_calls'],
    python_requires='>=3.8',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'openai==1.1.1',
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
        'js2py',
        'scikit-learn'
    ],
    entry_points={
        'console_scripts': [
            'promptwizard=promptwizard.main:main',
        ],
    },
    package_data={'': ['doc/doc.md']},
    include_package_data=True,
    description='Prompt Wizard is a package for evaluating custom prompts using various evaluation methods. It allows you to provide your own prompts or generate them automatically and then obtain the results in a JSON file.',
    license='MIT',
)