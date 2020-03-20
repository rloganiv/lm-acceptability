from setuptools import setup, find_packages


setup(
    name='lm-accepability',
    version='0.0.1',
    author='rloganiv',
    author_email='rlogan@uci.edu',
    description='AllenNLP library for language model-based acceptability judgements.',
    keywords='allennlp language model',
    packages=find_packages(exclude=['tests']),
    install_requires = [
        'allennlp @ git+ssh://git@github.com/allenai/allennlp@v1.0-prerelease'
    ],
    tests_requre=['pytest'],
    python_requires='>=3.7.0'
)