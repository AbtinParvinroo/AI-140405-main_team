from setuptools import setup, find_packages

setup(
    name='data_science_helper',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'category-encoders==2.0.0',
        'matplotlib==3.9.1',
        'numpy==2.0.2',
        'pandas==2.2.2',
        'scikit-learn==1.6.1',
        'seaborn==0.13.2',
        'tensorflow==2.17.0',
        'pytest==8.4.2',
        'joblib==1.5.2',
    ],

    description='A collection of Python modules for data science tasks',
    author='Abtin Parvinroo',
    url='Your Project URL',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],

    python_requires='>=3.8',
)