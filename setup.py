from setuptools import setup, find_packages

setup(
    name='hybrid_model',  
    version='0.1.0',  
    author='Ameya Chawla',  
    author_email='ameya.chawla.ml@gmail.com', 
    description='A hybrid machine learning model combining sentence transformers with logistic regression',  
    long_description=open('README.md').read(),  
    long_description_content_type='text/markdown',  
    url='https://github.com/ameyachawlaggsipu/Hybrid_LLM_Classifier',  
    packages=find_packages(),  
    install_requires=[  
        'torch',
        'numpy',
        'scikit-learn',
        'sentence-transformers',
    ],
    python_requires='>=3.6',  
    classifiers=[ 
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
