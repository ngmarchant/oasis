from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='oasis',
    version='0.1.3',
    description='Optimal Asymptotic Sequential Importance Sampling',
    long_description=readme(),
    keywords='F-measure active sampling evaluation classification recall precision',
    url='http://ngmarchant.github.io/oasis',
    author='Neil G. Marchant',
    author_email='ngmarchant@gmail.com',
    license='MIT',
    packages=['oasis'],
    install_requires=[
        'numpy',
        'tables',
        'scipy',
        'sklearn'
    ],
    include_package_data=True,
    zip_safe=False,
    classifiers=['Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python',
                 'Topic :: Software Development',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Programming Language :: Python :: 3',]
    )
