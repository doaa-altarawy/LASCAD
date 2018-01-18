import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name='LASCAD',
        version="1.0.0",
        description='LASCAD',
        author='Doaa Altarawy',
        author_email='daltarawy@vt.edu',
        url="https://github.com/doaa-altarawy/LASCAD",
        license='BSD-3C',
        packages=setuptools.find_packages(),
        install_requires=[
            'numpy>=1.7',
            'pandas',
            'scikit-learn',
            'scipy',
            'seaborn',
            'matplotlib',
            'nltk',
            'whoosh'
        ],
        extras_require={
            'docs': [

            ],
            'tests': [
                'pytest',
                'pytest-cov',
                'pytest-pep8',
                'tox',
            ],
        },

        tests_require=[
            'pytest',
            'pytest-cov',
            'pytest-pep8',
            'tox',
        ],

        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
        ],
        zip_safe=True,
    )
