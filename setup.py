#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Sun Jul  8 20:35:55 CEST 2012

from setuptools import setup, find_packages

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

    name='antispoofing.lbp',
    version='1.0',
    description='Texture (LBP) based counter-measures for the REPLAY-ATTACK database',
    url='http://github.com/bioidiap/antispoofing.lbp',
    license='LICENSE.txt',
    author_email='Ivana Chingovska <ivana.chingovska@idiap.ch>',
    long_description=open('doc/howto.rst').read(),

    # This line is required for any distutils based packaging.
    packages=find_packages(),

    install_requires=[
        "bob",      # base signal proc./machine learning library
        "argparse", # better option parsing
    ],

    entry_points={
      'console_scripts': [
        'calclbp.py = antispoofing.lbp.script.calclbp:main',
        'calcframelbp.py = antispoofing.lbp.script.calcframelbp:main',
        'mkhistmodel.py = antispoofing.lbp.script.mkhistmodel:main',
        'cmphistmodels.py = antispoofing.lbp.script.cmphistmodels:main',
        'ldatrain_lbp.py = antispoofing.lbp.script.ldatrain_lbp:main',
        'svmtrain_lbp.py = antispoofing.lbp.script.svmtrain_lbp:main',
        ],
      },

)
