#!/usr/bin/env python

'''
Please run
   pip install -e . --user
   pip uninstall candy2nd
for setuptools in develop mode, so any change in source reflects immediately

'''

from __future__ import print_function
from setuptools import find_packages
from setuptools import setup

packages = ['sctools']

with open('/home/hddl/dockerv0/second_candy/setup_log.txt', 'a') as the_file:
    the_file.write("{}".format(repr(packages)))

setup(
    name='candy2nd',
    version='0.1',
    include_package_data=True,
    packages=packages,
    description='second_candy',
)
