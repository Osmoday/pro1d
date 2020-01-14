from setuptools import setup

setup(
   name='pro1d',
   version='1.0',
   description='',
   author='Tadeusz Pawlonka, Dmytro Shevchenko, Witold Kurpiewski',
   author_email='',
   packages=['pro1d'],
   install_requires=['pandas', 'mlxtend']
   # pip install . (the dot is important, to install globally)
   # pip install -r < requirements.txt (to deploy the virtual environment for development)
)
