from setuptools import setup, find_packages

REQUIRED = [
    'scanpy[leiden]>=1.10.3, <=1.10.3',
    'numpy>=1.26.3, <=1.26.3',
    'numba>=0.60.0, <=0.60.0',
    'pandas>=2.2.3, <=2.2.3',
    'scipy>=1.14.1, <=1.14.1',
    'torchdata>=0.7.1, <=0.7.1',
    'pyyaml>=6.0.2, <=6.0.2',
]
setup(name='STAX',
      author='Zhen-Hao Guo',
      author_email='guozhenhao17@mails.ucas.ac.cn',
      url='https://github.com/zhanglabtools/STAX',
      version='0.0.1',
      packages=find_packages(),
      install_requires=REQUIRED,
      )
