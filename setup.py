from setuptools import setup

setup(name='pyspark_kmetamodes',
      description='k-metamodes clustering for PySpark',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT',
        'Programming Language :: Python :: 3.0',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis'
      ],
      url='https://github.com/asapegin/pyspark-kmetamodes',
      license='MIT',
      packages=['pyspark_kmetamodes'],
      long_description=open('./README.rst').read(),
      include_package_data=True,
      zip_safe=False)
