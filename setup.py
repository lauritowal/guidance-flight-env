from setuptools import setup, find_packages


setup(name='guidance_flight_env-flight-env',
      version='0.1',
      description='',
      url='',
      author='',
      license='MIT',
      install_requires=[
            'numpy',
            'gym',
            'matplotlib',
            'pandas',
            'plotly==4.14.3',
            'celluloid',
            'ray[rllib]==1.2.0',
            'jsbsim',
            'torch',
            'imageio',
            'aioredis<2.0.0'
      ],
      packages=find_packages(),
      classifiers=[
            'License :: OSI Approved :: MIT License',
            'Development Status :: 2 - Pre-Alpha',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.6',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      python_requires='>=3.6',
      include_package_data=True,
      zip_safe=False)
