from setuptools import setup, Extension
import numpy.distutils.core

pml = numpy.distutils.core.Extension(name='wave_1d_fd_pml.pml', sources=['wave_1d_fd_pml/pml.f90'], extra_f90_compile_args=['-O0', '-march=native', '-g', '-W', '-Wall', '-Wextra', '-pedantic', '-fbounds-check'])

numpy.distutils.core.setup(
        name='wave_1d_fd_pml',
        version='0.0.1',
        description='A comparison of PML profiles for 1D scalar wave equation finite difference simulations',
        url='https://github.com/ar4/wave_1d_fd_pml',
        author='Alan Richardson',
        author_email='alan@ausargeo.com',
        license='MIT',
        packages=['wave_1d_fd_pml'],
        install_requires=['numpy','pandas'],
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
        ],
        ext_modules=[pml]
)
