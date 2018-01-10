from setuptools import setup

from rust_setuptools import (build_rust_cmdclass, build_install_lib_cmdclass,
                             RustDistribution)

setup(
    name='rust-qpick',
    version='0.0.1',
    author='Dragan Cvetinovic',
    author_email='dcvetinovic@gmail.com',
    description=('Python bindings for the rust `qpick` crate - Query Pickaxe'),
    license='MIT',
    keywords=['inverted', 'index', 'rust'],
    url='https://github.com/dncc/pyqpick',
    setup_requires=[
        'cffi>=1.0.0'],
    install_requires=['cffi>=1.0.0'],
    cffi_modules=['rust_qpick/_build_ffi.py:ffi'],
    distclass=RustDistribution,
    cmdclass={
        'build_rust': build_rust_cmdclass([('qpickwrapper', 'rust_qpick')]),
        'install_lib': build_install_lib_cmdclass()
    },
    packages=['rust_qpick'],
    zip_safe=False,
    classifiers=[
        'Development Status :: 0 - Beta',
        'License :: OSI Approved :: MIT License',
        'Topic :: Text Processing :: Indexing']
)
