============
Installation
============

The OASIS package currently supports Python 3. Legacy Python (2.x) is not
supported.

Installation from source
------------------------
You can get the latest stable release from `PyPI <https://pypi.python.org/pypi>`_ by running::

    $ pip3 install oasis
    
Alternatively, you can experiment with the latest development code by cloning the source from `GitHub <https://www.github.com/ngmarchant/oasis>`_::

    $ git clone https://github.com/ngmarchant/oasis.git

You can then install the package to your user site directory as follows::

    $ cd oasis
    $ python3 setup.py install --user

Dependencies
------------
OASIS depends on the following Python packages:

* numpy
* scipy
* sklearn (to use the experimental stratification method based on K-means clustering)
* tables (to use :class:`Data` container)

These dependencies should be automatically resolved during installation.
