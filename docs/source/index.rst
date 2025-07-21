Furax Documentation
====================

**Furax** is a Framework for Unified and Robust data Analysis with JAX, providing building blocks for solving inverse problems in astrophysical and cosmological domains. It focuses on Cosmic Microwave Background (CMB) analysis with two main components:

1. **Linear Operators** (``furax.core``): Composable linear algebra operators built on top of JAX and Lineax
2. **Stokes Parameters & Landscapes** (``furax.obs``): Data structures for CMB polarization analysis

Key Features
------------

🚀 **High Performance**: Built on JAX for GPU acceleration and just-in-time compilation

🔧 **Composable Operators**: Mathematical operators that compose naturally with ``@`` and ``+``

🌌 **CMB-Ready**: Specialized data structures for polarization analysis and sky pixelization

🔬 **Scientific Computing**: Integration with HEALPix, AstroPy, and the broader scientific Python ecosystem

📊 **Flexible Analysis**: Support for component separation, mapmaking, and likelihood analysis

Getting Started
---------------

Install Furax with:

.. code-block:: bash

   pip install furax

For development or component separation features:

.. code-block:: bash

   # Development installation
   pip install -e .[dev]

   # With component separation features
   pip install -e .[comp_sep]

Quick Example
-------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from furax.obs import HealpixLandscape, Stokes
   from furax.core import DiagonalOperator

   # Create a HEALPix landscape for polarization data
   landscape = HealpixLandscape(nside=32, stokes='QU')

   # Generate random Stokes parameters
   key = jax.random.PRNGKey(42)
   stokes_data = landscape.normal(key)

   # Create a diagonal operator for weighting
   weights = jnp.ones(landscape.size)
   weight_op = DiagonalOperator(weights)

   # Apply the operator
   weighted_data = weight_op @ stokes_data

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/getting_started.md
   user_guide/data_structures.md
   user_guide/operators.md

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/component_separation.md
   examples/mapmaking.md

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/data_structures
   api/operators
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/contributing.md
   development/changelog.md

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
