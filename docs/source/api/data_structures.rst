Data Structures API
==================

This section provides detailed API documentation for Furax data structures.

Stokes Parameters
-----------------

.. currentmodule:: furax.obs

Abstract Base Class
~~~~~~~~~~~~~~~~~~~

.. autoclass:: stokes.Stokes
   :members:
   :undoc-members:
   :show-inheritance:

Stokes Implementations
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: stokes.StokesI
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: stokes.StokesQU
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: stokes.StokesIQU
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: stokes.StokesIQUV
   :members:
   :undoc-members:
   :show-inheritance:

Landscapes
----------

.. currentmodule:: furax.obs

Abstract Base Classes
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: landscapes.Landscape
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: landscapes.StokesLandscape
   :members:
   :undoc-members:
   :show-inheritance:

Implementations
~~~~~~~~~~~~~~

.. autoclass:: landscapes.HealpixLandscape
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: landscapes.FrequencyLandscape
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

.. currentmodule:: furax.obs.stokes

.. autofunction:: _validate_stokes_arrays

.. currentmodule:: furax.obs.landscapes

.. autofunction:: _validate_nside
