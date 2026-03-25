Linear Operators API
====================

This section provides detailed API documentation for Furax linear operators.

Core Operators
--------------

.. currentmodule:: furax.core._base

Abstract Base Class
~~~~~~~~~~~~~~~~~~~

.. autoclass:: AbstractLinearOperator
   :members:
   :undoc-members:
   :show-inheritance:

Diagonal Operators
------------------

.. currentmodule:: furax.core._diagonal

.. autoclass:: DiagonalOperator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: BroadcastDiagonalOperator
   :members:
   :undoc-members:
   :show-inheritance:

Block Operators
---------------

.. currentmodule:: furax.core._blocks

.. autoclass:: BlockDiagonalOperator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: BlockRowOperator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: BlockColumnOperator
   :members:
   :undoc-members:
   :show-inheritance:

Dense Operators
---------------

.. currentmodule:: furax.core._dense

.. autoclass:: DenseBlockDiagonalOperator
   :members:
   :undoc-members:
   :show-inheritance:

Toeplitz Operators
------------------

.. currentmodule:: furax.core._toeplitz

.. autoclass:: SymmetricBandToeplitzOperator
   :members:
   :undoc-members:
   :show-inheritance:

Index and Reshape Operators
---------------------------

.. currentmodule:: furax.core._indices

.. autoclass:: IndexOperator
   :members:
   :undoc-members:
   :show-inheritance:

.. currentmodule:: furax.core._axes

.. autoclass:: ReshapeOperator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: MoveAxisOperator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: RavelOperator
   :members:
   :undoc-members:
   :show-inheritance:

Tree Operators
--------------

.. currentmodule:: furax.core._trees

.. autoclass:: TreeOperator
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. currentmodule:: furax._config

.. autoclass:: Config
   :members:
   :undoc-members:
   :show-inheritance:
