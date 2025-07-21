Linear Operators API
===================

This section provides detailed API documentation for Furax linear operators.

Core Operators
--------------

.. currentmodule:: furax.core

Abstract Base Class
~~~~~~~~~~~~~~~~~~~

.. autoclass:: _base.AbstractLinearOperator
   :members:
   :undoc-members:
   :show-inheritance:

Diagonal Operators
-----------------

.. autoclass:: _diagonal.DiagonalOperator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: _diagonal.BroadcastDiagonalOperator
   :members:
   :undoc-members:
   :show-inheritance:

Block Operators
---------------

.. autoclass:: _blocks.BlockDiagonalOperator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: _blocks.BlockRowOperator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: _blocks.BlockColumnOperator
   :members:
   :undoc-members:
   :show-inheritance:

Dense Operators
--------------

.. autoclass:: _dense.DenseOperator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: _dense.DenseBlockDiagonalOperator
   :members:
   :undoc-members:
   :show-inheritance:

Toeplitz Operators
-----------------

.. autoclass:: _toeplitz.SymmetricBandToeplitzOperator
   :members:
   :undoc-members:
   :show-inheritance:

Index and Reshape Operators
---------------------------

.. autoclass:: _indices.IndexOperator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: _axes.ReshapeOperator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: _axes.MoveAxisOperator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: _axes.RavelOperator
   :members:
   :undoc-members:
   :show-inheritance:

Tree Operators
--------------

.. autoclass:: _trees.TreeOperator
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions and Decorators
--------------------------------

Property Decorators
~~~~~~~~~~~~~~~~~~

.. currentmodule:: furax.core._base

.. autofunction:: symmetric

.. autofunction:: positive_semidefinite

.. autofunction:: diagonal

Composition Functions
~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: furax.core.rules

.. autofunction:: composition_rule

.. autofunction:: addition_rule

Utilities
~~~~~~~~

.. currentmodule:: furax.core.utils

.. autofunction:: validate_operator_compatibility

.. autofunction:: infer_operator_properties

Configuration
-------------

.. currentmodule:: furax

.. autoclass:: _config.Config
   :members:
   :undoc-members:
   :show-inheritance: