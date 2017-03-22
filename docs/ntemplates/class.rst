:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. automethod:: __init__
   {% endblock %}

.. Not needed since there is no "Examples" section in the docstrings
.. .. include:: {{module}}.{{objname}}.examples

.. raw:: html

    <div class="clearer"></div>
