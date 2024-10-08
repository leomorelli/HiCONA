{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {% endfor %}
   {% endblock %}
   
   {% block methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in methods %}
      {% if item != '__init__' %}
      ~{{ name }}.{{ item }}
      {% endif %}
   {% endfor %}
   {% endblock %}


