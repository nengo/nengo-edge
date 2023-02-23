***************
NengoEdge Tools
***************

`NengoEdge <https://edge.nengo.ai/>`_ is a cloud-based platform for training and
deploying high accuracy, low power audio AI models on edge devices. This package
contains tools and examples to assist in taking a trained model exported from
NengoEdge and deploying it in your own application.

To get started running NengoEdge models locally,
set up a Python environment using the installation instructions below.
Then download the
`live microphone demo notebook
<https://www.nengo.ai/nengo-edge/examples/microphone-demo/microphone-demo.ipynb>`_
and open it with::

  jupyter notebook /path/to/microphone-demo.ipynb

.. image:: https://www.nengo.ai/nengo-edge/_static/demo.png
   :target: https://www.youtube.com/watch?v=sccLaootrGk

Installation
============

NengoEdge models use the `TensorFlow <https://www.tensorflow.org/>`_
machine learning library. If you already have TensorFlow installed,
then all you need is to::

  pip install nengo-edge

If you do not have TensorFlow installed, see the see the full
`installation instructions <https://www.nengo.ai/nengo-edge/installation>`_
for more details.
