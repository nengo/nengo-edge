***************
NengoEdge Tools
***************

.. role:: raw-html(raw)
   :format: html

.. highlight:: shell

`NengoEdge <https://edge.nengo.ai/>`_ is a cloud-based platform for training and
deploying high accuracy, low power audio AI models on edge devices. This package
contains tools and examples to assist in taking a trained model exported from
NengoEdge and deploying it in your own application.

To get started running NengoEdge models locally,
set up a Python environment using the installation instructions below.
Then download the
:raw-html:`<a href="examples/microphone-demo/microphone-demo.ipynb" download>live microphone demo notebook</a>`
and open it with::

  jupyter notebook /path/to/microphone-demo.ipynb

.. raw:: html

   <iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/sccLaootrGk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

Installation
============

NengoEdge models use the `TensorFlow <https://www.tensorflow.org/>`_
machine learning library. If you already have TensorFlow installed,
then all you need is to::

  pip install nengo-edge

If you do not have TensorFlow installed, see the see the full
`installation instructions <https://www.nengo.ai/nengo-edge/installation>`_
for more details.
