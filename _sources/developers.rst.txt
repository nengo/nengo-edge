***************
Developer tools
***************

.. role:: raw-html(raw)
   :format: html

.. highlight:: shell

NengoEdge Tools is a Python library that contains tools and examples
to assist in taking a trained model exported from
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

The minimal NengoEdge Tools library can be installed via::

  pip install nengo-edge

Note that this installation does not include TensorFlow, which is needed for some
components of NengoEdge Tools. To install all dependencies, including TensorFlow, use::

  pip install nengo-edge[optional]

Command line interface
======================

NengoEdge Tools comes with a command line interface that can be used to perform
some useful functions:

.. click:: nengo_edge.cli:cli
   :prog: nengo-edge
   :nested: full
