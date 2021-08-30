====
yabf
====


.. image:: https://img.shields.io/pypi/v/yabf.svg
    :target: https://pypi.python.org/pypi/yabf

.. image:: https://img.shields.io/travis/steven-murray/yabf.svg
    :target: https://travis-ci.org/steven-murray/yabf

.. image:: https://readthedocs.org/projects/yabf/badge/?version=latest
    :target: https://yabf.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://results.pre-commit.ci/badge/github/steven-murray/yabf/dev.svg
    :target: https://results.pre-commit.ci/latest/github/steven-murray/yabf/dev
    :alt: pre-commit.ci status



**Yet Another Bayesian Framework**


* Free software: MIT license
* Documentation: https://yabf.readthedocs.io.


Features
--------

Why another Bayesian Framework? There are a relative host_ of Bayesian codes in
Python, including major players such as PyMC3_, emcee_ and stan_, as well as a
seemingly never-ending set of scientific-field-specific codes (eg. cosmology
has CosmoMC_, CosmoHammer_, MontePython_, cobaya_...).

``yabf`` was written because the author found that all the frameowrks they tried
were either too lean or too involved. ``yabf`` tries to find the happy medium.
It won't be the right tool for everyone, but it might be the right tool for you.

``yabf`` is designed to support "black box" likelihoods, by which we mean those
that don't necessarily have analytic derivatives. This separates it from codes
such as PyMC3_ and stan_, and limits its use to samplers that do not require
that information. This is more often the case in scientific applications, where
likelihoods can in principle depend on some enormous black-box simulation code.
Thus, in this regard it is more like emcee_ or polychord_.

On the other hand, ``yabf`` is *not* another MCMC sampler. Apart from the
limitations concerning likelihood derivatives, it is sampler-agnostic. It is
rather a *specification* of a format, and an implementation of that specification.
That is, it specifies that likelihoods should have certain properties (like
parameters), and gives tools that enable that. Or as another example, it
specifies that samplers should contain certain attributes pre- and post-sampling.
In this regard, ``yabf`` *is* more like PyMC3_ or stan_, and unlike emcee_ or
polychord_.

``yabf`` is perhaps most similar to codes such as CosmoHammer_ or cobaya_,
which provide an interface for creating (cosmological) likelihoods which can
then be sampled by somie specified sampler. However, ``yabf`` is different in
that it is *intended* to be field-agnostic, and entirely general. In addition,
I found that these codes didn't quite satisfy my criteria for ease-of-use
and extensibility.

I hope that ``yabf`` provides these. Here are a few of its features:

* Deisgn is both DRY/modular and easy-to-use: while components of the model can
  be separately defined (to make it DRY), they don't *need* to be combined into
  a rigid structure in order to perform most calculations. This makes it easy
  to evaluate partial models for debugging.
* Extremely extensible: write your own class that subclasses from the in-built
  ``Component`` or ``Likelihood`` classes, and it is immediately useable.
* Parameters are attached to to the model, for encapsulation, but they can be
  specified at run-time externally for modularity.
* Models are heirarchical, in the sense that parameters may be specified at
  any of three levels, and they are propagated through the model heirarchy (note
  that this doesn't refer to heirarchical parameters, i.e. parameters that
  depend on other parameters).
* Parameters can be set as fixed or constrained at run-time.
* Models are well-specified, in the sense that they can be entirely specified
  by a YAML file (and/or written to YAML file), for reproducibility.

Credits
-------

This package was created with Cookiecutter_ and the
`audreyr/cookiecutter-pypackage`_ project template.

Many of the ideas in this code are adaptations of other MCMC codes, especially
CosmoHammer_ and cobaya_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _host: https://github.com/Gabriel-p/pythonMCMC
.. _PyMC3: https://docs.pymc.io/
.. _emcee: https://emcee.readthedocs.io/en/latest/tutorials/quickstart/
.. _stan: https://pystan.readthedocs.io/en/latest/
.. _CosmoMC: https://cosmologist.info/cosmomc/
.. _CosmoHammer: https://github.com/cosmo-ethz/CosmoHammer
.. _MontePython: http://baudren.github.io/montepython.html
.. _cobaya: https://cobaya.readthedocs.io/en/latest/
.. _polychord: https://github.com/PolyChord/PolyChordLite
