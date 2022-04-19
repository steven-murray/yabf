=======
History
=======

v2.2.0
------

Added
-----

* Ability to set the YAML loader when loading likelihoods from YAML. This can be done
  programmatically or via CLI by using eg. ``--yaml-loader astropy.io.misc.AstropyLoader``

v2.1.1
------

Fixed
-----

* Issue with using polychord and a ``LikelihoodContainer``.

v2.0.1
------

Fixed
-----

* Params with ``transforms`` now give the right bounds.

v2.0.0 [25 Oct 2021]
--------------------

Fixed
-----

* Bayesian evidence computed correctly in polychord now (with tests)
* polychord now saves into the correct directory.


v1.0.0 [18 May 2021]
----------------------

Changed
~~~~~~~

* Nicer YAML interface
* Better data loading interface


0.0.1 (2019-04-25)
------------------

* First release on PyPI.
