---
title: 'FURAX: A Modular JAX Toolbox for Solving Inverse Problems in Cosmology'
tags:
  - Python
  - astronomy
  - cosmology
authors:
  - name: Pierre Chanial
    orcid: 0000-0003-1753-524X
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: SciPol ERC team
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
  - given-names: Ludwig
    dropping-particle: van
    surname: Beethoven
    affiliation: 3
affiliations:
 - name: Université Paris Cité, CNRS, Astroparticule et Cosmologie, F-75013 Paris, France
   index: 1
   ror: 03tnjrr49
date: 25 May 2025
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

FURAX (Framework for Unified and Robust data Analysis with JAX) is an open-source Python library designed to address
inverse problems in astrophysics and cosmology. Built upon `JAX`, FURAX provides modular building blocks for
constructing instrument and noise models, benefiting from JAX's capabilities in automatic differentiation,
just-in-time compilation, and hardware acceleration. The framework is particularly suited for applications such as
cosmic microwave background (CMB) data analysis, including map-making and component separation tasks.

# Statement of need
Modern cosmological data analyses involve complex models and large datasets, necessitating tools that are:

    * Modular and extensible to facilitate experimentation and rapid prototyping.
    * Compatible with automatic differentiation to enable gradient-based optimization and inference.
    * Efficient and scalable, leveraging hardware acceleration for large-scale computations.

FURAX addresses these needs by providing a flexible and high-performance framework tailored for inverse problems in cosmology.
Josquin: should we add here details, specific examples of needs in the CMB data analysis context?}

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements


We acknowledge the developers of JAX and related libraries for providing the foundational tools upon which FURAX is
built. This work is supported by the European Research Council under the SciPol project. josquin{acknowledge Jean Zay}

# References
