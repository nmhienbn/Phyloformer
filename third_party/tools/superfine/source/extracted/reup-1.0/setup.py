#!/usr/bin/env python

###########################################################################
##    Copyright 2010 Rahul Suri and Tandy Warnow.
##    This file is part of ReUP.
##
##    ReUP is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation, either version 3 of the License, or
##    (at your option) any later version.
##
##    ReUP is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with ReUP.  If not, see <http://www.gnu.org/licenses/>.
###########################################################################

from distutils.core import setup

setup(name = "reup",
      version = "1.0",
      description = "Refinement of Unresolved Phylogenies.",
      packages = ["reup"],

      url = "http://www.cs.utexas.edu/~phylo/software/superfine", 
      author = "Rahul Suri",
      author_email = "rsuri@cs.utexas.edu",

      license="General Public License (GPL)",
      requires = ["spruce (>= 1.0)"],
      provides = ["reup"],

      scripts = ["reup/scripts/runReup.py"],

      classifiers = ["Environment :: Console",
                     "Intended Audience :: Developers",
                     "Intended Audience :: Science/Research",
                     "License :: OSI Approved :: GNU General Public License (GPL)",
                     "Natural Language :: English",
                     "Operating System :: OS Independent",
                     "Programming Language :: Python",
                     "Topic :: Scientific/Engineering :: Bio-Informatics"])
