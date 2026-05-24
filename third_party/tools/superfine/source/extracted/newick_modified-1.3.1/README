
  This module contains a python parser for the Newick tree format.  It
reads in trees like:

(
  ('Chimp' 1 : 0.052,
   'Human' 1 : 0.042) 0.71 : 0.007,
  'Gorilla' 1 : 0.060,
  ('Gibbon' 1 : 0.124,
   'Orangutan' 1 : 0.0971) 1 : 0.038
);

where parenthesis denotes the sub-trees and the edges are annotated
"bootstrap-value : length", where both bootstrap value and length is
optional.

  The simplest usage is to load the parser from newick.tree and read
the tree from a string:

from newick.tree import parse_tree
print parse_tree("('A', 'B')")

  A tree parsed this way can then be traversed and manipulated.


  By using "handlers" it is possible to extract information from the
tree--through call-backs--without first building the entire tree.  The
following example uses this to calculate the total branch sum:

import newick
import sys

class BranchLengthSum(newick.AbstractHandler):
    def __init__(self):
        self.sum = 0.0

    def new_edge(self,b,l):
        if l:
            self.sum += l

    def get_sum(self):
        return self.sum

input = sys.stdin.read()
lexer = newick.Lexer(input)
handler = BranchLengthSum()
parser  = newick.Parser(lexer,handler)

parser.parse()
print handler.get_sum()
