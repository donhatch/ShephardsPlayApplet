#!/usr/bin/python

# Read in a DUMP.off file, spit out verts and polys in java source code format.

import re
import sys

assert sys.stdin.readline() == 'OFF\n';
line = sys.stdin.readline()
assert line != ''
line = line.strip()
tokens = line.split(' ')
#print >>sys.stderr, "tokens = "+`tokens`
nVerts = int(tokens[0])
nFaces = int(tokens[1])
print >>sys.stderr, "nVerts = "+`nVerts`
print >>sys.stderr, "nFaces = "+`nFaces`
print ' '*16 + '{'
for iVerts in xrange(nVerts):
  line = sys.stdin.readline()
  assert line != ''
  line = line.strip()
  tokens = line.split(' ')
  assert len(tokens) == 3
  print ' '*20 + '{' + ', '.join(tokens) + '},'
print ' '*16 + '};'

print ' '*16 + '{'
for iFace in xrange(nFaces):
  line = sys.stdin.readline()
  assert line != ''
  line = line.strip()
  tokens = re.split(' +', line)
  assert len(tokens) >= 4
  nVertsThisFace = int(tokens[0])
  #print >>sys.stderr, "tokens = "+`tokens`
  #print >>sys.stderr, "nVertsThisFace = "+`nVertsThisFace`
  assert len(tokens) == 1 + nVertsThisFace
  print ' '*20 + '{' + ', '.join(tokens) + '},'
print ' '*16 + '};'



'''
OFF
7 7 0
0.0 0.0 -0.008020815280171303
-0.08352722772277227 -0.2019335511982571 -0.0017612635800124503
0.08352722772277227 -0.2019335511982571 -0.0017612635800124503
0.6924010344512123 0.4990795875600487 -0.049470592251842305
-0.6924010344512123 0.4990795875600487 -0.049470592251842305
0.7093047726894582 0.4989548544055984 -0.050756793434422576
-0.7093047726894582 0.4989548544055984 -0.050756793434422576
6  6 4 3 5 2 1
3  4 6 1
3  5 3 2
3  3 4 0
3  2 0 1
3  0 2 3
3  1 0 4
# Cuts: 4 2 22 18 16 13
# Folds: 0 20 9 6 10 14
'''
