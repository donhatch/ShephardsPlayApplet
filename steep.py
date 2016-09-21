#!/usr/bin/python

# steep.py - explore a configuration that might force downward 45 degrees
# or steeper.
# Try to make both syndromes positive.
# Seems to be impossible.

import sys


def vpv(a,b):
  return [ai+bi for ai,bi in zip(a,b)]
def sxv(s,v):
  return [s*vi for vi in v]
def dot(a,b):
  return sum(ai*bi for ai,bi in zip(a,b))
def vxm(v,m):
  answer = [0] * len(m[0])
  for vi,mi in zip(v,m):
    answer = vpv(answer, sxv(vi,mi))
  return answer

def main(argv):
  if len(argv) != 4:
    return "Usage: steep.py <m> <n> <xnumerator>[/<xdenominator]"
  m = float(argv[1])
  n = float(argv[2])
  tokens = argv[3].split('/')
  assert len(tokens) == 2
  x = float(tokens[0])/float(tokens[1])

  print "m = "+`m`
  print "n = "+`n`
  print "x = "+`float(tokens[0])`+"/"+`float(tokens[1])`+" = "+`x`

  w1 = m
  w2 = (n+1.)/2.
  w3 = n*(n+1.)**2/(2*(n-1.))
  print "w1 = "+`w1`
  print "w2 = "+`w2`
  print "w3 = "+`w3`

  wsum = w1+w2+w3
  w1 /= wsum
  w2 /= wsum
  w3 /= wsum
  print "wsum = "+`wsum`
  print "w1 = "+`w1`
  print "w2 = "+`w2`
  print "w3 = "+`w3`
  v1 = [-1.,-m]
  v2 = [0,0]
  v3 = [x,0]
  print "v1 = "+`v1`
  print "v2 = "+`v2`
  print "v3 = "+`v3`
  center = vxm([w1,w2,w3],[v1,v2,v3])
  print "center = "+`center`

  syndrome1 = dot(center, [-n,1])
  syndrome2 = dot(center, [-1,n]) - dot(v3, [-1,n])
  print "syndrome1 = "+`syndrome1`
  print "syndrome2 = "+`syndrome2`

exit(main(sys.argv))
