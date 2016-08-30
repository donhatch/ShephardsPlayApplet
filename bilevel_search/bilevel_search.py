#!/usr/bin/python

# This is the prototype.  Does a lot of work on implausible situations.
# Faster version is over in bilevel_search.cc.

import sys

def index_to_syndrome(nVerts, iSyndrome, syndrome):
  scratch = iSyndrome
  for i1minusi0 in xrange(0, nVerts):
    for i0 in xrange(0, nVerts-(i1minusi0)):
      i1 = i0 + i1minusi0
      nChoices = (2 if i0==i1 else 3)
      syndrome[i0][i1] = scratch % nChoices
      scratch /= nChoices
  assert scratch == 0

def syndrome2string(syndrome):
  nVerts = len(syndrome)
  answer = ''
  for i1minusi0 in xrange(0, nVerts):
    for i0 in xrange(0, nVerts-(i1minusi0)):
      i1 = i0 + i1minusi0
      if answer != '':
        answer += ' '
      answer += "["+`i0`+","+`i1`+"]:"+`syndrome[i0][i1]`
  return answer

infinity = 1000000 # effectively

def syndrome2interval(syndrome,i0,i1):
  syndromeEntry = syndrome[i0][i1]
  if i0 == i1:
    if syndromeEntry == 0:
      return [-infinity,i0]
    elif syndromeEntry == 1:
      return [i0,infinity]
    else:
      assert False
  else:
    if syndromeEntry == 0:
      return [-infinity,i0]
    elif syndromeEntry == 1:
      return [i0,i1]
    elif syndromeEntry == 2:
      return [i1,infinity]
    else:
      assert False

def calcIsPlausible(syndrome):
  verboseLevel = 0
  nVerts = len(syndrome)
  answer = True
  for i0 in xrange(nVerts):
    for i1 in xrange(i0, nVerts):
      i2 = i1+1
      for i3 in xrange(i2, nVerts):
        # is [i0,i1],[i2,i3] a violation?
        if verboseLevel >= 1: print "              checking ["+`i0`+","+`i1`+"]["+`i2`+","+`i3`+"]"
        interval01 = syndrome2interval(syndrome,i0,i1)
        interval23 = syndrome2interval(syndrome,i2,i3)
        interval03 = syndrome2interval(syndrome,i0,i3)
        if verboseLevel >= 1: print "                  interval01 = " +`interval01`
        if verboseLevel >= 1: print "                  interval23 = " +`interval23`
        if verboseLevel >= 1: print "                  interval03 = " +`interval03`
        intervalUnion = [min(interval01[0],interval23[0]),
                         max(interval01[1],interval23[1])]
        intervalIntersection = [max(interval03[0],intervalUnion[0]),
                                min(interval03[1],intervalUnion[1])]
        # intervalIntersection must have nonempty interior
        if not (intervalIntersection[0] < intervalIntersection[1]):
          if verboseLevel >= 1: print "                      bad"
          answer = False
          #assert False # coverage
          if True:
            return answer
        else:
          if verboseLevel >= 1: print "                      good"
          pass
  return answer

def calcIsGood(syndrome, gap):
  nVerts = len(syndrome)
  nVertsBeforeGap = gap
  nVertsAfterGap = nVerts - nVertsBeforeGap
  iVertJustBeforeGap = gap-1 # ok if out of bounds
  iVertJustAfterGap = gap # ok if out of bounds
  for iVertBeforeGap in xrange(nVertsBeforeGap):
    i0 = iVertBeforeGap
    i1 = iVertJustBeforeGap
    forbidden = 0
    if syndrome[i0][i1] == forbidden:
      #print "                      gap="+`gap`+": no good because syndrome["+`i0`+"]["+`i1`+"] = "+`syndrome[i0][i1]`+" == "+`forbidden`
      return False
  for iVertAfterGap in xrange(nVertsBeforeGap, nVerts):
    i0 = iVertJustAfterGap
    i1 = iVertAfterGap
    forbidden = 1 if i0==i1 else 2
    if syndrome[i0][i1] == forbidden:
      #print "                      gap="+`gap`+": no good because syndrome["+`i0`+"]["+`i1`+"] = "+`syndrome[i0][i1]`+" == "+`forbidden`
      return False
  return True

def main(argv):
  verboseLevel = 1
  if verboseLevel >= 1: print "in main"

  if len(argv) not in [2,3]:
    exit("Usage: bilevel_search.py [<minVerts>] <maxVerts>")

  minVerts = 0 if len(argv)==2 else int(argv[1])
  maxVerts = int(argv[len(argv)-1])

  for nVerts in xrange(minVerts, maxVerts+1):
    nSyndromes = 3**(nVerts*(nVerts-1)/2) * 2**nVerts
    if verboseLevel >= 1: print "      nVerts="+`nVerts`+" : "+`nSyndromes`+" syndromes"
    syndrome = [[None for i in xrange(nVerts)] for j in xrange(nVerts)] # scratch for loop. syndrome[<i][i] not used.
    nPlausible = 0
    nImplausible = 0
    nPlausibleButImpossible = 0
    nPossible = 0
    for iSyndrome in xrange(nSyndromes):
      index_to_syndrome(nVerts, iSyndrome, syndrome)
      isPlausible = calcIsPlausible(syndrome)

      if False: isPlausible = True # to exercise alarm

      if not isPlausible:
        nImplausible += 1
        if verboseLevel >= 3: print "          nVerts="+`nVerts`+" syndrome "+`iSyndrome`+"/"+`nSyndromes`+": "+syndrome2string(syndrome)+" -> IMPLAUSIBLE"
        continue
      else:
        nPlausible += 1
        if verboseLevel >= 2: print "          nVerts="+`nVerts`+" syndrome "+`iSyndrome`+"/"+`nSyndromes`+": "+syndrome2string(syndrome)+" -> plausible",
        possibleSyndrome = ''
        hasGoodGap = False
        for gap in xrange(nVerts+1):
          isGoodGap = calcIsGood(syndrome, gap)
          possibleSyndrome += '1' if isGoodGap else '0'
          if isGoodGap:
            hasGoodGap = True
        if verboseLevel >= 2 or not hasGoodGap: print "-> "+`possibleSyndrome`+('!!!!!' if not hasGoodGap else '')
        if hasGoodGap:
          nPossible += 1
        else:
          nPlausibleButImpossible += 1
    if verboseLevel >= 1: print "          "+`nImplausible`+" implausible, "+`nPlausibleButImpossible`+" plausible but impossible, "+`nPossible`+" possible"
    assert nPlausible == nPlausibleButImpossible + nPossible
  if verboseLevel >= 1: print "out main"

main(sys.argv)
