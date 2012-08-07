#!/usr/bin/python

#
# Spit out an image of {p,q,r}
#


# TODO: brezenham
# TODO:  ./pqr.py 5 3 4 100  -- what are those dots?

import sys
from math import *
from Vec import Vec
from Vec import Mat
import hyperbolicHoneycombMeasurements
import HyperbolicIsometry


# given hyperbolic distance from origin,
# return euclidean distance from origin in poincare disk.
def h2p(h):
    return tanh(h*.5)
# given hyperbolic distance from origin,
# return euclidean distance from origin in klein disk.
def h2k(h):
    return tanh(h)

# Fudge numbers near nice numbers
def fudgeNice(x):
    for absNice in [
        0.,
        1/6.,
        1/3.,
        1/2.,
        2./3.,
        5/6.,
        1.,
        sqrt(1/6.),
        sqrt(2.)/2.,
        sqrt(2.)*2/3.,
        sqrt(2.)/3.,
        sqrt(3.)/2.,
        sqrt(3.)/3.,
        sqrt(3.)/6.,
        sqrt(2./3.),
        sqrt(5.)/5.,
        sqrt(5.)*2/5.,
        sqrt(5.)*3/10.,
    ]:
        for sign in [-1,1]:
            nice = sign * absNice
            if (x-nice)**2 + 1. == 1.:
                if x != nice:
                    #print >>sys.stderr, "    FUDGING "+`mat[i][j]`+" to "+`nice`
                    pass
                return nice
    #print "NOT FUDGING: "+`x`+" = sqrt("+`x*x`+")"
    return x
def fudgeNiceMat(mat):
    return Mat([[fudgeNice(x) for x in row] for row in mat])


# Compute the rotational symmetries of a {q,r}
# with one vertex along the x axis
# and one edge out of that vertex in the first quadrant of the xy plane
def computeRotations(q,r):

    # rGen = rotation of 2*pi/r about x axis (y towards z)
    c = cos(2*pi/r)
    s = sin(2*pi/r)
    rGen = Mat([[1,0,0],[0,c,s],[0,-s,c]])

    do('rGen')


    # compute spherical edge lengths of fundamental triangle on the sphere
    cosr01 = cos(pi/q)/sin(pi/r) # spherical half-edge-length
    cosr02 = 1./(tan(pi/q)*tan(pi/r)) # spherical face circum-radius
    cosr12 = cos(pi/r)/sin(pi/q) # spherical face in-radius
    sinr01 = sqrt(1-cosr01**2)
    sinr02 = sqrt(1-cosr02**2)
    sinr12 = sqrt(1-cosr12**2)

    do('cosr01')
    do('cosr02')
    do('cosr12')
    do('sinr01')
    do('sinr02')
    do('sinr12')

    r01 = acos(cosr01)
    r02 = acos(cosr02)
    r12 = acos(cosr12)

    # v0,v1,v2 are the verts of the fundamental triangle
    # (though we don't compute them explicitly:
    #   v0 = [1,0,0]
    #   v1 = [cos(r01), sin(r01), 0]
    #   v2 = [cos(r12),0,sin(r12)]) * (rotation taking v0 to v1)

    # qGen = rotation of 2*pi/q about v2

    xToAxis = (Mat([[cosr12,0,sinr12],[0,1,0],[-sinr12,0,cosr12]]) # x->z by r12
             * Mat([[cosr01,sinr01,0],[-sinr01,cosr01,0],[0,0,1]])) # x->y by r01
    xToAxis = fudgeNiceMat(xToAxis)
    axisToX = xToAxis.transposed()

    c = cos(2*pi/q)
    s = sin(2*pi/q)
    qGen = (axisToX
          * fudgeNiceMat(Mat([[1,0,0],[0,c,s],[0,-s,c]]))
          * xToAxis)
    qGen = fudgeNiceMat(qGen)

    do('qGen')

    # O(n^2) which is fine
    answer = [Mat([[1,0,0],[0,1,0],[0,0,1]])]
    iAnswer = 0
    while iAnswer < len(answer):
        for gen in [qGen,rGen]:
            newRot = answer[iAnswer] * gen
            newRot = fudgeNiceMat(newRot)

            foundIt = False
            for rot in answer:
                if (newRot-rot).length2() < 1e-3*1e-3:
                    foundIt = True
                    break
            if not foundIt:
                answer.append(newRot)
                assert len(answer) <= 60 # that's the max possible for any platonic solid
        iAnswer += 1

    do('len(answer)')
    return answer

def computeIsometries(p,q,r, maxIsometries):
    rotations = computeRotations(q,r)
    if False:
        for i in xrange(len(rotations)):
            do('rotations['+`i`+']')
    rotations = [HyperbolicIsometry.HyperbolicIsometry(mat) for mat in rotations]

    # compute edge lengths of fundamental tet,
    # excluding anything involving the cell center
    # (which may be infinite or ultrainfinite)
    cosh2r01,coshr01,r01 = hyperbolicHoneycombMeasurements.measure([p,q,r], 0,1)
    cosh2r02,coshr02,r02 = hyperbolicHoneycombMeasurements.measure([p,q,r], 0,2)
    cosh2r12,coshr12,r12 = hyperbolicHoneycombMeasurements.measure([p,q,r], 1,2)

    assert abs(r01.imag) < 1e-12
    assert abs(r02.imag) < 1e-12
    assert abs(r12.imag) < 1e-12
    r01 = r01.real
    r02 = r02.real
    r12 = r12.real

    # compute verts of fundamental simplex in poincare ball
    # (excluding v3 which may be infinite or ultrainfinite)
    v0 = Vec([0,0,0])
    v1 = Vec([h2p(r01),0,0])
    v2 = HyperbolicIsometry.translate(Vec([0,h2p(r12),0]),v1)

    # compute the rotation
    # of 2*pi/p about the axis from v2 to v3.
    # I.e. translate v2 to origin,
    # rotate by 2*pi/p about the z axis,
    # translate origin back to v2.
    c = cos(2*pi/p)
    s = sin(2*pi/p)
    pGen = (HyperbolicIsometry.HyperbolicIsometry(None,-v2) # translate v2 to origin
   .compose(HyperbolicIsometry.HyperbolicIsometry([[c,s,0],[-s,c,0],[0,0,1]])) # rotate by 2*pi/r about z axis (x towards y)
   .compose(HyperbolicIsometry.HyperbolicIsometry(None,v2))) # translate origin to v2

    def isometryToKey(isometry):
        # add pi to make it unlikely the number will be problematic,
        # then print to 5 decimal places
        key = ' '.join(['%.5f'%(x+pi) for x in sum([list(row) for row in isometry.R],isometry.t[:])])
        #do('key')
        return key

    answer = []
    isometryToIndex = {}

    if len(answer) == maxIsometries: return answer,len(rotations)
    for rotation in rotations:
        key = isometryToKey(rotation)
        assert key not in isometryToIndex
        isometryToIndex[key] = len(answer)
        answer.append(rotation)
        if len(answer) == maxIsometries: return answer,len(rotations)
    lo = 0
    hi = len(answer)
    while True:
        for isometry in answer[lo:hi]: # for each isometry in the previous wave
            for rotation in rotations:
                newIsometry = isometry.compose(pGen).compose(rotation)
                key = isometryToKey(newIsometry)
                if key not in isometryToIndex:
                    isometryToIndex[key] = len(answer)
                    answer.append(newIsometry)
                    if len(answer) == maxIsometries: return answer,len(rotations)
                else:
                    #print >>sys.stderr, "    DUP"
                    pass
        lo = hi
        hi = len(answer)
    return answer,len(rotations)


def lerp(a,b,t):
    return (1-t)*a + t*b

# Little test program
if __name__ == '__main__':

    def do(s):
        import inspect
        answer = eval(s, globals(), inspect.currentframe().f_back.f_locals)
        print >>sys.stderr, '            '+s+' = '+`answer`

    if len(sys.argv) not in [4,5]:
        print >>sys.stderr, "usage: "+sys.argv[0]+" p q r [maxIsometries]"
        sys.exit(1)
    p,q,r = [int(arg) for arg in sys.argv[1:4]]
    maxIsometries = 100
    if len(sys.argv) == 5:
        maxIsometries = int(sys.argv[4])
    isometries,nRotations = computeIsometries(p,q,r, maxIsometries)
    print >>sys.stderr, ''+`nRotations`+' rotations'
    print >>sys.stderr, ''+`len(isometries)`+' isometries'

    nx = ny = 257
    image = [[[0,0,0] for ix in xrange(nx)] for iy in xrange(ny)]

    # TODO: brezenham
    def drawSegment(image, x0,y0, x1,y1, color):
        ny = len(image)
        nx = len(image[0])
        n = int(nx * max(abs(x0-x1),y0-y1)) * 10 # 10 per pixel roughly
        if n == 0:
            n = 1
        for i in xrange(n+1):
            x = lerp(x0,x1,float(i)/float(n))
            y = lerp(y0,y1,float(i)/float(n))
            tx = (x+1.)*.5
            ty = (y+1.)*.5
            ix = int(round(tx*(nx-1)))
            iy = int(round(ty*(ny-1)))
            image[iy][ix] = color

    print >>sys.stderr, "drawing..."


    cosh2r01,coshr01,r01 = hyperbolicHoneycombMeasurements.measure([p,q,r], 0,1)
    assert abs(r01.imag) < 1e-12
    r01 = r01.real
    v0 = Vec([0,0,0])
    v1 = Vec([h2p(r01),0,0])
    v2 = Vec([h2p(2*r01),0,0])
    white = [255,255,255]

    for isometry in isometries:
        if True:
            w0 = isometry.apply(v0)
            w1 = isometry.apply(v1)
            w2 = isometry.apply(v2)
        else:
            w0 = isometry.t # = isometry.apply(v0)
            w1 = v1[0] * isometry.R[0] + isometry.t # = isometry.apply(v1)
            w2 = v2[0] * isometry.R[0] + isometry.t # = isometry.apply(v2)
        drawSegment(image, w0[0],w0[1], w1[0],w1[1], white)
        drawSegment(image, w1[0],w1[1], w2[0],w2[1], white)

    print >>sys.stderr, "printing..."


    # dump image in ppm format
    print "P3"
    print "# "+`p`+`q`+`r`+".ppm"
    print ''+`nx`+' '+`ny`
    print '255'
    for iy in xrange(ny):
        row = image[ny-1-iy]
        for ix in xrange(nx):
            pixel = row[ix]
            print ' ' + ' '.join(`x` for x in pixel),
        print
