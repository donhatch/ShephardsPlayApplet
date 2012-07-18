#!/usr/bin/python

import sys
from math import *
from Vec import Vec
from Vec import Mat
import hyperbolicHoneycombMeasurements

# z,p are two vectors, of same dimension, of length < 1.
# transform z by the isometry of the poincare disk
# that takes the origin to p.
# In the complex plane, this transformation is z -> (z + p) / (1 + z conj(p)).
def translate(p,t):
    p = Vec(p)
    t = Vec(t)
    tt = t.dot(t)
    pt = p.dot(t)
    pp = p.dot(p)

    # Worked out on paper...
    # Also agrees with the paper "The Hyperbolic Triangle Centroid"
    # by Abraham A. Ungar.

    denominator = 1 + 2*pt + pp*tt
    tCoeff = (1 + 2*pt + pp) / denominator
    pCoeff = (1-tt) / denominator
    answer = tCoeff*t + pCoeff*p
    return answer


# A hyperbolic isometry consists
# of a row-oriented rotation matrix R (including optional reflection)
# followed by a translation vector t.
class HyperbolicIsometry:
    def __init__(self,R=None,t=None):
        if R == None: R = [[1,0,0],[0,1,0],[0,0,1]]
        if t == None: t = [0,0,0]
        self.R = Mat(R)
        self.t = Vec(t)
    def apply(self,p):
        return translate(p * self.R, self.t)
    def applyInverse(self,p):
        return self.R * translate(p,-self.t) # R * p = p * R^-1 since R is orthogonal
    def compose(self,rhs):
        lhs = self
        nDims = len(self.t)
        t = rhs.apply(self.t) # = f1(f0(0))
        R = Mat([[(1 if i==j else 0) for j in xrange(nDims)] for i in xrange(nDims)])

        for i in xrange(nDims):
            R[i] = translate(rhs.apply(lhs.apply(R[i])), -t) # R[i] = Isometry(I,t)^-1(rhs(lhs(I[i])))
        R = Mat(R)
        return HyperbolicIsometry(R,t)
    def inverse(self):
        return HyperbolicIsometry(None,t).compose(HyperbolicIsometry(R,None))
    def __repr__(self):
        return 'HyperbolicIsometry('+`self.R`+','+`self.t`+')'
    def __str__(self):
        return self.__repr__()


def h2p(h):
    return tanh(h*.5)
def h2k(h):
    return tanh(h)

def computeProjectionAtInfinity(schlafli):
    p,q,r = schlafli
    cosh2r31,coshr31,r31 = hyperbolicHoneycombMeasurements.measure(schlafli, 3,1)
    cosh2r32,coshr32,r32 = hyperbolicHoneycombMeasurements.measure(schlafli, 3,2)

    assert r31.imag == 0.
    r31 = r31.real
    assert r32.imag == 0.
    r32 = r32.real
    do('r31')
    do('r32')

    verts = []
    tris = []
    vert2index = {}
    tri2index = {}
    if p == 3 and 3 == 3:
        oneEdgeCenter = Vec(0,0,h2p(r31))
        p = Vec(sqrt(.5),sqrt(.5),0)
        v = translate(p,oneEdgeCenter)
        do('v')
        assert v[0] == v[1]
        a,a,b = v
        verts.append(Vec([a,a,b]))
        verts.append(Vec([b,a,a]))
        verts.append(Vec([a,b,a]))
        tris.append((0,1,2))

        # Generators for the rotational symmetry group...
        gens = []
        if True:
            # rotation about diagonal [1,1,1]...
            gens.append(HyperbolicIsometry([[0,1,0],[0,0,1],[1,0,0]], None))
        if True:
            # rotation about [1,1,-1]...
            gens.append(HyperbolicIsometry([[0,0,-1],[1,0,0],[0,-1,0]], None))
        if True:
            c = cos(2*pi/r)
            s = sin(2*pi/r)
            # rotation about an edge...
            gens.append(HyperbolicIsometry(None,-oneEdgeCenter) # translate edge center to origin
               .compose(HyperbolicIsometry([[sqrt(.5),-sqrt(.5),0],[sqrt(.5),sqrt(.5),0],[0,0,1]])) # rotate edge to x axis (-45 degrees about z axis)
               .compose(HyperbolicIsometry([[1,0,0],[0,c,s],[0,-s,c]])) # rotate by 2*pi/r about x axis (y towards z)
               .compose(HyperbolicIsometry([[sqrt(.5),sqrt(.5),0],[-sqrt(.5),sqrt(.5),0],[0,0,1]])) # rotate x axis back to edge (45 degrees about z axis)
               .compose(HyperbolicIsometry(None,oneEdgeCenter))) # translate origin to edge center
        do('gens')



        def findOrAddVert(vert,verts,vert2index):
            # not completely reliable but should get it in most cases
            key = tuple(['%.3f'%x for x in vert])
            #print >>sys.stderr, 'key = '+`key`
            if key in vert2index:
                return vert2index[key]
            verts.append(vert)
            vert2index[key] = len(verts)-1
            return len(verts)-1

        def findOrAddTri(tri,tris,tri2index):
            if tri in tri2index:
                return tri2index[tri]
            tris.append(tri)
            tri2index[tri] = len(tris)-1
            return len(tris)-1


        maxTris = 10000
        iTri = 0
        while True:
            if len(tris) >= maxTris:
                break
            if iTri == len(tris):
                print >>sys.stderr, "nothing more to do!"
                break
            tri = tris[iTri]
            for gen in gens:
                newTriVerts = [gen.apply(verts[iVert]) for iVert in tri]
                newTri = tuple([findOrAddVert(newVert,verts,vert2index) for newVert in newTriVerts])
                # put newTri in canonical order
                while newTri[0] != min(newTri):
                    newTri = (newTri[1],newTri[2],newTri[0])
                #do('newTri')
                findOrAddTri(newTri,tris,tri2index)
            iTri += 1


    else:
        assert False # unimplemented

    return verts,tris

def lerp(a,b,t):
    return (1-t)*a + t*b

# Little test program
if __name__ == '__main__':

    def do(s):
        import inspect
        answer = eval(s, globals(), inspect.currentframe().f_back.f_locals)
        print >>sys.stderr, '            '+s+' = '+`answer`

    if len(sys.argv) != 4:
        print >>sys.stderr, "usage: "+sys.argv[0]+" p q r"
        sys.exit(1)
    p,q,r = [int(arg) for arg in sys.argv[1:]]
    verts,tris = computeProjectionAtInfinity([p,q,r])
    print >>sys.stderr, ''+`len(verts)`+' verts'
    print >>sys.stderr, ''+`len(tris)`+' tris'

    if False:
        do('verts')
        do('tris')

    nx = 513
    ny = 513
    image = [[[0,0,0] for ix in xrange(nx)] for iy in xrange(ny)]

    def drawSegment(image, x0,y0, x1,y1, color):
        n = 1000 # number of subSegments
        ny = len(image)
        nx = len(image[0])
        for i in xrange(n+1):
            x = lerp(x0,x1,float(i)/float(n))
            y = lerp(y0,y1,float(i)/float(n))
            tx = (x+1.)*.5
            ty = (y+1.)*.5
            ix = int(round(tx*(nx-1)))
            iy = int(round(ty*(ny-1)))
            image[iy][ix] = color

    print >>sys.stderr, "drawing..."
    for drawFront in [False,True]:
        for tri in tris:
            v0,v1,v2 = [verts[i] for i in tri]
            e01 = v1-v0
            e02 = v2-v0
            normal = e01.cross(e02)
            front = normal[2] >= 0
            if front != drawFront:
                continue
            if front:
                color = [255,255,255] # white
            else:
                color = [0,0,255] # blue

            drawSegment(image, v0[0],v0[1], v1[0],v1[1], color)
            drawSegment(image, v1[0],v1[1], v2[0],v2[1], color)
            drawSegment(image, v2[0],v2[1], v0[0],v0[1], color)


    print >>sys.stderr, "printing..."



    # dump image in ppm format
    print "P3"
    print "# feep.ppm"
    print ''+`nx`+' '+`ny`
    print '255'
    for iy in xrange(ny):
        row = image[ny-1-iy]
        for ix in xrange(nx):
            pixel = row[ix]
            print ' ' + ' '.join(`x` for x in pixel),
        print







