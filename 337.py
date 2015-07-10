#!/usr/bin/python

#
# Spit out an image of the surface of {3,3,7}
# XXX TODO: is this obsoleted by pqr.py?
#

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
        v = HyperbolicIsometry.translate(p,oneEdgeCenter)
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
            gens.append(HyperbolicIsometry.HyperbolicIsometry([[0,1,0],[0,0,1],[1,0,0]], None))
        if True:
            # rotation about [1,1,-1]...
            gens.append(HyperbolicIsometry.HyperbolicIsometry([[0,0,-1],[1,0,0],[0,-1,0]], None))
        if True:
            c = cos(2*pi/r)
            s = sin(2*pi/r)
            # rotation about an edge...
            gens.append(HyperbolicIsometry.HyperbolicIsometry(None,-oneEdgeCenter) # translate edge center to origin
               .compose(HyperbolicIsometry.HyperbolicIsometry([[sqrt(.5),-sqrt(.5),0],[sqrt(.5),sqrt(.5),0],[0,0,1]])) # rotate edge to x axis (-45 degrees about z axis)
               .compose(HyperbolicIsometry.HyperbolicIsometry([[1,0,0],[0,c,s],[0,-s,c]])) # rotate by 2*pi/r about x axis (y towards z)
               .compose(HyperbolicIsometry.HyperbolicIsometry([[sqrt(.5),sqrt(.5),0],[-sqrt(.5),sqrt(.5),0],[0,0,1]])) # rotate x axis back to edge (45 degrees about z axis)
               .compose(HyperbolicIsometry.HyperbolicIsometry(None,oneEdgeCenter))) # translate origin to edge center
        do('gens')



        def findOrAddVert(vert,verts,vert2index):
            # not completely reliable but should get it in most cases
            key = tuple(['%.6f'%x for x in vert])
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


        maxTris = 100000
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
    print "# "+`p`+`q`+`r`.ppm"
    print ''+`nx`+' '+`ny`
    print '255'
    for iy in xrange(ny):
        row = image[ny-1-iy]
        for ix in xrange(nx):
            pixel = row[ix]
            print ' ' + ' '.join(`x` for x in pixel),
        print








