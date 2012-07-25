#!/usr/bin/env python

from math import *
import cmath
import inspect
from Vec import Vec

# A * B^T
def mxmt(A,B):
    return [[a.dot(b) for b in B] for a in A]
def det(M):
    if len(M) == 0:
        return 1.
    if len(M) == 1:
        return M[0][0]
    if len(M) == 2:
        return M[0][0]*M[1][1]-M[0][1]*M[1][0]
    if len(M) == 3:
        return (M[0][0]*M[1][1]*M[2][2]
              + M[0][1]*M[1][2]*M[2][0]
              + M[0][2]*M[1][0]*M[2][1]
              - M[0][0]*M[1][2]*M[2][1]
              - M[0][1]*M[1][0]*M[2][2]
              - M[0][2]*M[1][1]*M[2][0])
    assert False

# z,p are two vectors, of same dimension, of length < 1.
# transform z by the isometry of the poincare disk
# that takes the origin to p.
# In the complex plane, this transformation is z -> (z + p) / (1 + z conj(p)).
def xformComplex(z,p):
    answer = (z+p)/(1+z*p.conjugate())
    return answer
def xformVec(z,p):
    z = Vec(z)
    p = Vec(p)
    pp = p.dot(p)
    zp = z.dot(p)
    zz = z.dot(z)

    # Worked out on paper...
    # Also agrees with the paper "The Hyperbolic Triangle Centroid"
    # by Abraham A. Ungar.

    denominator = 1 + 2*zp + zz*pp
    pCoeff = (1 + 2*zp + zz) / denominator
    zCoeff = (1-pp) / denominator
    answer = pCoeff*p + zCoeff*z
    return answer

def c2v(c):
    #return [(c+c.conjugate())*.5, (c-c.conjugate())*.5]
    return Vec([c.real, c.imag])
def v2c(v):
    return complex(v[0],v[1])

def ptok(p):
    return htwice(p)
def ktop(k):
    return hhalf(k)

def xform(z,p):
    tol = 1e-2 # XXX hmm, one or both of those formulas obviously isn't robust
    if isinstance(z,list): # list or Vec
        answer = xformVec(z,p)
        assert len(z) == len(p)
        if len(z) == 2:
            Answer = c2v(xformComplex(v2c(z),v2c(p)))
            #print "z = "+`z`
            #print "p = "+`p`
            #print "Answer = "+`Answer`
            #print "answer = "+`answer`
            assert abs(v2c(Answer)-v2c(answer)) <= tol
    elif type(z) in [int,float,complex]:
        answer = xformComplex(z,p)
        Answer = v2c(xformVec(c2v(z),c2v(p)))
        assert abs(answer-Answer) <= tol
    else:
        print type(z)
        assert False

    return answer

def xformKleinCheat(z,p):
    return ptok(xform(ktop(z),ktop(p)))

# should allow z,p to be ideal,
# as long as they are not opposites.
# in particular, if p is ideal, everything should go to p (except z=-p)
def xformKlein(z,p):
    if type(z) == complex:
        return v2c(xformKlein(c2v(z),c2v(p)))
    if type(z) == list:
        z = Vec(z)
        p = Vec(p)

    # formula (13) from paper "The Hyperbolic Triangle Centroid"
    # rearranged so it's robust even for ideal points
    zp = z.dot(p)
    pp = p.dot(p)
    denominator = 1 + zp
    pCoeff = (1 + zp/(1+sqrt(1-pp)) ) / denominator
    zCoeff = sqrt(1-pp) / denominator
    answer = pCoeff*p + zCoeff*z
    do('answer')
    return answer

def invGammaKleinSegment(a,b):
    if type(a) == complex:
        a = c2v(a)
        b = c2v(b)
    if type(a) == list:
        a = Vec(a)
        b = Vec(b)

    assert type(a) == Vec


    if False:
        ab = xformKlein(-a,b)
        return invGamma(ab)

    aa = a.dot(a)
    ab = a.dot(b)
    bb = b.dot(b)

    print "-------"
    if True:
        if False:
            foo = (1 - ab/(1+sqrt(1-bb)))*b - sqrt(1-bb)*a

            length2foo = length2(foo)
            print "-------"
            do('length2foo')
            length2foo = (1 - ab/(1+sqrt(1-bb)))**2*bb + (1-bb)*aa - 2*(1 - ab/(1+sqrt(1-bb)))*sqrt(1-bb)*ab
            do('length2foo')
            length2foo = (1 + (ab/(1+sqrt(1-bb)))**2 - 2*ab/(1+sqrt(1-bb))) * bb + (1-bb)*aa - 2*(1 - ab/(1+sqrt(1-bb)))*sqrt(1-bb)*ab
            do('length2foo')
            length2foo = (1 + ab**2/(1+sqrt(1-bb))**2 - 2*ab/(1+sqrt(1-bb))) * bb + (1-bb)*aa - (2*ab*sqrt(1-bb) - 2*ab**2/(1+sqrt(1-bb))*sqrt(1-bb))
            do('length2foo')
            length2foo = (aa + bb - aa*bb
                        + bb*ab**2/(1+sqrt(1-bb))**2 - 2*bb*ab/(1+sqrt(1-bb)) - 2*ab*sqrt(1-bb) + 2*ab**2*sqrt(1-bb)/(1+sqrt(1-bb)))
            do('length2foo')
            # why is it symmetric in a,b?? it doesn't look it, yet
            length2foo = (bb + aa - bb*aa
                        + aa*ab**2/(1+sqrt(1-aa))**2 - 2*aa*ab/(1+sqrt(1-aa)) - 2*ab*sqrt(1-aa) + 2*ab**2*sqrt(1-aa)/(1+sqrt(1-aa)))
            do('length2foo')

        length2foo = (bb + aa - bb*aa + ab*(
            aa*ab/(1+sqrt(1-aa))**2 - 2*aa/(1+sqrt(1-aa)) - 2*sqrt(1-aa) + 2*ab*sqrt(1-aa)/(1+sqrt(1-aa))
        ))
        if False:
            do('length2foo')


        answer = sqrt(1 - length2foo/(1-ab)**2)
        do('answer')
    if False:
        # This was was right but inaccurate
        # convert to poincare disk and do calculation there.
        # A and B are the points in the poincare disk.
        A = hhalf(a)
        B = hhalf(b)
        AA = A.dot(A)
        AB = A.dot(B)
        BB = B.dot(B)
        

        # pp = || (A-B) / (1 - A*conj(B)) ||
        #    = ||A-B|| / ((1-A*conj(B)) * (1-B*conj(A)))
        #    = ||A-B|| / ((1-A*conj(B)) * (1-B*conj(A)))
        #    = ||A-B|| / (1-2*(A dot B) + (A dot A)*(B dot B))
        pp = length2(A-B)/(1 - 2*AB + AA*BB)

        # given p poincare,
        #     k = 2*p/(1+pp)
        # so kk = 4*pp/(1+pp)^2
        kk = 4*pp/(1+pp)**2

        answer = sqrt(1-kk)
        do('answer')
    if True:
        # convert to poincare disk and do calculation there
        A = a / (1+sqrt(1-length2(a)))
        B = b / (1+sqrt(1-length2(b)))
        AA = A.dot(A)
        AB = A.dot(B)
        BB = B.dot(B)
        
        # given p poincare,
        #     k = 2*p/(1+pp)
        # so sqrt(1-kk) = sqrt((1-k)*(1+k))

        pp = length2(A-B)/(1-2*AB+AA*BB)
        k = 2*sqrt(pp)/(1+pp)
        answer = sqrt((1-k)*(1+k))

        do('answer')
    if True:
        aa = a.dot(a)
        ab = a.dot(b)
        bb = b.dot(b)

        afoo = (1+sqrt(1-aa))
        bfoo = (1+sqrt(1-bb))

        afoo2 = (1+(1-aa)+2*sqrt(1-aa))
        bfoo2 = (1+(1-bb)+2*sqrt(1-bb))
        afoo2 = (2-aa+2*sqrt(1-aa))
        bfoo2 = (2-bb+2*sqrt(1-bb))


        pp = (aa*bfoo2-2*ab*afoo*bfoo+bb*afoo2)/(afoo2*bfoo2-2*ab*afoo*bfoo+aa*bb)

        k = 2*sqrt(pp)/(1+pp)
        answer = sqrt((1-k)*(1+k))

        do('answer')
    if True:
        aa = a.dot(a)
        ab = a.dot(b)
        bb = b.dot(b)

        afoo = (1+sqrt(1-aa))
        bfoo = (1+sqrt(1-bb))

        afoo2 = (2-aa+2*sqrt(1-aa))
        bfoo2 = (2-bb+2*sqrt(1-bb))


        pp = (aa*(2-bb+2*sqrt(1-bb))-2*ab*afoo*bfoo+bb*(2-aa+2*sqrt(1-aa))) / ((2-aa+2*sqrt(1-aa))*(2-bb+2*sqrt(1-bb))-2*ab*afoo*bfoo+aa*bb)
        pp = (2*aa-2*bb*aa+2*aa*sqrt(1-bb)-2*ab*afoo*bfoo+2*bb+2*bb*sqrt(1-aa)) / ((2-aa+2*sqrt(1-aa))*(2-bb+2*sqrt(1-bb))-2*ab*afoo*bfoo+aa*bb)
        pp = (2*aa-2*bb*aa+2*bb + 2*aa*sqrt(1-bb) - 2*ab*(1+sqrt(1-aa))*(1+sqrt(1-bb)) + 2*bb*sqrt(1-aa)) / ((2-aa+2*sqrt(1-aa))*(2-bb+2*sqrt(1-bb))-2*ab*(1+sqrt(1-aa))*(1+sqrt(1-bb))+aa*bb)

        k = 2*sqrt(pp)/(1+pp)
        answer = sqrt((1-k)*(1+k))

        do('answer')
    if False:
        aa = a.dot(a)
        ab = a.dot(b)
        bb = b.dot(b)

        pp = (aa-bb*aa+bb + aa*sqrt(1-bb) - ab*(1+sqrt(1-aa))*(1+sqrt(1-bb)) + bb*sqrt(1-aa)) / (  2 + aa*bb + 2*sqrt(1-aa)*sqrt(1-bb) - aa - bb + 2*sqrt*(1-aa) + 2*sqrt(1-bb) -aa*sqrt(1-bb) -bb*sqrt(1-aa) -ab*(1+sqrt(1-aa))*(1+sqrt(1-bb)))

        k = 2*sqrt(pp)/(1+pp)
        answer = sqrt((1-k)*(1+k))

        do('answer')
    if False:
        aa = a.dot(a)
        ab = a.dot(b)
        bb = b.dot(b)

        pp = (aa-bb*aa+bb + aa*sqrt(1-bb) - ab*(1+sqrt(1-aa))*(1+sqrt(1-bb)) + bb*sqrt(1-aa)) / (  2 + aa*bb + 2*sqrt(1-aa)*sqrt(1-bb) - aa - bb + 2*sqrt*(1-aa) + 2*sqrt(1-bb) -aa*sqrt(1-bb) -bb*sqrt(1-aa) -ab*(1+sqrt(1-aa))*(1+sqrt(1-bb)))

        k = 2*sqrt(pp)/(1+pp)
        answer = sqrt((1-k)*(1+k))

        do('answer')
    if True:
        # and then a miracle happens... I read the book. easy as pie.
        # gamma(xform(b,-a)) = gamma(a)*gamma(b)*(1 - (a dot b))
        # so, invGamma(xform(b,-a)) = invGamma(a)*invGamma(b)/(1-(a dot b))
        answer = invGamma(a)*invGamma(b)/(1-a.dot(b))
        answer = sqrt((1-length2(a))*(1-length2(b)))/(1-a.dot(b))
        do('answer')

    # IDEA: can we just compute the coeffs in poincare space? might be easier


    return answer


def length2(v):
    if type(v) in [int,long,float,complex]:
        return abs(v)**2
    elif type(v) == Vec:
        return v.length2()
    else:
        return Vec(v).length2()

def htwice(v):
    return v * (2/(1+length2(v)))
def hhalf(v):
    return v / (1+sqrt(1-length2(v)))

def havg(a,b):
    if isinstance(a,list): # list or Vec
        return c2v(havg(v2c(a),v2c(b)))

    print xform(b,-a)

    return xform(hhalf(xform(b, -a)), a)

# Given euclidean center and radius,
# find the poincare center.
# Probably not really useful,
# since we are only dealing with one size disk (in hyperbolic metric)
def e2pcenter(x,r):
    answer = havg(x-r, x+r)
    do('answer')
    v = xform(x+r, -(x-r))
    do('v')
    v = (x+r-(x-r)) / (1 + (x+r)*(-x+r))
    do('v')
    v = (2*r / (1 + (x+r)*(-x+r)))
    do('v')
    v = (2*r / (1 -x*x + r*r))
    do('v')
    do('xform(v/(1+sqrt(1-v*v)), x-r)')
    do('(x-r+v/(1+sqrt(1-v*v))) / (1 + (x-r)*v/(1+sqrt(1-v*v)))')
    do('(x-r+(2*r / (1 - x*x + r*r ))/(1+sqrt(1-(2*r / (1 - x*x + r*r ))**2))) \
      / (1 + (x-r)*(2*r / (1 - x*x + r*r ))/(1+sqrt(1-(2*r / (1 - x*x + r*r ))**2)))')

    do('((x-r)*(1 - x*x + r*r )*(1+sqrt(1-(2*r / (1 - x*x + r*r ))**2)) + 2*r) \
      / ((1 - x*x + r*r )*(1+sqrt(1-(2*r / (1 - x*x + r*r ))**2)) + (x-r)*2*r)')

    do('((x-r)*(1 - x*x + r*r )*(1+sqrt(1-(2*r / (1 - x*x + r*r ))**2)) + 2*r) \
      / ((1 - x*x + r*r )*(1+sqrt(1-(2*r / (1 - x*x + r*r ))**2)) + (x-r)*2*r)')

    do('((x-r)*((1-x*x+r*r)+sqrt((1-x*x+r*r)**2-(2*r)**2)) + 2*r) \
      / ((1 - x*x + r*r )*(1+sqrt(1-(2*r / (1 - x*x + r*r ))**2)) + (x-r)*2*r)')

    do('((x-r)*((1-x*x+r*r)+sqrt((1-x*x+r*r)**2-(2*r)**2)) + 2*r) \
      / ((1-x*x+r*r)+sqrt((1-x*x+r*r)**2-(2*r)**2) + (x-r)*2*r)')

    do('((x-r) * (1 - x*x + r*r + sqrt((1-x*x+r*r)**2-(2*r)**2)) + 2*r) \
      / (1-x*x-r*r+2*r*x + sqrt((1-x*x+r*r)**2-(2*r)**2))')
    do('((x-r) * (1 - x*x + r*r) + (x-r)* sqrt((1-x*x+r*r)**2-(2*r)**2) + 2*r) \
      / (1-x*x-r*r+2*r*x + sqrt((1-x*x+r*r)**2-(2*r)**2))')

    do('(x-x*x*x+x*r*r-r+r*x*x-r*r*r + (x-r)* sqrt((1-x*x+r*r)**2-(2*r)**2) + 2*r) \
      / (1-x*x-r*r+2*r*x + sqrt((1-x*x+r*r)**2-(2*r)**2))')

    do('(x-x*x*x+x*r*r+r+r*x*x-r*r*r + (x-r)* sqrt((1-x*x+r*r)**2-(2*r)**2)) \
      / (1-x*x-r*r+2*r*x + sqrt((1-x*x+r*r)**2-(2*r)**2))')

    answer = ((x-x*x*x+x*r*r+r+r*x*x-r*r*r + (x-r)* sqrt((1-x*x+r*r)**2-(2*r)**2))
            / (1-x*x-r*r+2*r*x + sqrt((1-x*x+r*r)**2-(2*r)**2)))
    do('answer')
    return answer


def reflect0(z,p):
    z = v2c(z)
    p = v2c(p)

    z = xform(z,-p)
    z = -z.conjugate()*p/p.conjugate()
    z = xform(z,p)

    z = c2v(z)
    return z

def reflect1(z,p):
    z = v2c(z)
    p = v2c(p)

    z = -xform(z,-p).conjugate()*p/p.conjugate()
    z = xform(z,p)

    z = c2v(z)
    return z

def c(z): return z.conjugate()
def reflect2(z,p):
    z = v2c(z)
    p = v2c(p)

    #z = xform(-xform(z,-p).conjugate()*p/p.conjugate(), p)
    #z = xform(-((z-p)/(1-z*p.conjugate())).conjugate()*p/p.conjugate(), p)
    #z = (-((z-p)/(1-z*p.conjugate())).conjugate()*p/p.conjugate() + p) / (1 + (-((z-p)/(1-z*p.conjugate())).conjugate()*p/p.conjugate())*p.conjugate())
    #z = (-((z-p)/(1-z*p.conjugate())).conjugate()*p/p.conjugate() + p) / (1 + (-((z-p)/(1-z*p.conjugate())).conjugate()*p))
    #z = (-c((z-p)/(1-z*c(p)))*p/c(p) + p) / (1 + (-c((z-p)/(1-z*c(p)))*p))
    #z = (-((c(z)-c(p))/(1-c(z)*p))*p/c(p) + p) / (1 - ((c(z)-c(p))/(1-c(z)*p))*p)
    #z = (-((c(z)-c(p))/(1-c(z)*p))*p + p*c(p)) / (c(p) - ((c(z)-c(p))/(1-c(z)*p))*c(p)*p)
    #z = (-((c(z)-c(p))/(1-c(z)*p))*p + p*c(p)) / (c(p) - ((c(z)-c(p))/(1-c(z)*p))*c(p)*p)
    #z = (-(c(z)-c(p))*p + (1-c(z)*p)*p*c(p)) / (c(p)*(1-c(z)*p) - ((c(z)-c(p)))*c(p)*p)
    #z = (-c(z)*p+c(p)*p + p*c(p) - c(z)*p*p*c(p)) / (c(p)*(1-c(z)*p) - ((c(z)-c(p)))*c(p)*p)
    #z = (-c(z)*p+c(p)*p + p*c(p) - c(z)*p*p*c(p)) / (c(p) - c(p)*c(z)*p - (c(z)*c(p)*p-c(p)*c(p)*p))
    #z = (-c(z)*p + c(p)*p + p*c(p) - c(z)*p*p*c(p)) / (c(p) - c(p)*c(z)*p - c(z)*c(p)*p + c(p)*c(p)*p)
    #z = (-c(z)*p + 2*c(p)*p - c(z)*p*p*c(p)) / (c(p) - 2*c(z)*c(p)*p + c(p)*c(p)*p)
    #z = (-c(z) + 2*c(p) - c(z)*p*c(p)) / (c(p)/p - 2*c(z)*c(p) + c(p)*c(p))
    #z = (c(z) - 2*c(p) + c(z)*p*c(p)) / (2*c(z)*c(p) -c(p)/p - c(p)*c(p))
    #z = (c(z)/c(p) - 2 + c(z)*p) / (2*c(z) -1/p - c(p))
    #z = (c(z)*(p+1/c(p)) - 2) / (2*c(z) -1/p - c(p))
    #z = (c(z)*(p+1/c(p)) - 2) / (2*c(z) - (c(p) + 1/p))
    #z = (c(z)*(p+1/c(p))/2 - 1) / (c(z) - (c(p)+1/p)/2 )



    # hmm, use fact that z*c(z) == 1:
    assert abs(abs(z)-1) < 1e-9
    #z = ((p+1/c(p))/2 - z) / (1 - z*(c(p)+1/p)/2)
    #z = (2*z - (p+1/c(p))) / (-2 + z*(c(p) + 1/p))
    z = - (z - (p+1/c(p))/2 ) / (1 - z*(c(p)+1/p)/2 )




    # wtf? can't we get something nicer than that?

    # okay wait...
    # isn't it just conjugation, followed by multiplying by -p^2/|p|^2
    # followed by "translation" twice by p (i.e. translation by 2p/(1+|p|^2))?
    #z = (-c(z)*p/c(p) + 2*p/(1+p*c(p))) / (1 + -c(z)*p/c(p) * (2*c(p)/(1+c(p)*p)))
    # but that didn't help make it simpler.
    #z = (-c(z)*p/c(p) + 2/(1/p+c(p))) / (1 + -c(z)*p/c(p) * 2/(1/c(p)+p))


    z = c2v(z)
    return z


def areaSquared(verts):
    edgeVecs = [vert-verts[0] for vert in verts[1:]]
    M = mxmt(edgeVecs, edgeVecs)
    return det(M)

def idealSimplexCenter(verts):
    if type(verts[0]) == complex:
        return v2c(idealSimplexCenter([c2v(vert) for vert in verts]))
    if type(verts[0]) == list:
        verts = [Vec(vert) for vert in verts]
    print "    in idealSimplexCenter"
    # fix non-units, so caller can be sloppy
    verts = [vert.normalized() for vert in verts]
    do('verts')

    # actually we can get this way more directly,
    # as (inverses, maybe, of) the lengths of the columns
    # of the inverse of the edge vec matrix...
    # but it breaks down when not invertible.  really need adjugate matrix or something?
    # do we know how to calculate adjugate in general?
    facetAreasSquared = [areaSquared([verts[j] for j in xrange(len(verts)) if j != i]) for i in xrange(len(verts))]
    do('facetAreasSquared')
    denominator = sum(facetAreasSquared)
    kleinCenter = sum([facetAreaSquared/denominator * v for facetAreaSquared,v in zip(facetAreasSquared,verts)])

    do('kleinCenter')
    poincareCenter = hhalf(kleinCenter)
    do('poincareCenter')

    # Sanity check:
    # if we transform the verts by minus the poincare center,
    # the results should average to 0
    p = poincareCenter
    do('sum([xform(z,-p) for z in verts])')

    print "    out idealSimplexCenter"
    return poincareCenter

# This works!!
def idealTriangleCenterSmart(a,b,c):
    if type(a) == complex:
        return v2c(idealTriangleCenterSmart(c2v(a),c2v(b),c2v(c)))
    if type(a) != Vec:
        a = Vec(a)
        b = Vec(b)
        c = Vec(c)
    # fix non-units, so caller can be sloppy
    a = a.normalized()
    b = b.normalized()
    c = c.normalized()
    print "    in idealTriangleCenterSmart"

    # Either of the following works...

    if False:
        aCoeff = 1-b.dot(c)
        bCoeff = 1-c.dot(a)
        cCoeff = 1-a.dot(b)
    else:
        # better
        aCoeff = (c-b).length2()
        bCoeff = (a-c).length2()
        cCoeff = (b-a).length2()
    denominator = aCoeff + bCoeff + cCoeff
    aCoeff /= denominator
    bCoeff /= denominator
    cCoeff /= denominator

    kleinCenter = a*aCoeff + b*bCoeff + c*cCoeff
    poincareCenter = hhalf(kleinCenter)

    p = poincareCenter
    do('xform(a,-p)+xform(b,-p)+xform(c,-p)')


    # poincareCenter = hhalf(kleinCenter) = kleinCenter / (1+sqrt(1-length2(kleinCenter)))
    # We need a better way of estimating 1-length2(kleinCenter).
    # Even if kleinCenter is unstable due to a,b,c being close together, 1-length2(kleinCenter) should be totally stable.
    # 1-length2(kleinCenter)
    # = 1-kleinCenter.dot(kleinCenter)
    # = 1 - (A*a+B*b+B*c).dot(A*a+B*b+C*c)
    # = 1 - (A^2*a.a + B^2*b.b + C^2*c.c + 2*A*B*a.b + 2*B*C*b.c + 2*C*A*c.a)
    # = 1 - (A^2*(1-(1-a.a)) + B^2*(1-(1-b.b)) + C^2*(1-(1-c.c)) + 2*A*B*(1-(1-a.b)) + 2*B*C*(1-(1-b.c)) + 2*C*A*(1-(1-c.a)))
    # = 1 - (A^2+B^2+C^2+2*A*B+2*B*C+2*C*A) + 2*(A*B*(1-a.b) + B*C*(1-b.c) + C*A*(1-c.a))
    # = 1 - (A+B+C)^2 + 2*(A*B*(1-a.b) + B*C*(1-b.c) + C*A*(1-c.a))
    # = 1 - 1 + 2*(A*B*(1-a.b) + B*C*(1-b.c) + C*A*(1-c.a))
    # = 2*(A*B*(1-a.b) + B*C*(1-b.c) + C*A*(1-c.a))
    #     But |a-b|^2 = (a-b).(a-b) = a.a - 2*a.b + b.b = 2-2*a.b = 2*(1-a.b), so...
    # = A*B*|b-a|^2 + B*C*|c-b|^2 + C*A*|a-c|^2
    poincareCenter = kleinCenter / (1+sqrt(aCoeff*bCoeff*length2(b-a) + bCoeff*cCoeff*length2(c-b) + cCoeff*aCoeff*length2(a-c)))
    p = poincareCenter
    do('xform(a,-p)+xform(b,-p)+xform(c,-p)')

    # GRRR that was worse!! why?? did I mess it up?
    # okay, the following is a little better... still not what I was hoping though. maybe what I was hoping was not realistic.
    # oh hell, it's not better with thin isosceles... bleah! what went wrong??
    # maybe its main advantage is when summing lots of points, not sure

    scaleFactor = (1+sqrt(aCoeff*bCoeff*length2(b-a) + bCoeff*cCoeff*length2(c-b) + cCoeff*aCoeff*length2(a-c)))
    aCoeff /= scaleFactor
    bCoeff /= scaleFactor
    cCoeff /= scaleFactor
    poincareCenter = aCoeff*a + bCoeff*b + cCoeff*c
    p = poincareCenter
    do('xform(a,-p)+xform(b,-p)+xform(c,-p)')


    print "    out idealTriangleCenterSmart"
    return poincareCenter

def gamma(v):
    return 1/invGamma(v)
def invGamma(v):
    return sqrt(1-length2(v))


# from the paper "hyperbolic barycentric coordinates" by a.a.ungar.
# a,b,c must be finite (i.e. not on the disk boundary),
# but I'm hoping for a formulation that doesn't require that.
def kleinOrthoCenter(a,b,c):
    if type(a) != complex:
        return c2v(kleinOrthoCenter(v2c(a),v2c(b),v2c(c)))
    print "    in kleinOrthoCenter"
    do('a')
    do('b')
    do('c')

    fab = invGammaKleinSegment(a,b)
    fbc = invGammaKleinSegment(b,c)
    fca = invGammaKleinSegment(c,a)

    Cab = fab-fbc*fca
    Cbc = fbc-fca*fab
    Cca = fca-fab*fbc

    CcaCab = (fca-fab*fbc)*(fab-fbc*fca)
    CcaCab = fca*fab - fab**2*fbc - fbc*fca**2 + fab*fbc**2*fca


    coeffa = CcaCab*(invGamma(b)*invGamma(c))
    coeffb = Cab*Cbc*(invGamma(c)*invGamma(a))
    coeffc = Cbc*Cca*(invGamma(a)*invGamma(b))


    denom = coeffa+coeffb+coeffc
    do('coeffa')
    do('coeffb')
    do('coeffc')
    do('denom')
    do('coeffa/denom')
    do('coeffb/denom')
    do('coeffc/denom')

    if False:
        # Messing around with intermediate formulas from the paper...
        Pc = ((gab*gbc-gca)*gamma(a)*a + (gca*gab-gbc)*gamma(b)*b) / ((gab*gbc-gca)*gamma(a) + (gca*gab-gbc)*gamma(b))
        Pa = ((gbc*gca-gab)*gamma(b)*b + (gab*gbc-gca)*gamma(c)*c) / ((gbc*gca-gab)*gamma(b) + (gab*gbc-gca)*gamma(c))
        Pb = ((gca*gab-gbc)*gamma(c)*c + (gbc*gca-gab)*gamma(a)*a) / ((gca*gab-gbc)*gamma(c) + (gca*gab-gab)*gamma(a))
        do('Pa')
        do('Pb')
        do('Pc')
        # In the case b is a right angle,
        # equation 12.7 should hold,
        # i.e. gab*gbc == gca. Does it?
        do('gab*gbc')
        do('gca')



    print "    out kleinOrthoCenter"
    return coeffa/denom * a + coeffb/denom * b + coeffc/denom * c

def idealTriangleCenterDumb(a,b,c):
    if type(a) != complex:
        return c2v(idealTriangleCenterDumb(v2c(a),v2c(b),v2c(c)))

    print "    in idealTriangleCenterDumb"

    # fix non-units, so caller can be sloppy
    a /= abs(a)
    b /= abs(b)
    c /= abs(c)

    # absolute value of curvatures...
    ka = abs(b+c)/abs(b-c)
    kb = abs(c+a)/abs(c-a)
    kc = abs(a+b)/abs(a-b)
    # correct sign # XXX there's gotta be a more elegant way
    if (b.conjugate()*c).imag < 0: ka *= -1
    if (c.conjugate()*a).imag < 0: kb *= -1
    if (a.conjugate()*b).imag < 0: kc *= -1
    do('a')
    do('b')
    do('c')
    do('ka')
    do('kb')
    do('kc')
    # Descartes' circle theorem
    k = ka+kb+kc + 2*sqrt(ka*kb + kb*kc + kc*ka)
    do('k')
    do('1./k')

    # Centers-times-curvatures
    kza = 2j*(b-c)/length2(b-c)
    kzb = 2j*(c-a)/length2(c-a)
    kzc = 2j*(a-b)/length2(a-b)
    do('kza')
    do('kzb')
    do('kzc')

    kz = kza+kzb+kzc + 2*cmath.sqrt(kza*kzb + kzb*kzc + kzc*kza)
    z = kz / k
    do('z')

    # z is the euclidean center of the circle...
    # figure out the hyperbolic center.
    # solve for p (in same dir as z):
    #   (xform(2-sqrt(3),|p|) + xform(-(2-sqrt(3)),|p|))/2 == |z|
    # mathematica says:
    p = (7+4*sqrt(3)) / (3+2*sqrt(3) + (2+sqrt(3)) * sqrt(abs(z)**2 + 3))  *  z

    # The in-circle
    # has (euclidean) center z and radius k.
    # Find the poincare center of it.
    # solve for p (in same dir as z):
    #   (xform(2-sqrt(3),|p|) + xform(-(2-sqrt(3)),|p|))/2 == |z|
    # mathematica says:
    poincare_center = (7+4*sqrt(3)) / (3 + 2*sqrt(3) + (2+sqrt(3))*sqrt(length2(z)+3))  * z
    do('poincare_center')
    klein_center = htwice(poincare_center)
    do('klein_center')

    # Sanity check:
    # if we transform a,b,c by minus the poincare center,
    # the results should average to 0
    p = poincare_center
    do('xform(a,-p)+xform(b,-p)+xform(c,-p)')
    do('poincare_center')

    print "    out idealTriangleCenterDumb"
    return poincare_center


def idealTriangleCenter(a,b,c):
    print "in idealTriangleCenter"

    dumb = idealTriangleCenterDumb(a,b,c)
    do('dumb')

    smart = idealTriangleCenterSmart(a,b,c)
    do('smart')

    general = idealSimplexCenter([a,b,c])
    do('general')

    print "out idealTriangleCenter"
    return smart



def do(s):
    callerLocals = inspect.currentframe(1).f_locals
    answer = eval(s, globals(), callerLocals)
    print '        '+s+' = '+`answer`

# Little test program
if __name__ == '__main__':


    do('0')
    do('True')
    do('Vec([.3,.4])')
    do('xform([0,0],[0,0])')

    do('xform([0,0],[.5,0])')
    do('xform([0,0],[-.5,0])')
    do('xform([0,0],[0,.5])')
    do('xform([0,0],[0,-.5])')

    do('xform([0,0,0],[.3,.4,.5])')
    do('xform([0,0,0],[-.3,.4,.5])')
    do('xform([0,0,0],[-.3,-.4,.5])')
    do('xform([0,0,0],[.3,-.4,-.5])')


    do('xform([.3,.4,-.5],[0,0,0])')
    do('xform([-.3,.4,-.5],[0,0,0])')
    do('xform([-.3,-.4,-.5],[0,0,0])')
    do('xform([.3,-.4,.5],[0,0,0])')

    do('xform([.001,.002,.003],[.004,.005,.006])')
    do('xform([.01,.02,.03],[.04,.05,.06])')
    do('xform([.1,.2,.3],[.4,.5,.6])')

    do('xform([.1,.2],[.4,.5])')
    do('reflect0([.4,.5],[.4,.5])')
    do('reflect0([.6,.8],[.4,.5])')
    do('reflect1([.6,.8],[.4,.5])')
    do('reflect2([.6,.8],[.4,.5])') # z should have norm 1

    do('xform([0,1],[-.17157287525,0])')
    do('xform([0,1],[-.2679491924311227  ,0])')

    do('xform([.17157287525,0],[.17157287525,0])')

    p = .123
    do('p')
    from math import *
    do('xform([-.5,sqrt(3)/2], [p,0])')

    do('(complex(-.5,sqrt(3)/2)+p)/(1+complex(-.5,sqrt(3)/2)*p)')
    do('(complex(p-.5,sqrt(3)/2))/(1+complex(-.5,sqrt(3)/2)*p)')
    do('(complex(p-.5,sqrt(3)/2))/(complex(1-.5*p,sqrt(3)/2*p))')
    do('(complex(p-.5,sqrt(3)/2)) * (complex(1-.5*p,-sqrt(3)/2*p)) /  ((complex(1-.5*p,sqrt(3)/2*p))*(complex(1-.5*p,-sqrt(3)/2*p)))')
    do('(complex(p-.5,sqrt(3)/2)) * (complex(1-.5*p,-sqrt(3)/2*p)) /  ((1-.5*p)**2 + 3/4.*p**2)')
    do('((p-.5)* (1-.5*p) + 3/4.*p) /  ((1-.5*p)**2 + 3/4.*p**2)')

    do('((p-.5)*(1-.5*p)+3./4*p)/((1-.5*p)**2+3./4*p**2)')

    del p


    print "-----------------"
    do('havg(-.5,.5)')
    do('havg(.5,.6)')
    do('e2pcenter(0., .5)')
    do('e2pcenter(.55, .05)')
    if False:
        do('idealTriangleCenter([-.5,-sqrt(3)/2],[1,0],[-.5,sqrt(3)/2])')
        do('idealTriangleCenter([-3/5.,-4/5.],[1,0],[-3/5.,4/5.])')
        do('idealTriangleCenter([0,-1],[1,0],[0,1])')

        #do('idealTriangleCenter([0,-1],[1,8],[0,1])')
        #do('idealTriangleCenter([0,-1],[1,0],[-.2,1])')

        #do('idealTriangleCenter([0,-1],[1,0],[0,1])')

    if False:
        #do('idealTriangleCenter([-.5,-sqrt(3)/2],[sqrt(.5),sqrt(.5)],[sqrt(.5),-sqrt(.5)])')
        do('idealTriangleCenter([-sqrt(.5),-sqrt(.5)],[1,0],[-sqrt(.5),sqrt(.5)])')
        do('htwice(.5)');
        do('htwice(1/3.)');
        do('hhalf(.5)');
        do('htwice(hhalf(.5))');
        do('hhalf(htwice(.5))');
        do('idealTriangleCenter([1,0], [-.34,-.47], [-.34,.47])') # XXX sign is messed up!?
        do('idealTriangleCenter([1,0], [-.34,.47], [-.34,-.47])')
        do('idealTriangleCenter([-.34,-.47], [1,0], [-.34,.47])')
    if False:
        do('idealTriangleCenter([.1,.2], [-.34,-.47], [-.36,.9])')


    if False:
        # from work...
        do('htwice(Vec([.5]))')
        do('hhalf(Vec([.5]))')
        do('htwice(hhalf(Vec([.5])))')
        do('hhalf(htwice(Vec([.5])))')
        do('idealTriangleCenter([0,-1],[1,0],[0,1])')
        do('idealTriangleCenter([-.5,-sqrt(3)/2],[1,0],[-.5,sqrt(3)/2])')
    if False:
        do('idealTriangleCenter([-.5,-sqrt(3)/2],[-.5+.0001,sqrt(3)/2],[-.5,sqrt(3)/2])')
        do('idealTriangleCenter([0,-1],[-.5+.001,sqrt(3)/2],[-.5,sqrt(3)/2])')
        do('idealTriangleCenter([-1,0],[-.5+.001,sqrt(3)/2],[-.5,sqrt(3)/2])')
        do('idealTriangleCenter([-.5-.001,sqrt(3)/2],[-.5+.001,sqrt(3)/2],[-.5,sqrt(3)/2])')
    if False:
        do('xformKlein([0,1],[1,0])')
        do('xformKleinCheat([.1,.2],[.4,.5])')
        do('xformKlein([.1,.2],[.4,.5])')
    if True:
        # Exercise kleinOrthoCenter
        do('kleinOrthoCenter([-5/16.,-10/16.],[15/16.,0],[-5/16.,10/16.])')
        do('kleinOrthoCenter([0,0],[.5,0],[0,.5])')
        do('kleinOrthoCenter([-.5,-.5],[0,0],[-.5,.5])')
        do('kleinOrthoCenter([.1,.2], [-.34,-.47], [-.36,.9])')

        s = .9999999
        do('kleinOrthoCenter([0,-s],[s,0],[0,s])')

    do('idealTriangleCenter([.1,.2], [-.36,.9], [-.34,-.47])')
    do('idealTriangleCenter([.1,.2], [-.36,.9], [-.34,-.47])')
    if False:
        do('invGammaKleinSegment([.1,.2],[.5,.7])')
        do('invGammaKleinSegment([.1,.2],[0,0])')
        do('invGammaKleinSegment([1,0],[0,0])')
        do('invGammaKleinSegment([1,0],[0,1])')
    if False:
        do('idealTriangleCenter([.1,.2], [-.36,.9], [-.34,-.47])')
        do('idealTriangleCenter([.1,.2], [-.36,.9], [-.34,-.47])')

    if False:
        do('idealTriangleCenter([1,1],[1,1+1e-3],[1,1+2e-3])')
        do('idealTriangleCenter([1,1],[1,1+1e-4],[1,1+2e-4])')
        do('idealTriangleCenter([1,1],[1,1+1e-5],[1,1+2e-5])')
        do('idealTriangleCenter([1,1],[1,1+1e-6],[1,1+2e-6])')
    do('idealTriangleCenter([.1,.2], [.36,.9], [.34,.47])')
    do('idealTriangleCenter([.1,.2,.3], [.36,.9,.754], [.34,.47,.2946])')

    #do('idealSimplexCenter([[.1,.2], [.36,.9], [.34,.47],[.29,.567]])')


    do('idealTriangleCenter([-cos(pi/3),-sin(pi/3)],[1,0],[-cos(pi/3),sin(pi/3)])')
    do('idealTriangleCenter([-cos(pi/6),-sin(pi/6)],[1,0],[-cos(pi/6),sin(pi/6)])')
    do('idealTriangleCenter([-cos(pi/10),-sin(pi/10)],[1,0],[-cos(pi/10),sin(pi/10)])')
    do('idealTriangleCenter([-cos(pi/100),-sin(pi/100)],[1,0],[-cos(pi/100),sin(pi/100)])') # ditto... exact with smart, actually

    do('idealTriangleCenter([-.123,-.89],[1,2],[-.492,.78])') # not good with dumb, very good with smart
    do('idealTriangleCenter([1,1],[1,1+1e-5],[1,1+2e-5])') # totally sucky with dumb, kinda sucky with smart, as expected... need even smarter, need to avoid expressing in klein space
    do('idealTriangleCenter([1,1],[1,1+1e-6],[1,1+2e-6])') # totally sucky with dumb, kinda sucky with smart, as expected... need even smarter, need to avoid expressing in klein space
    do('idealTriangleCenter([1,1],[1,1-1e-6],[1,1+2e-6])') # totally sucky with dumb, kinda sucky with smart, as expected... need even smarter, need to avoid expressing in klein space
