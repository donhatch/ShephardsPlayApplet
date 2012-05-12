#!/usr/bin/env python

from math import *
import cmath
import inspect
from Vec import Vec

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
    if isinstance(z,list): # list or Vec
        answer = xformVec(z,p)
        assert len(z) == len(p)
        if len(z) == 2:
            Answer = c2v(xformComplex(v2c(z),v2c(p)))
            #print "Answer = "+`Answer`
            #print "answer = "+`answer`
            assert abs(v2c(Answer)-v2c(answer)) < 1e-9
    elif type(z) in [int,float,complex]:
        answer = xformComplex(z,p)
        Answer = v2c(xformVec(c2v(z),c2v(p)))
        assert abs(answer-Answer) < 1e-9
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
        # convert to poincare disk and do calculation there
        A = hhalf(a)
        B = hhalf(b)
        AA = A.dot(A)
        AB = A.dot(B)
        BB = B.dot(B)
        
        # given p poincare,
        #     k = 2*p/(1+pp)
        # so kk = 4*pp/(1+pp)^2

        pp = length2(A-B)/(1-2*AB+AA*BB)
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
        # convert to poincare disk and do calculation there
        A = a / (1+sqrt(1-length2(a)))
        B = b / (1+sqrt(1-length2(b)))
        AA = aa / (1+sqrt(1-length2(a)))**2
        AB = ab / ((1+sqrt(1-length2(a)))*(1+sqrt(1-length2(b))))
        BB = bb / (1+sqrt(1-length2(b)))**2

        pp = (AA-2*AB+BB)/(1-2*AB+AA*BB)
        k = 2*sqrt(pp)/(1+pp)
        answer = sqrt((1-k)*(1+k))

        do('answer')


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


# This only works if a,b,c are unit vectors.
# Actually it doesn't work :-(
def centroidOfUnits(a,b,c):
    if type(a) == complex:
        return v2c(c2v(a),c2v(b),c2v(c))

    kleinCentroid = (a+b+c)/3.
    poincareCentroid = kleinCentroid / (1 + sqrt(1-kleinCentroid.length2()))
    return answer

# magnitude of cross product of two complex numbers,
# TODO: can this be expressed using complex arithmetic?
def cross(a,b):
    return a.real*b.imag - a.imag*b.real
def twiceTriArea(a,b,c):
    return cross(b-a,c-a)
def computeBarycentrics(p,a,b,c):
    denom = twiceTriArea(a,b,c)
    bary_a = twiceTriArea(p,b,c) / denom
    bary_b = twiceTriArea(a,p,c) / denom
    bary_c = twiceTriArea(a,b,p) / denom
    return bary_a,bary_b,bary_c

# this happens to work for iscosceles triangles, but not in general :-(
def idealTriangleCenterSimple(a,b,c):
    if type(a) == complex:
        return v2c(idealTriangleCenterSimple(c2v(a),c2v(b),c2v(c)))
    if type(a) != Vec:
        a = Vec(a)
        b = Vec(b)
        c = Vec(c)
    a = a.normalized()
    b = b.normalized()
    c = c.normalized()
    ab2 = (b-a).length2()
    bc2 = (c-b).length2()
    ca2 = (a-c).length2()
    kleinCenter = (a*bc2 + b*ca2 + c*ab2) / (ab2 + bc2 + ca2)
    poincareCenter = hhalf(kleinCenter)
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

def idealTriangleCenter(a,b,c):
    if type(a) != complex:
        return c2v(idealTriangleCenter(v2c(a),v2c(b),v2c(c)))

    print "    in idealTriangleCenter"

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

    s = .9999999
    do('s')
    do('kleinOrthoCenter(s*a,s*b,s*c)')
    kleinEstimate = kleinOrthoCenter(s*a,s*b,s*c)
    do('kleinEstimate')
    do('kleinEstimate.conjugate()*klein_center') # should have imaginary part 0

    barya,baryb,baryc = computeBarycentrics(klein_center,a,b,c)
    do('barya')
    do('baryb')
    do('baryc')

    # Sanity check:
    # if we transform a,b,c by minus the poincare center,
    # the results should average to 0
    p = poincare_center
    do('xform(a,-p)+xform(b,-p)+xform(c,-p)')

    print "    out idealTriangleCenter"
    return poincare_center

def twiceTriArea(a,b,c):
    return ((b-a).conjugate()*(c-a)).imag
def computeBarycentrics(p,a,b,c):
    sum = twiceTriArea(a,b,c)
    do('sum')
    return (twiceTriArea(p,b,c)/sum,
            twiceTriArea(a,p,c)/sum,
            twiceTriArea(a,b,p)/sum)


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

        #do('idealTriangleCenterSimple([0,-1],[1,0],[0,1])')

    if False:
        #do('idealTriangleCenter([-.5,-sqrt(3)/2],[sqrt(.5),sqrt(.5)],[sqrt(.5),-sqrt(.5)])')
        do('idealTriangleCenter([-sqrt(.5),-sqrt(.5)],[1,0],[-sqrt(.5),sqrt(.5)])')
        do('htwice(.5)');
        do('htwice(1/3.)');
        do('hhalf(.5)');
        do('htwice(hhalf(.5))');
        do('hhalf(htwice(.5))');
        do('idealTriangleCenter([1,0], [-.34,-.47], [-.34,.47])') # XXX sign is messed up!?
        do('idealTriangleCenterSimple([1,0], [-.34,.47], [-.34,-.47])')
        do('idealTriangleCenterSimple([-.34,-.47], [1,0], [-.34,.47])')
        #do('idealTriangleCenter([.1,.2], [-.34,-.47], [-.36,.9])')
        #do('idealTriangleCenterSimple([.1,.2], [-.34,-.47], [-.36,.9])')
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
        do('idealTriangleCenter([-cos(pi/3),-sin(pi/3)],[1,0],[-cos(pi/3),sin(pi/3)])')
        do('idealTriangleCenter([-cos(pi/6),-sin(pi/6)],[1,0],[-cos(pi/6),sin(pi/6)])')
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

