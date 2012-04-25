#!/usr/bin/env python

import math
from Vec import Vec

# z,p are two vectors, of same dimension, of length < 1.
# transform z by the isometry of the poincare disk
# that takes the origin to p.
# In the complex plane, this transformation is z -> (z + p) / (1 + z conj(p)).
def xform(z,p):
    z = Vec(z)
    p = Vec(p)
    p2 = p.length2()
    if p2 == 0:
        return z
    z2 = z.length2()


    # Let z = x u + y v
    # where u,v is the orthonormal basis
    #     u = p/||p||
    #     v = z orthonormalized with respect to u.
    zdotp = z.dot(p) # so x = z dot u = z dot p / ||p||^2 = zdotp / p2 (but we never compute x or u explicitly)
    yv = z - zdotp/p2*p # = z - (z dot u)*u,  so v = yv normalized (but we never need to compute v explicitly)
    y2 = yv.length2()    # = y^2 where y=(z dot v)

    # Worked out on paper...

    denominator = (1+zdotp)**2 + y2*p2
    pCoeff = (1 + 2*zdotp + z2)/denominator
    zCoeff = (1-p2)/denominator
    answer = pCoeff*p + zCoeff*z
    return answer


# Little test program
if __name__ == '__main__':

    def do(s):
        answer = eval(s)
        print s+' = '+`answer`

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


    do('xform([.3,.4,-.5],[0,0])')
    do('xform([-.3,.4,-.5],[0,0])')
    do('xform([-.3,-.4,-.5],[0,0])')
    do('xform([.3,-.4,.5],[0,0])')

    do('xform([.001,.002,.003],[.004,.005,.006])')
    do('xform([.01,.02,.03],[.04,.05,.06])')
    do('xform([.1,.2,.3],[.4,.5,.6])')
