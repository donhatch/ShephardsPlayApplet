#!/usr/bin/python

import sys
from math import *
from Vec import Vec
from Vec import Mat

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

    @staticmethod
    def identity(nDims):
        return HyperbolicIsometry(Mat.identity(nDims), [0]*nDims)

    def __init__(self,R=None,t=None):
        # R and t can't both be None, or we wouldn't know the dimension
        if R == None: R = Mat.identity(len(t))
        if t == None: t = [0]*len(R)
        self.R = Mat(R)
        self.t = Vec(t)
    def apply(self,p):
        return translate(p * self.R, self.t)
    def applyInverse(self,p):
        return self.R * translate(p,-self.t) # R * p = p * R^-1 since R is orthogonal
    # Return f such that for all p,
    # f(p) = rhs(self(p))
    def compose(self,rhs):
        lhs = self
        nDims = len(self.t)
        t = rhs.apply(self.t) # = rhs(lhs(0))
        if True:
            R = Mat.identity(nDims)
            for i in xrange(nDims):
                R[i] = translate(rhs.apply(lhs.apply(R[i])), -t) # R[i] = Isometry(I,t)^-1(rhs(lhs(I[i])))
            R = Mat(R)
        else:
            # Argh, I thought this was right, but it's not?? wtf?
            R = self.R * rhs.R
        return HyperbolicIsometry(R,t)
    def inverse(self):
        return HyperbolicIsometry(None,-self.t).compose(HyperbolicIsometry(self.R.transposed(),None))
    def dist2(self,rhs):
        return (rhs.R-self.R).length2() + (rhs.t-self.t).length2()
    def __repr__(self):
        return 'HyperbolicIsometry('+`self.R`+','+`self.t`+')'
    def __str__(self):
        return self.__repr__()

    #
    # Operator notation:
    #    f(p) = f.apply(p)
    #     p*f = f.apply(p)
    #   f0*f1 = f0.compose(f1)
    # note that
    #     (f0*f1)*f2 == f0*(f1*f2)
    # and  (p*f0)*f1 == p*(f0*f1)
    #
    def __call__(self,p):
        return self.apply(p)
    def __rmul__(self,lhs):
        # actually relies on fact that Vec's __mul__
        # explicitly calls rhs.__rmul__ when rhs is unrecognized type
        return self.apply(lhs)
    def __mul__(self,rhs):
        return self.compose(rhs)
    def __pow__(self,rhs):
        assert type(rhs) == int
        if rhs == -1: # most common case
            return self.inverse()
        if rhs < 0:
            # either of the following work
            if True:
                return (self^-rhs).inverse()
            else:
                return self.inverse()^-rhs
        if rhs > 1:
            return self**int(rhs/2) * self**(rhs-int(rhs/2))
        if rhs == 1:
            return self
        assert rhs == 0
        return HyperbolicIsometry.identity(len(self.t))
    # XXX TODO: I think this is a bad idea, since ^ binds looser than * and even +
    def __xor__(self,rhs):
        return self.__pow__(rhs)


def do(s):
    import inspect
    answer = eval(s, globals(), inspect.currentframe(1).f_locals)
    print '        '+s+' = '+`answer`

# Little test program
if __name__ == '__main__':
    p = Vec([0,.5])

    ang = 20*pi/180.
    c = cos(ang)
    s = sin(ang)
    R0 = Mat([[c,s],[-s,c]])
    t0 = Vec([.5,.1])

    ang = 10*pi/180.
    c = cos(ang)
    s = sin(ang)
    R1= Mat([[c,s],[-s,c]])
    t1 = Vec([-.3,.2])

    F0 = HyperbolicIsometry(R0,t0)
    F1 = HyperbolicIsometry(R1,t1)
    do('               F0.compose(F1).apply(p)')
    do('                               p*F0*F1')
    do('                             (p*F0)*F1')
    do('                             p*(F0*F1)')
    do('                 F1.apply(F0.apply(p))')
    do('   translate(translate(p*R0,t0)*R1,t1)')
    do('translate(translate(p*R0*R1,t0*R1),t1)')
    do('translate(p*R0*R1,translate(t0*R1,t1))') # WRONG
    print
    do('    F1.inverse()(p)')
    do(' F1.applyInverse(p)')
    do('         (F1^-1)(p)')
    do('           p*F1**-1')
    do('          p*(F1^-1)')
    print
    do('            p*F1**1')
    do('           p*(F1^1)')
    print
    do('           p*F1*F1')
    do('  p*F1.compose(F1)')
    print
    do('           p*F1**2')
    do('          p*(F1^2)')
    print
    do('          p*F1**-2')
    do('   p*F1**-1*F1**-1')
    do(' p*(F1**-1*F1**-1)')
    print
    do('         p*F1**-1')
    do('         p*F1**-1')


