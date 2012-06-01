#!/usr/bin/env python2

#
# Simple vector class, implemented as a list with overridden:
#       constructor([...])
#       constructor(x,y,...)
#       *
#       *=
#       /
#       /=
#       +
#       +=
#       -
#       -=
# and methods:
#       cross
#       dot
#       length2
#       length
#       normalized
#

import math

class Vec(list):
    def __init__(self,*args):
        if len(args) == 1: # e.g. Vec([x,y,z])
            arg = [float(x) for x in args[0]]
            list.__init__(self,arg)
        else: # e.g. Vec(x,y,z).  This is just a convenience for more than 1 arg; to initialize a vector of length 1 you still have to say Vec([x])
            list.__init__(self,[float(x) for x in args])
    def __mul__(self,rhs):
        return Vec([x*rhs for x in self])
    def __rmul__(self,lhs):
        return Vec([lhs*x for x in self])
    def __div__(self,rhs):
        return Vec([x/rhs for x in self])
    def __add__(self, rhs):
        assert len(self) == len(rhs)
        return Vec([a+b for a,b in zip(self,rhs)]) # throws if differing sizes
    # need to override +=, otherwise we get list's
    def __iadd__(self, rhs):
        self = self + rhs
        return self
    def __sub__(self, rhs):
        assert len(self) == len(rhs)
        return Vec([a-b for a,b in zip(self,rhs)]) # throws if differing sizes
    def __neg__(self):
        return Vec([-x for x in self])
    def __radd__(self, lhs):
        # evil special case so sum() will work on a list of Vecs
        if lhs == 0:
            return self
        assert len(self) == len(lhs)
        return Vec([a+b for a,b in zip(lhs,self)]) # throws if differing sizes
    def perpDot(self):
        assert len(self) == 2
        return Vec(-self[1],self[0])
    def cross(self,rhs):
        if len(self) == 3 and len(rhs) == 3:
            return Vec([self[1]*rhs[2]-self[2]*rhs[1],
                        self[2]*rhs[0]-self[0]*rhs[2],
                        self[0]*rhs[1]-self[1]*rhs[0]])
        elif len(self) == 2 and len(rhs) == 2:
            return self[0]*rhs[1] - self[1]*rhs[0]
        else:
            assert False
    def dot(self,rhs):
        assert len(self) == len(rhs)
        return sum([a*b for a,b in zip(self,rhs)]) # throws if differing sizes
    def length2(self):
        return self.dot(self)
    def length(self):
        return math.sqrt(self.length2())
    def normalized(self):
        length = self.length()
        assert length != 0.
        return self * 1./length


# Little test program
if __name__ == '__main__':

    def do(s):
        answer = eval(s)
        print s+' = '+`answer`

    do('0')
    do('1')
    do('None')
    do('True')
    do('False')
    do('2+3')
    do('Vec(10,20)')
    do('Vec([10,20])')
    do('Vec([])')
    do('Vec([1])')
    do('Vec(100,200,300)')
    do('Vec(100,200,300) * 2')
    do('2 * Vec(100,200,300)')
    do('Vec([10,20,30])')
    do('Vec([10,20,30])')
    do('"z" * Vec(1,2,3)')
    do('Vec(1,2,3) * "z"')
    do('Vec([10,20,30]) / 10')
    do('Vec([10,20,30]) / 100')
    do('Vec([10,20,30]) / 100.')

    v = Vec(1,2,3)
    v *= 2
    do('v')
    v[1] = 1000
    do('v')
    v[1] *= 2
    do('v')

    do('Vec(1,2,3) + Vec(4,5,6)')
    do('Vec(1,2,3) + [4,5,6]')
    do('Vec(1,2,3) - [4,5,6]')
    do('Vec(1,2).cross([3,4])')
    do('Vec(1,2,3).cross([10,100,1000])')
    do('Vec(1,2,3).dot([10,20,30])')
    do('Vec(1,2,3).cross([4,5,8]).dot([1,1,1])')
    do('Vec(1,2).cross([4,5])')
    do('-Vec(1,2)')
    do('sum([Vec(1,2),[3,4]])')

