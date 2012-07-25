#!/usr/bin/env python

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
            arg = [(float(x) if type(x)==int else x) for x in args[0]]
            list.__init__(self,arg)
        else: # e.g. Vec(x,y,z).  This is just a convenience for more than 1 arg; to initialize a vector of length 1 you still have to say Vec([x])
            list.__init__(self,[(float(x) if type(x)==int else x) for x in args])
    def __mul__(self,rhs):
        if isinstance(rhs,float):
            return Vec([x*rhs for x in self])
        else:
            # how to make this happen automatically??
            return rhs.__rmul__(self)
    def __rmul__(self,lhs):
        return Vec([lhs*x for x in self])
    def __div__(self,rhs):
        return Vec([x/rhs for x in self])
    def __add__(self, rhs):
        return Vec([a+b for a,b in zip(self,rhs)]) # throws if differing sizes
    # need to override +=, otherwise we get list's
    def __iadd__(self, rhs):
        self = self + rhs
        return self
    def __sub__(self, rhs):
        return Vec([a-b for a,b in zip(self,rhs)]) # throws if differing sizes
    # need to override -=, otherwise we get list's
    def __isub__(self, rhs):
        self = self - rhs
        return self
    def __neg__(self):
        return Vec([-x for x in self])
    def __radd__(self, lhs):
        # evil special case so sum() will work on a list of Vecs
        if lhs == 0:
            return self
        return Vec([a+b for a,b in zip(lhs,self)]) # throws if differing sizes
    def length2(self): # mostly for computing distance
        return sum(length2(row) for row in self)
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
        return sum([a*b for a,b in zip(self,rhs)]) # throws if differing sizes
    def length2(self):
        return self.dot(self)
    def length(self):
        return math.sqrt(self.length2())
    def normalized(self):
        length = self.length()
        assert length != 0.
        return self * 1./length

class Mat(list):

    @staticmethod
    def identity(nDims):
        return Mat([[(1 if i==j else 0) for j in xrange(nDims)]
                                        for i in xrange(nDims)])

    def __init__(self,*args):
        assert len(args) == 1
        arg = [Vec(x) for x in args[0]]
        list.__init__(self,arg)
    def __mul__(self,rhs):
        if isinstance(rhs,Mat):
            return Mat([row*rhs for row in self])
        elif isinstance(rhs,Vec):
            return Vec([row.dot(rhs) for row in self])
        elif isinstance(rhs,float):
            return Mat([row*rhs for row in self])
        else:
            assert False
    def __rmul__(self,lhs):
        if isinstance(lhs,Vec):
            return sum([x*row for x,row in zip(lhs,self)])

        elif isinstance(lhs, float):
            return Mat([lhs*row for row in self])
        else:
            assert False
    def __add__(self, rhs):
        return Mat([a+b for a,b in zip(self,rhs)]) # throws if differing sizes
    # need to override +=, otherwise we get list's
    def __iadd__(self, rhs):
        self = self + rhs
        return self
    def __sub__(self, rhs):
        return Mat([a-b for a,b in zip(self,rhs)]) # throws if differing sizes
    # need to override -=, otherwise we get list's
    def __isub__(self, rhs):
        self = self - rhs
        return self
    def __div__(self,rhs):
        if isinstance(rhs,float):
            return Mat([row / rhs for row in self])
        else:
            assert False
    def length2(self):
        return sum([row.length2() for row in self])
    def transposed(self):
        return Mat(zip(*self))
    # adjoint/adjugate
    def adj(self):
        if len(self) == 1:
            return Mat([[1.]])
        elif len(self) == 2:
            return Mat([
                [self[1][1],-self[0][1]],
                [-self[1][0],self[0][0]],
            ])
        elif len(self) == 3:
            # from vec.h
            return Mat([
                [ (self[1][1]* self[2][2] + self[1][2]*-self[2][1]),
                 -(self[1][0]* self[2][2] + self[1][2]*-self[2][0]),
                  (self[1][0]* self[2][1] + self[1][1]*-self[2][0])],
                [-(self[0][1]* self[2][2] + self[0][2]*-self[2][1]),
                  (self[0][0]* self[2][2] + self[0][2]*-self[2][0]),
                 -(self[0][0]* self[2][1] + self[0][1]*-self[2][0])],
                [ (self[0][1]* self[1][2] + self[0][2]*-self[1][1]),
                 -(self[0][0]* self[1][2] + self[0][2]*-self[1][0]),
                  (self[0][0]* self[1][1] + self[0][1]*-self[1][0])]
            ]).transposed()
        elif len(self) == 4:
            # from vec.h
            return Mat([
                [(((self)[1])[1]* (((self)[2])[2]* (((self)[3])[3]) + ((self)[2])[3]*-(((self)[3])[2])) + ((self)[1])[2]*-(((self)[2])[1]* (((self)[3])[3]) + ((self)[2])[3]*-(((self)[3])[1])) + ((self)[1])[3]* (((self)[2])[1]* (((self)[3])[2]) + ((self)[2])[2]*-(((self)[3])[1]))),
                -(((self)[1])[0]* (((self)[2])[2]* (((self)[3])[3]) + ((self)[2])[3]*-(((self)[3])[2])) + ((self)[1])[2]*-(((self)[2])[0]* (((self)[3])[3]) + ((self)[2])[3]*-(((self)[3])[0])) + ((self)[1])[3]* (((self)[2])[0]* (((self)[3])[2]) + ((self)[2])[2]*-(((self)[3])[0]))),
                (((self)[1])[0]* (((self)[2])[1]* (((self)[3])[3]) + ((self)[2])[3]*-(((self)[3])[1])) + ((self)[1])[1]*-(((self)[2])[0]* (((self)[3])[3]) + ((self)[2])[3]*-(((self)[3])[0])) + ((self)[1])[3]* (((self)[2])[0]* (((self)[3])[1]) + ((self)[2])[1]*-(((self)[3])[0]))),
                -(((self)[1])[0]* (((self)[2])[1]* (((self)[3])[2]) + ((self)[2])[2]*-(((self)[3])[1])) + ((self)[1])[1]*-(((self)[2])[0]* (((self)[3])[2]) + ((self)[2])[2]*-(((self)[3])[0])) + ((self)[1])[2]* (((self)[2])[0]* (((self)[3])[1]) + ((self)[2])[1]*-(((self)[3])[0])))],

                [-(((self)[0])[1]* (((self)[2])[2]* (((self)[3])[3]) + ((self)[2])[3]*-(((self)[3])[2])) + ((self)[0])[2]*-(((self)[2])[1]* (((self)[3])[3]) + ((self)[2])[3]*-(((self)[3])[1])) + ((self)[0])[3]* (((self)[2])[1]* (((self)[3])[2]) + ((self)[2])[2]*-(((self)[3])[1]))),
                (((self)[0])[0]* (((self)[2])[2]* (((self)[3])[3]) + ((self)[2])[3]*-(((self)[3])[2])) + ((self)[0])[2]*-(((self)[2])[0]* (((self)[3])[3]) + ((self)[2])[3]*-(((self)[3])[0])) + ((self)[0])[3]* (((self)[2])[0]* (((self)[3])[2]) + ((self)[2])[2]*-(((self)[3])[0]))),
                -(((self)[0])[0]* (((self)[2])[1]* (((self)[3])[3]) + ((self)[2])[3]*-(((self)[3])[1])) + ((self)[0])[1]*-(((self)[2])[0]* (((self)[3])[3]) + ((self)[2])[3]*-(((self)[3])[0])) + ((self)[0])[3]* (((self)[2])[0]* (((self)[3])[1]) + ((self)[2])[1]*-(((self)[3])[0]))),
                (((self)[0])[0]* (((self)[2])[1]* (((self)[3])[2]) + ((self)[2])[2]*-(((self)[3])[1])) + ((self)[0])[1]*-(((self)[2])[0]* (((self)[3])[2]) + ((self)[2])[2]*-(((self)[3])[0])) + ((self)[0])[2]* (((self)[2])[0]* (((self)[3])[1]) + ((self)[2])[1]*-(((self)[3])[0])))],

                [(((self)[0])[1]* (((self)[1])[2]* (((self)[3])[3]) + ((self)[1])[3]*-(((self)[3])[2])) + ((self)[0])[2]*-(((self)[1])[1]* (((self)[3])[3]) + ((self)[1])[3]*-(((self)[3])[1])) + ((self)[0])[3]* (((self)[1])[1]* (((self)[3])[2]) + ((self)[1])[2]*-(((self)[3])[1]))),
                -(((self)[0])[0]* (((self)[1])[2]* (((self)[3])[3]) + ((self)[1])[3]*-(((self)[3])[2])) + ((self)[0])[2]*-(((self)[1])[0]* (((self)[3])[3]) + ((self)[1])[3]*-(((self)[3])[0])) + ((self)[0])[3]* (((self)[1])[0]* (((self)[3])[2]) + ((self)[1])[2]*-(((self)[3])[0]))),
                (((self)[0])[0]* (((self)[1])[1]* (((self)[3])[3]) + ((self)[1])[3]*-(((self)[3])[1])) + ((self)[0])[1]*-(((self)[1])[0]* (((self)[3])[3]) + ((self)[1])[3]*-(((self)[3])[0])) + ((self)[0])[3]* (((self)[1])[0]* (((self)[3])[1]) + ((self)[1])[1]*-(((self)[3])[0]))),
                -(((self)[0])[0]* (((self)[1])[1]* (((self)[3])[2]) + ((self)[1])[2]*-(((self)[3])[1])) + ((self)[0])[1]*-(((self)[1])[0]* (((self)[3])[2]) + ((self)[1])[2]*-(((self)[3])[0])) + ((self)[0])[2]* (((self)[1])[0]* (((self)[3])[1]) + ((self)[1])[1]*-(((self)[3])[0])))],

                [-(((self)[0])[1]* (((self)[1])[2]* (((self)[2])[3]) + ((self)[1])[3]*-(((self)[2])[2])) + ((self)[0])[2]*-(((self)[1])[1]* (((self)[2])[3]) + ((self)[1])[3]*-(((self)[2])[1])) + ((self)[0])[3]* (((self)[1])[1]* (((self)[2])[2]) + ((self)[1])[2]*-(((self)[2])[1]))),
                (((self)[0])[0]* (((self)[1])[2]* (((self)[2])[3]) + ((self)[1])[3]*-(((self)[2])[2])) + ((self)[0])[2]*-(((self)[1])[0]* (((self)[2])[3]) + ((self)[1])[3]*-(((self)[2])[0])) + ((self)[0])[3]* (((self)[1])[0]* (((self)[2])[2]) + ((self)[1])[2]*-(((self)[2])[0]))),
                -(((self)[0])[0]* (((self)[1])[1]* (((self)[2])[3]) + ((self)[1])[3]*-(((self)[2])[1])) + ((self)[0])[1]*-(((self)[1])[0]* (((self)[2])[3]) + ((self)[1])[3]*-(((self)[2])[0])) + ((self)[0])[3]* (((self)[1])[0]* (((self)[2])[1]) + ((self)[1])[1]*-(((self)[2])[0]))),
                (((self)[0])[0]* (((self)[1])[1]* (((self)[2])[2]) + ((self)[1])[2]*-(((self)[2])[1])) + ((self)[0])[1]*-(((self)[1])[0]* (((self)[2])[2]) + ((self)[1])[2]*-(((self)[2])[0])) + ((self)[0])[2]* (((self)[1])[0]* (((self)[2])[1]) + ((self)[1])[1]*-(((self)[2])[0])))]
            ]).transposed()

        else:
            assert False
    # determinant
    def det(self):
        if len(self) == 1:
            return self[0][0]
        elif len(self) == 2:
            return self[0][0]*self[1][1] - self[0][1]*self[1][0];
        elif len(self) == 3:
            # from vec.h
            return (self[0][0]* (self[1][1]*self[2][2] + self[1][2]*-self[2][1])
                  + self[0][1]*-(self[1][0]*self[2][2] + self[1][2]*-self[2][0])
                  + self[0][2]* (self[1][0]*self[2][1] + self[1][1]*-self[2][0]))
        elif len(self) == 4:
            return (((((self)[0])[0]* (((self)[1])[1]* (((self)[2])[2]* (((self)[3])[3]) + ((self)[2])[3]*-(((self)[3])[2])) + ((self)[1])[2]*-(((self)[2])[1]* (((self)[3])[3]) + ((self)[2])[3]*-(((self)[3])[1])) + ((self)[1])[3]* (((self)[2])[1]* (((self)[3])[2]) + ((self)[2])[2]*-(((self)[3])[1]))) + ((self)[0])[1]*-(((self)[1])[0]* (((self)[2])[2]* (((self)[3])[3]) + ((self)[2])[3]*-(((self)[3])[2])) + ((self)[1])[2]*-(((self)[2])[0]* (((self)[3])[3]) + ((self)[2])[3]*-(((self)[3])[0])) + ((self)[1])[3]* (((self)[2])[0]* (((self)[3])[2]) + ((self)[2])[2]*-(((self)[3])[0]))) + ((self)[0])[2]* (((self)[1])[0]* (((self)[2])[1]* (((self)[3])[3]) + ((self)[2])[3]*-(((self)[3])[1])) + ((self)[1])[1]*-(((self)[2])[0]* (((self)[3])[3]) + ((self)[2])[3]*-(((self)[3])[0])) + ((self)[1])[3]* (((self)[2])[0]* (((self)[3])[1]) + ((self)[2])[1]*-(((self)[3])[0]))) + ((self)[0])[3]*-(((self)[1])[0]* (((self)[2])[1]* (((self)[3])[2]) + ((self)[2])[2]*-(((self)[3])[1])) + ((self)[1])[1]*-(((self)[2])[0]* (((self)[3])[2]) + ((self)[2])[2]*-(((self)[3])[0])) + ((self)[1])[2]* (((self)[2])[0]* (((self)[3])[1]) + ((self)[2])[1]*-(((self)[3])[0]))))))
        else:
            assert False
    # inverse
    def inverse(self):
        return self.adj() / self.det()


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

