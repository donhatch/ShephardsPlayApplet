#!/usr/bin/env python2

from math import *
from Vec import Vec

print "importing numpy"
import numpy
print "importing numpy.linalg"
import numpy.linalg
print "done."


# reciprocate about circle of radius 1 centered at the origin
def reciprocal(vs):
    dual_vs = []
    for i in xrange(len(vs)):
        v0 = vs[i-1]
        v1 = vs[1]
        e = v1-v0
        outwardNormal = Vec([e[1],-e[0]])
        dual_v = v0.dot(outwardNormal)/outwardNormal.dot(outwardNormal)
        dual.vs.append(dual_v)
    return dual_vs


# actually simplexCircumMomentAndParallelogramArea
def simplexCircumMomentAndArea(vs):
    assert len(vs) == 3
    a = Vec(vs[0])
    b = Vec(vs[1])
    c = Vec(vs[2])
    if False:
        twiceArea = (b-a).cross(c-a)
        area = .5 * twiceArea
        denominator = 2 * twiceArea**2
        alpha = (c-b).dot(c-b) * (b-a).dot(c-a) / denominator
        beta  = (a-c).dot(a-c) * (c-b).dot(a-b) / denominator
        gamma = (b-a).dot(b-a) * (a-c).dot(b-c) / denominator
        #do('alpha+beta+gamma')
        moment = (alpha*a + beta*b + gamma*c) * area
    elif False:
        area = (b-a).cross(c-a)
        denominator = 2 * area**2
        alpha = (c-b).dot(c-b) * (b-a).dot(c-a) / denominator
        beta  = (a-c).dot(a-c) * (c-b).dot(a-b) / denominator
        gamma = (b-a).dot(b-a) * (a-c).dot(b-c) / denominator
        #do('alpha+beta+gamma')
        moment = (alpha*a + beta*b + gamma*c) * area
    elif False:
        area = (b-a).cross(c-a)
        denominator = 2 * area**2
        alpha = (c-b).dot(c-b) * (b-a).dot(c-a) / (2*area)
        beta  = (a-c).dot(a-c) * (c-b).dot(a-b) / (2*area)
        gamma = (b-a).dot(b-a) * (a-c).dot(b-c) / (2*area)
        #do('alpha+beta+gamma')
        moment = alpha*a + beta*b + gamma*c
    elif False:
        area = (b-a).cross(c-a)
        assert len(a) == 2
        A = a-c
        B = b-c
        moment = (Vec(B[1],-B[0]) * A.dot(A)/2
                + Vec(-A[1],A[0]) * B.dot(B)/2) + c*area
        moment = (Vec(b[1]-c[1],c[0]-b[0]) * (a-c).dot(a-c)/2
                + Vec(c[1]-a[1],a[0]-c[0]) * (b-c).dot(b-c)/2) + c*area
        moment = c*area + (Vec(b[1]-c[1],c[0]-b[0]) * ((a[0]-c[0])**2+(a[1]-c[1])**2)/2
                         + Vec(c[1]-a[1],a[0]-c[0]) * ((b[0]-c[0])**2+(b[1]-c[1])**2)/2)

        ax,ay = a
        bx,by = b
        cx,cy = c
        momentX = c[0]*area + (b[1]-c[1])*((a[0]-c[0])**2+(a[1]-c[1])**2)/2 + (c[1]-a[1])*((b[0]-c[0])**2+(b[1]-c[1])**2)/2
        momentY = c[1]*area + (c[0]-b[0])*((a[0]-c[0])**2+(a[1]-c[1])**2)/2 + (a[0]-c[0])*((b[0]-c[0])**2+(b[1]-c[1])**2)/2
        moment = Vec(momentX,momentY)
    else:
        area = (b-a).cross(c-a)
        assert len(a) == 2
        ax,ay = a
        bx,by = b
        cx,cy = c
        momentX = cx*area + (by-cy)*(ax**2+cx**2+ay**2+cy**2-2*ax*cx-2*ay*cy)/2 + (cy-ay)*(bx**2+cx**2+by**2+cy**2-2*bx*cx-2*by*cy)/2
        momentY = cy*area + (cx-bx)*(ax**2+cx**2+ay**2+cy**2-2*ax*cx-2*ay*cy)/2 + (ax-cx)*(bx**2+cx**2+by**2+cy**2-2*bx*cx-2*by*cy)/2
        # ARGH!!! getting worse and worse!
        moment = Vec(momentX,momentY)

        # what's the contribution of cx to momentX? (omit contribution of bx and ax... assume they are zero or something)
        c_momentX = (area + ((by-cy)*cx - 2*ax*(by-cy))/2 + (cy-ay)*(cx-2*bx)/2) * cx
        a_momentX = (area + ((cy-ay)*ax - 2*bx*(cy-ay))/2 + (ay-by)*(ax-2*cx)/2) * ax
        b_momentX = (area + ((ay-by)*bx - 2*cx*(ay-by))/2 + (by-cy)*(bx-2*ax)/2) * bx
        #do('momentX')
        #do('a_momentX + b_momentX + c_momentX')
        # ARGH! it doesn't match (maybe typo or something)
    #do('area')
    return moment,area


def circumMomentAndArea(vs):
    #print "    in circumMomentAndArea"
    totalMoment,totalArea = simplexCircumMomentAndArea([vs[0],vs[1],vs[2]])
    for i in xrange(len(vs)-3):
        moment,area = simplexCircumMomentAndArea([vs[0],vs[2+i],vs[3+i]])
        totalMoment += moment
        totalArea += area
    #print "    out circumMomentAndArea"
    return totalMoment,totalArea

def circumCenter(vs):
    #print "    in circumCenter"
    moment,area = circumMomentAndArea(vs)
    answer = moment / area
    #print "    out circumCenter"
    return answer

# exercise circumCenter in every way
def circumCenterAll(vs):
    for i in xrange(len(vs)):
        answer = circumCenter(vs[i:]+vs[:i])
        do('answer')
        answer = circumCenter(list(reversed(vs[i:]+vs[:i])))
        do('answer')
    return answer


def inCenter3(vs):
    assert len(vs) == 3
    a = Vec(vs[0])
    b = Vec(vs[1])
    c = Vec(vs[2])
    alpha = (c-b).length()
    beta = (a-c).length()
    gamma = (b-a).length()
    denominator = alpha+beta+gamma
    answer = (a * (alpha/denominator)
            + b * (beta/denominator)
            + c * (gamma/denominator))
    return answer

def inCenter4(vs):
    vs = [Vec(v) for v in vs]
    n = len(vs)
    assert n == 4
    inwardNormals = [(vs[(i+1)%n]-vs[i]).perpDot().normalized() for i in xrange(n)]
    offsets = [inwardNormals[i].dot(vs[i]) for i in xrange(n)]
    M = numpy.matrix([
        list(inwardNormals[0])+[1,0],
        list(inwardNormals[1])+[0,1],
        list(inwardNormals[2])+[1,0],
        list(inwardNormals[3])+[0,1],
    ])
    xyrr = numpy.linalg.solve(M,offsets)
    x,y,r,R = xyrr
    return Vec(x,y)

def inCenter(vs):
    if False:
        do('vs')
    vs = [Vec(v) for v in vs]
    n = len(vs)
    inwardNormals = [(vs[(i+1)%n]-vs[i]).perpDot().normalized() for i in xrange(n)]
    offsets = [inwardNormals[i].dot(vs[i]) for i in xrange(n)]

    # compute n-2 tri-side in-centers...
    centers = []
    radii = []
    for i in xrange(n-2):
        M = numpy.matrix([
            list(inwardNormals[ 0 ])+[-1],
            list(inwardNormals[i+1])+[-1],
            list(inwardNormals[i+2])+[-1],
        ])
        o = numpy.matrix([
            [offsets[ 0 ]],
            [offsets[i+1]],
            [offsets[i+2]],
        ])
        x,y,r = numpy.linalg.solve(M,o)
        if False:
            # FUDGE
            r = abs(r)
        centers.append(Vec(x,y))
        radii.append(r)
        if True:
            #do('x')
            #do('y')
            do('r')


    if n == 3:
        # single weight will be zero in this case... no point in doing the undefined arithmetic
        return centers[0]

    if n == 4:
        # Either way works, but neither way is robust when cocircular
        weights = [
            1./(inwardNormals[3].dot(centers[0]) - offsets[3] - radii[0]),
            1./(inwardNormals[n-3].dot(centers[n-2-1])-offsets[n-3] - radii[n-2-1])
        ]
        weights = [
            inwardNormals[1].dot(centers[1])-offsets[1] - radii[1],
            inwardNormals[3].dot(centers[0])-offsets[3] - radii[0],
        ]

    if n == 5:
        # I fear this doesn't really work
        weights = [
            (inwardNormals[1].dot(centers[1])-offsets[1] - radii[1])*(inwardNormals[2].dot(centers[2])-offsets[2] - radii[2]),
            (inwardNormals[2].dot(centers[2])-offsets[2] - radii[2])*(inwardNormals[3].dot(centers[0])-offsets[3] - radii[0]),
            (inwardNormals[3].dot(centers[0])-offsets[3] - radii[0])*(inwardNormals[4].dot(centers[1])-offsets[4] - radii[1]),
        ]


    if False: # XXX GET RID
        # weights other than first and last are
        # 1 / (distance from last side not involving that circle) / (distance from first side not involving that circle)
        for i in xrange(n-4):
            weights.append(1./((inwardNormals[1+i].dot(centers[1+i]) - offsets[1+i] - radii[1+i])
                             * (inwardNormals[4+i].dot(centers[1+i]) - offsets[4+i] - radii[1+i]) ))

    weightsSum = sum(weights)
    if False:
        do('[float(weight) for weight in weights]')
        do('weightsSum')
        do('[float(weight/weightsSum) for weight in weights]')

    return sum([center*(weight/weightsSum) for center,weight in zip(centers,weights)])


# exercise inCenter in every way
def inCenterAll(vs):
    print
    funNames = []
    if len(vs) == 4:
        funNames += ["inCenter4"]
    funNames += ["inCenter"]

    vsPermuteds = []
    if True:
        for i in xrange(len(vs)):
            vsPermuteds.append(vs[i:]+vs[:i])
            vsPermuteds.append(list(reversed(vs[i:]+vs[:i])))
    else:
        # just examine two cases
        vsPermuteds.append(vs)
        vsPermuteds.append([vs[1],vs[0]]+list(reversed(vs[2:])))

    for funName in funNames:
        fun = eval(funName)
        for vsPermuted in vsPermuteds:
            answer = fun(vsPermuted)
            print "        "+funName+" returned "+`answer`
    return answer


# Little test program
if __name__ == '__main__':

    def do(s):
        import inspect
        answer = eval(s, globals(), inspect.currentframe().f_back.f_locals)
        print '        '+s+' = '+`answer`

    do('circumCenterAll([[1,0],[0,1],[-1,0]])')
    do('circumCenterAll([[2,0],[1,1],[0,0]])')
    do('circumCenterAll([[2,0],[1,1],[0,0],[1,-1]])')
    do('circumCenterAll([[1.,2.],[5.6,9.2],[0.,0.]])')
    do('circumCenterAll([[1.,2.],[5.6,9.2],[0.,0.],[3.8,9.1]])')
    do('circumCenterAll([[0,0],[1,0],[2,2],[0,1]])')
    do('inCenter3([[0,0],[20,0],[0,15]])')
    do('inCenterAll([[0,0],[20,0],[0,15]])')
    do('inCenterAll([[0,0],[1,0],[1,1],[0,1]])')
    do('inCenterAll([[0,0],[10,0],[10,7.5],[0,15]])')
    do('inCenterAll([[0,0],[1,0],[2,2],[0,1]])')
    do('inCenterAll([[0,0],[2,0],[2,1],[0,1]])')
    do('inCenterAll([[0,0],[1,0],[2,1],[2,2]])')
    do('inCenterAll([[0,0],[1.1,0],[2.34,1],[2,5]])')
    do('inCenterAll([[-15,-2.5],[15,-25],[15,25],[-15,2.5]])')
    do('inCenterAll([[-15,-2.5],[15,-25],[15,25],[-15,2.5],[-16,0]])')
    do('inCenterAll([[0,0],[40,0],[40,7.5],[20,22.5],[0,7.5]])')
    do('inCenterAll([[0,0],[50,0],[20,22.5],[0,7.5]])')
