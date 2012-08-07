#!/usr/bin/env python

#
# The main thing of interest in here is "pseudocenter"
# which is the point that centers the euclidean centroid
# of a set of points on the boundary of the poincare disk.
#

from math import *
from Vec import Vec
from Vec import Mat

use_numpy = True
if use_numpy:
    print "importing numpy"
    import numpy
    print "importing numpy.linalg"
    import numpy.linalg
    print "done."


# reciprocate about circle of radius 1 centered at the origin
def reciprocal(vs):
    dual_vs = []
    for i in xrange(len(vs)):
        v0 = vs[-i-1] # more CW
        v1 = vs[-i]   # more CCW
        e = v1-v0
        outwardNormal = Vec([e[1],-e[0]])
        #dual_v = v0.dot(outwardNormal)/outwardNormal.dot(outwardNormal) * outwardNormal
        dual_v = outwardNormal/outwardNormal.dot(v0)
        dual_vs.append(dual_v)
    return dual_vs


# actually simplexCircumMomentAndParallelogramArea
def simplexCircumMomentAndTwiceArea(vs):
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


def circumMomentAndTwiceArea(vs):
    #print "    in circumMomentAndTwiceArea"
    #print "        vs = "+`vs`
    totalMoment,totalArea = simplexCircumMomentAndTwiceArea([vs[0],vs[1],vs[2]])
    for i in xrange(len(vs)-3):
        moment,area = simplexCircumMomentAndTwiceArea([vs[0],vs[2+i],vs[3+i]])
        totalMoment += moment
        totalArea += area
    #print "    out circumMomentAndTwiceArea"
    return totalMoment,totalArea

def circumCenter(vs):
    #print "    in circumCenter"
    moment,area = circumMomentAndTwiceArea(vs)
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


# possible radii:
#   - equalize primal and dual perimeters
#   - equalize primal and dual (square root of) areas
#   - equalize primal and dual sum of vert distances from center
#   - equalize primal and dual sum of squares of vert distances from center
#   - equalize primal and dual max vert distance from center
#   - equalize avg distance from perimeter to center
#   - equalize avg distance from area to center
#   - equalize geom mean of distances from verts to center
#   - equalize geom mean of distance from perimter to center
#   - equalize geom mean of distance from area to center
def reciprocate(vs,c):
    n = len(vs)
    unitNormals = [(vs[-1-i]-vs[-i]).perpDot().normalized() for i in xrange(n)]
    distances = [normal.dot(v-c) for normal,v in zip(unitNormals,reversed(vs))]
    dualVerts = [c+normal/distance for normal,distance in zip(unitNormals,distances)]
    # we've now reciprocated with respect to the unit sphere centered at c.

    # equalize something.
    # say, mean squared dist from verts to center.

    if True:
        # equalize avg squared distance from reciprocation center
        primalMeasure = sqrt(sum([(v-c).length2() for v in vs])/n)
        dualMeasure = sqrt(sum([(v-c).length2() for v in dualVerts])/n)
    else:
        # equalize avg distance from reciprocation center
        primalMeasure = sum([(v-c).length() for v in vs])/n
        dualMeasure   = sum([(v-c).length() for v in dualVerts])/n

    #do('primalMeasure')
    #do('dualMeasure')

    adjustment = primalMeasure / dualMeasure
    dualVerts = [c+adjustment*(v-c) for v in dualVerts]

    return dualVerts

# grr, it diverges
def circumCenterSequence(vs):
    print "    in circumCenterSequence"
    vs = [Vec(v) for v in vs]
    do('vs')


    # first make sure reciprocate twice gives the original
    c = circumCenter(vs)
    dual = reciprocate(vs,c)
    Vs = reciprocate(dual,c) # should be vs
    do('Vs')

    scratch = vs
    for i in xrange(50):
        c = circumCenter(scratch)
        do('c')
        scratch = reciprocate(scratch, c)
    print "    out circumCenterSequence"
    return None

# Find x such that f(x) == y
# by newton's method,
# with derivative computed via finite diffences with given eps
def newtonSolve(f,yTarget,xInitialGuess,nIterations,eps):
    dim = len(yTarget)
    assert dim == len(xInitialGuess)
    x = xInitialGuess
    for i in xrange(nIterations):
        #do('i')
        do('x')
        y = f(x)
        #do('y')
        error = y - yTarget
        jacobian = []
        for iDim in xrange(dim):
            xx = Vec(x) # copy
            xx[iDim] += eps
            jacobian.append((f(xx)-y)/eps)
        invJacobian = Mat(jacobian).inverse()
        #do('error')
        #do('jacobian')
        #do('invJacobian')
        #do('x')
        #do('error * invJacobian')
        x1 = x - error * invJacobian
        # hackier and hackier
        if True and 2*i < nIterations:
            x += .25 * (x1-x)
        else:
            x = x1
    return x


# Try to compute an in-center
# as the point such that reciprocating
# around that point gives something whose
# circumcenter is that point.
# XXX I think this turns out to be a bogus concept
def inCenterSolve(primal):
    print "        in inCenterSolve"
    primal = [Vec(v) for v in primal]

    n = len(primal)
    inCenterInitialGuess = sum(primal)/n # centroid of verts

    # now try it via newton solve.
    # this seems to be MUCH better.
    def f(inCenterGuess):
        dual = reciprocal([v-inCenterGuess for v in primal])
        if True:
            # This works but is doing too much math.
            # OH WAIT, sometimes it makes the solve converge to something small
            # where the other didn't? bleah but it's not to something inside
            dualCircumCenter = circumCenter(dual)
            return dualCircumCenter
        else:
            dualCircumMoment,dualTwiceArea = circumMomentAndTwiceArea(dual)
            return dualCircumMoment

    nIterations = 20
    eps = 1e-6
    answer = newtonSolve(f,Vec(0,0),inCenterInitialGuess,nIterations,eps)


    print "        out inCenterSolve"
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

# Try to figure out in-center
# of special pentagon with closest-point-on-sides:
#       0,1
#       nx,ny    nx^2+ny^2==1, nx < 0
#       nx,-ny
#       0,-1
#       x,0
def inCenter5special(nx,ny,x):
    print "    in inCenter5special"


    # from calculations on paper (2012/6/8),
    # given center = (c,0),
    # right-hand dual moment of primal-c is:
    #   (1/(x-c)^2 - 1) / 4
    # and left-hand dual moment is:
    #   a*(b+1)*((1-b)*(1+b)/(2*a) - a/2)
    # where:
    #   a = -nx/(-nx*c+1)
    #   b = ny/(-nx*c+1)
    # and we want c such that the sum of the two moments is zero.
    # For starters, let's verify the moments.

    assert abs(nx**2 + ny**2 - 1) < 1e-6

    # First way that worked...
    def computeDualCircumMoment(nx,ny,x,c):

        # This seems to be right...
        rightTwiceArea = 2./(x-c)
        rightMoment = 1./(x-c)**2 - 1
        rightCircumCenter = rightMoment / rightTwiceArea

        a = -nx/(-nx*c+1) # all positive, since nx<0
        b = ny/(-nx*c+1)


        leftTwiceArea = 2*a*(b+1)
        leftMoment = 2*a*(b+1)*((1-b)*(1+b)/(2.*a) - a/2.)
        leftCircumCenter = leftMoment / leftTwiceArea

        totalMoment = leftMoment + rightMoment
        totalTwiceArea = leftTwiceArea+rightTwiceArea
        circumCenter = totalMoment / totalTwiceArea

        if False:
            do('leftMoment')
            do('leftTwiceArea')
            do('leftCircumCenter')
            print
            do('rightMoment')
            do('rightTwiceArea')
            do('rightCircumCenter')

        return totalMoment # scalar, for now

    # now simplify...
    def computeDualCircumMoment(nx,ny,x,c):

        # This seems to be right...
        rightMoment = 1./(x-c)**2 - 1

        leftMoment = (ny-nx*c+1)*((-ny-nx*c+1)*(ny-nx*c+1) - nx*nx) / (-nx*c+1)**3
        totalMoment = rightMoment + leftMoment

        return totalMoment # scalar, for now

    def computeDualCircumMomentAlt(nx,ny,x,c):
        dualVerts0 = [
            [0,1],
            [nx,ny],    # remember nx is negative
            [nx,-ny],   # remember nx is negative
            [0,-1],
            [1./x,0],
        ]
        dualVerts0 = [Vec(v) for v in dualVerts0]
        #do('dualVerts0')
        verts = reciprocal(dualVerts0)
        do('verts')
        dualVerts = reciprocal([v-Vec(c,0) for v in verts])
        #do('dualVerts')
        circumC = circumCenter(dualVerts)

        leftCircumCenter = circumCenter(dualVerts[:4])
        rightCircumCenter = circumCenter([dualVerts[3],dualVerts[4],dualVerts[0]])

        leftCircumMoment,leftTwiceArea = circumMomentAndTwiceArea(dualVerts[:4])
        rightCircumMoment,rightTwiceArea = circumMomentAndTwiceArea([dualVerts[3],dualVerts[4],dualVerts[0]])

        if False:
            do('leftCircumMoment')
            do('leftTwiceArea')
            do('leftCircumCenter')
            print
            do('rightCircumMoment')
            do('rightTwiceArea')
            do('rightCircumCenter')

        return leftCircumMoment+rightCircumMoment


    do('computeDualCircumMoment   (nx,ny,x,-1.3)')
    do('computeDualCircumMoment   (nx,ny,x,-1.2)')
    do('computeDualCircumMoment   (nx,ny,x,-1.1)')
    do('computeDualCircumMoment   (nx,ny,x,-1.)')
    do('computeDualCircumMoment   (nx,ny,x,-.9)')
    do('computeDualCircumMoment   (nx,ny,x,-.8)')
    do('computeDualCircumMoment   (nx,ny,x,-.7)')
    do('computeDualCircumMoment   (nx,ny,x,-.6)')
    do('computeDualCircumMoment   (nx,ny,x,-.5)')
    do('computeDualCircumMoment   (nx,ny,x,-.4)')
    do('computeDualCircumMoment   (nx,ny,x,-.3)')
    do('computeDualCircumMoment   (nx,ny,x,-.2)')
    do('computeDualCircumMoment   (nx,ny,x,-.1)')
    do('computeDualCircumMoment   (nx,ny,x,0)')
    do('computeDualCircumMoment   (nx,ny,x,.1)')
    do('computeDualCircumMoment   (nx,ny,x,.2)')
    do('computeDualCircumMoment   (nx,ny,x,.3)')
    do('computeDualCircumMoment   (nx,ny,x,.4)')
    do('computeDualCircumMoment   (nx,ny,x,.5)')
    do('computeDualCircumMoment   (nx,ny,x,.6)')
    do('computeDualCircumMoment   (nx,ny,x,.7)')
    do('computeDualCircumMoment   (nx,ny,x,.8)')
    do('computeDualCircumMoment   (nx,ny,x,.9)')
    if x > 1.5:
        do('computeDualCircumMoment   (nx,ny,x,1.)')
        do('computeDualCircumMoment   (nx,ny,x,1.1)')
        do('computeDualCircumMoment   (nx,ny,x,1.2)')
        do('computeDualCircumMoment   (nx,ny,x,1.3)')
        do('computeDualCircumMoment   (nx,ny,x,1.4)')
        do('computeDualCircumMoment   (nx,ny,x,1.5)')
        do('computeDualCircumMoment   (nx,ny,x,1.6)')
        do('computeDualCircumMoment   (nx,ny,x,1.7)')
        do('computeDualCircumMoment   (nx,ny,x,1.8)')
        do('computeDualCircumMoment   (nx,ny,x,1.9)')
        do('computeDualCircumMoment   (nx,ny,x,2.)')
        do('computeDualCircumMoment   (nx,ny,x,3.)')
        do('computeDualCircumMoment   (nx,ny,x,4.)')
        do('computeDualCircumMoment   (nx,ny,x,5.)')
        do('computeDualCircumMoment   (nx,ny,x,6.)')
        do('computeDualCircumMoment   (nx,ny,x,7.)')
        do('computeDualCircumMoment   (nx,ny,x,8.)')
        do('computeDualCircumMoment   (nx,ny,x,9.)')
    if x > 10.5:
        do('computeDualCircumMoment   (nx,ny,x,10.)')
        do('computeDualCircumMoment   (nx,ny,x,11.)')
        do('computeDualCircumMoment   (nx,ny,x,12.)')
        do('computeDualCircumMoment   (nx,ny,x,13.)')
        do('computeDualCircumMoment   (nx,ny,x,14.)')
        do('computeDualCircumMoment   (nx,ny,x,15.)')
        do('computeDualCircumMoment   (nx,ny,x,16.)')
        do('computeDualCircumMoment   (nx,ny,x,17.)')
        do('computeDualCircumMoment   (nx,ny,x,18.)')
        do('computeDualCircumMoment   (nx,ny,x,19.)')
        do('computeDualCircumMoment   (nx,ny,x,20.)')
        do('computeDualCircumMoment   (nx,ny,x,21.)')
        do('computeDualCircumMoment   (nx,ny,x,22.)')
        do('computeDualCircumMoment   (nx,ny,x,23.)')
        do('computeDualCircumMoment   (nx,ny,x,24.)')
        do('computeDualCircumMoment   (nx,ny,x,25.)')
        do('computeDualCircumMoment   (nx,ny,x,26.)')
        do('computeDualCircumMoment   (nx,ny,x,27.)')
        do('computeDualCircumMoment   (nx,ny,x,28.)')
        do('computeDualCircumMoment   (nx,ny,x,29.)')
        do('computeDualCircumMoment   (nx,ny,x,30.)')
        do('computeDualCircumMoment   (nx,ny,x,35.)')
        do('computeDualCircumMoment   (nx,ny,x,40.)')
        do('computeDualCircumMoment   (nx,ny,x,45.)')
        do('computeDualCircumMoment   (nx,ny,x,50.)')
        do('computeDualCircumMoment   (nx,ny,x,55.)')
        do('computeDualCircumMoment   (nx,ny,x,60.)')
        do('computeDualCircumMoment   (nx,ny,x,65.)')
        do('computeDualCircumMoment   (nx,ny,x,70.)')
        do('computeDualCircumMoment   (nx,ny,x,75.)')
        do('computeDualCircumMoment   (nx,ny,x,80.)')
        do('computeDualCircumMoment   (nx,ny,x,85.)')
        do('computeDualCircumMoment   (nx,ny,x,90.)')
        #do('computeDualCircumMomentAlt(nx,ny,x,90.)')
        do('computeDualCircumMoment   (nx,ny,x,95.)')
        do('computeDualCircumMoment   (nx,ny,x,97.)')
        do('computeDualCircumMoment   (nx,ny,x,98.)')
        do('computeDualCircumMoment   (nx,ny,x,99.)')
        do('computeDualCircumMoment   (nx,ny,x,101.)')
        do('computeDualCircumMoment   (nx,ny,x,110.)')
        do('computeDualCircumMoment   (nx,ny,x,120.)')
        do('computeDualCircumMoment   (nx,ny,x,130.)')
        do('computeDualCircumMoment   (nx,ny,x,140.)')
        do('computeDualCircumMoment   (nx,ny,x,150.)')
        do('computeDualCircumMoment   (nx,ny,x,160.)')
        do('computeDualCircumMoment   (nx,ny,x,170.)')
        do('computeDualCircumMoment   (nx,ny,x,180.)')
        do('computeDualCircumMoment   (nx,ny,x,190.)')
        do('computeDualCircumMoment   (nx,ny,x,200.)')
        do('computeDualCircumMoment   (nx,ny,x,300.)')
        do('computeDualCircumMoment   (nx,ny,x,500.)')
        do('computeDualCircumMoment   (nx,ny,x,1000.)')
        do('computeDualCircumMoment   (nx,ny,x,10000.)')

# ARGH! for x=100, this goes up and down... probably NOT a unique answer :-(


    # find c such that computeDualCircumMoment(nx,ny,x,c) == 0
    def f(inCenterGuess):
        if True:
            moment = computeDualCircumMoment(nx,ny,x,inCenterGuess[0])
            #do('inCenterGuess[0]')
            #do('moment')
            return Vec([moment])
        else:
            moment = computeDualCircumMomentAlt(nx,ny,x,inCenterGuess[0])[0]
            return Vec([moment])

    #initialGuess = Vec([.1])
    #initialGuess = Vec([3.55])
    #initialGuess = Vec([0])
    initialGuess = Vec([1.45])
    if x > 10.5:
        initialGuess = Vec([29.])

    target = Vec([0.])
    nIterations = 20
    eps = 1e-2
    c = newtonSolve(f, target, initialGuess, nIterations, eps)
    print "    out inCenter5special"
    return c


def inCenter4(vs):
    vs = [Vec(v) for v in vs]
    n = len(vs)
    assert n == 4
    inwardNormals = [(vs[(i+1)%n]-vs[i]).perpDot().normalized() for i in xrange(n)]
    offsets = [inwardNormals[i].dot(vs[i]) for i in xrange(n)]
    if use_numpy:
        M = numpy.matrix([
            list(inwardNormals[0])+[1,0],
            list(inwardNormals[1])+[0,1],
            list(inwardNormals[2])+[1,0],
            list(inwardNormals[3])+[0,1],
        ])
        xyrr = numpy.linalg.solve(M,offsets)
    else:
        M = Mat([
            list(inwardNormals[0])+[1,0],
            list(inwardNormals[1])+[0,1],
            list(inwardNormals[2])+[1,0],
            list(inwardNormals[3])+[0,1],
        ])
        xyrr = M.inverse() * Vec(offsets)
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
        if use_numpy:
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
            #x,y,r = numpy.linalg.solve(M,o)
            x,y,r = [float(x) for x in numpy.linalg.solve(M,o)]
        else:
            M = Mat([
                list(inwardNormals[ 0 ])+[-1],
                list(inwardNormals[i+1])+[-1],
                list(inwardNormals[i+2])+[-1],
            ])
            o = Vec([
                offsets[ 0 ],
                offsets[i+1],
                offsets[i+2],
            ])
            x,y,r = M.inverse() * o
        if False:
            # FUDGE
            r = abs(r)
        centers.append(Vec(x,y))
        radii.append(r)
        if False:
            #do('x')
            #do('y')
            do('r')


    if n == 3:
        # single weight will be zero in this case... no point in doing the undefined arithmetic
        return centers[0]

    if n == 4:
        # Either way works, but neither way is robust when cocircular
        if False:
            weights = [
                1./(inwardNormals[3].dot(centers[0]) - offsets[3] - radii[0]),
                1./(inwardNormals[n-3].dot(centers[n-2-1])-offsets[n-3] - radii[n-2-1])
            ]
        else:
            weights = [
                inwardNormals[1].dot(centers[1])-offsets[1] - radii[1],
                inwardNormals[3].dot(centers[0])-offsets[3] - radii[0],
            ]
            # fudge-- this shouldn't be needed, if I get a more robust formula to begin with
            if weights[0] == 0. and weights[1] == 0.:
                weights = [1.,1.]

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
    if weightsSum == 0.:
        return centers[0] # hack
    if False:
        do('[float(weight) for weight in weights]')
        do('weightsSum')
        do('[float(weight/weightsSum) for weight in weights]')

    return sum([center*(weight/weightsSum) for center,weight in zip(centers,weights)])


# exercise inCenter in every way
def inCenterAll(vs):
    print "    in inCenterAll"
    do('vs')


    if False: # don't care so much about generalized inCenter any more, I think I concluded it's a bogus concept
        funNames = []
        if len(vs[0]) == 2:
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

        if len(vs[0]) == 2:
            # Only do inCenterSolve for one permutation
            answer = inCenterSolve(vs)
            print "        inCenterSolve returned "+`answer`

    # Only do pseudoCenterSolve for one permutation
    answer = pseudoCentroid(vs)
    print "        pseudoCentroid returned "+`answer`


    print "    out inCenterAll"
    return answer

# vs are n-dimensional verts on the plane-at-infinity
# in the n+1-dimensional poincare half-space model.
# Compute the point in the n+1-dimensional poincare half-space model
# that, when centered in the n+1-dimensional poincare disk model,
# centers the euclidean average of the verts.
def pseudoCentroid(vs):

    print "        in pseudoCentroid"
    #
    # Initial guess
    # is a generalization
    # of the in-center of 3 points:
    # Take a weighted average of the vertices,
    # each vertex weighted by the variance
    # of the other n-1 vertices,
    # and offset from the plane
    # by the weighted average of the standard deviations
    # (I think this is roughly right).
    # This should immediately give the right answer for a triangle, anyway.
    #


    vs = [Vec(v) for v in vs]
    n = len(vs)
    nDims = len(vs[0])

    M = sum(vs)/n
    S = [sum([(v[i]-M[i])**2 for v in vs]) for i in xrange(nDims)] # n times variance, in each dimension separately

    # We want the variance of each subset
    # of n-1 points of vs.
    # From http://XXX
    # To compute the new S = n*newVariance
    # given s = (n-1)*oldVariance and a new sample x:
    #     S = s + (x-m)*(x-M)
    # so,
    #     s = S - (x-m)*(x-M)
    # where m and M are the old and new means:
    #   M = m + (x-m)/n
    #     = (1-1/n)*m + (1/n)*x
    # so,
    #   m = (M-x/n)/((n-1)/n)
    #     = (M*n-x)/(n-1)
    # The simplest way to think about it
    # is to do each dimension separately;
    # the full variance is the sum
    # of the variances in each dimension.
    weights = []
    for v in vs:
        s = 0.
        for i in xrange(nDims):
            mi = (M[i]*n-v[i])/(n-1.)
            si = S[i] - (v[i]-mi)*(v[i]-M[i])
            s += si
        weights.append(s/(n-1.))

    do('weights')
    weightsSum = sum(weights)

    height = sqrt(sum([weights[j]*weights[i]*(vs[i]-vs[j]).length2() for i in xrange(n) for j in xrange(i)])) / weightsSum


    weightedAvg = sum([weight*v for weight,v in zip(weights,vs)]) / weightsSum
    do('weightedAvg')

    initialGuess = Vec(list(weightedAvg)+[height]) # weightedAvg with height appended

    do('initialGuess')

    zero = Vec([0]*(nDims+1))
    half = Vec([0]*nDims+[.5])
    vs0 = [Vec(list(v)+[0]) for v in vs] # vs with 0 appended to each vertex

    # For the solve, solve for log(height) rather than height.
    doItInLogSpace = True
    #doItInLogSpace = False
    logInitialGuess = Vec(initialGuess[:-1] + [log(initialGuess[-1])])
    do('logInitialGuess')
    if not doItInLogSpace:
        logInitialGuess = initialGuess

    def f(logGuess):
        guess = logGuess[:-1] + [exp(logGuess[-1])]
        if not doItInLogSpace:
            guess = logGuess

        reflectionCenter = Vec(guess[:-1] + [-guess[-1]])
        reflectionRadius = guess[-1]
        assert reflectionRadius > 0 # guaranteed since it's the exp of something... if we were given it directly, then newton might overshoot, which would be a disaster
        # (image poincare disk radius is half that)
        #
        # The following inverts the reflection circle
        # and scales it down to unit size at the origin:
        #       v -> reflectionRadius * (v-reflectionCenter)/(v-reflectionCenter).length2()
        # which means the poincare half-plane
        # will get mapped to a disk of radius 1/2 centered at 0,0,...,1/2.
        # To turn that into a disk centered at 0,
        # we then subtract 0,0,...,1/2.

        imagesSum = Vec(zero)
        for v in vs0:
            image = v - reflectionCenter
            image = image/image.length2() * reflectionRadius
            # image is now on circle of radius 1/2 centered at 0,0,...,1/2
            image -= half
            # image is now on circle of radius 1/2 centered at origin
            imagesSum += image
        return imagesSum

    nIterations = 20
    eps = 1e-6
    logAnswer = newtonSolve(f, zero, logInitialGuess, nIterations, eps)
    answer = logAnswer[:-1] + [exp(logAnswer[-1])]
    if not doItInLogSpace:
        answer = logAnswer

    print "        out pseudoCentroid"
    return answer


# Little test program
if __name__ == '__main__':

    def do(s):
        import inspect
        answer = eval(s, globals(), inspect.currentframe().f_back.f_locals)
        print '            '+s+' = '+`answer`

    if False:
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
    if True:
        #do('inCenterAll([[-15,-2.5],[15,-25],[15,25],[-15,2.5],[-16,0]])')
        do('inCenterAll([[-20],[0],[1]])')
        #do('inCenterAll([[-1],[1],[1.25],[1.5]])')
        #do('inCenterAll([[-1.5],[1.5],[1.5],[-1.5],[-1.5]])')
        #do('inCenterAll([[-1.5],[1.5],[1.5],[-1.5],[-1.6]])')
        #do('inCenterAll([[-15],[15],[15],[-15],[-16]])')
        #do('inCenterAll([[-2.5],[-25],[25],[2.5],[0]])')
    if True:
        do('inCenterAll([[1,0],[0,1],[-1,0]])')
    if False:
        do('inCenterAll([[0,0],[50,0],         [20,22.5],[0,7.5]])')


    if False:
        do('inCenterAll([[0,0],[40,0],[40,7.5],[20,22.5],[0,7.5]])')
        do('inCenterAll([[0,0],[37.5,0],[42,6],[20,22.5],[0,7.5]])')
        if False:
            # obtuse
            for x in [5,6,7,8,9,10,100,1000,10000,1e5,1e6,1e7,1e8,1e9]:
                do('inCenterAll([[-2.5,5],[-6.25,0],[-2.5,-5],['+str(x)+',-5],['+str(x)+',5]])')
        if True:
            # acute
            for x in [5,6,7,8,9,10,100,1000,10000,1e5,1e6,1e7,1e8,1e9]:
                do('inCenterAll([[-5/3.,5],[-25/3.,0],[-5/3.,-5],['+str(x)+',-5],['+str(x)+',5]])')
        if False:
            # right
            for x in [5,6,7,8,9,10,100,1000,10000,1e5,1e6,1e7,1e8,1e9,1e10,1e11,1e12]:
                do('inCenterAll([[-2.5,5],[-7,-1],[-5/3.,-5],['+str(x)+',-5],['+str(x)+',5]])')


    if False:
        #circumCenterSequence([[0,0],[50,0],[20,22.5],[0,7.5]])
        a = Vec(20,8.501010913691152)
        b = Vec(20.761099158704887, 8.4349521148941093)
        c = Vec(35.,5.)
        do('(b-a).cross(c-a)')

    if False:
        #for x in [1,10,1e2,1e3,1e4,1e5,1e6]:
        for x in [1,10,100]:
            do('inCenterAll([[-.5,1],[-1.25,0],[-.5,-1],['+str(x)+',-1],['+str(x)+',1]])')
            do('inCenter5special(-4/5.,3/5.,'+str(x)+')')

        #do('inCenter5special(-4/5.,3/5.,2)')
        #do('inCenter5special(-3/5.,4/5.,1)')

    if False:
        do('circumCenterAll([[0,0],[2,0],[2,1],[1,2],[0,2]])')
        do('circumCenterAll([[0,0],[2,0],[1,2],[2,1],[0,2]])')

    if True:
        do('inCenterAll([[-1],[0],[2]])')
        do('inCenterAll([[-1.,.3,4],[.4,.6,.7],[.22,.743,.37]])')

