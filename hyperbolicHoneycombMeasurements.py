#!/usr/bin/python

'''

Here's a more detailed outline of how I'd go about drawing
the intersection of {3,3,7} with the sphere at infinity.
I'll probably do this in a few days if no one else does it first.

To get started, consider the center cell
of a cell-centered {3,3,7}.
The intersection of that cell with the sphere at infinity
consists of three little regular spherical triangles;
we need to know the euclidean coords of one of these
little spherical triangles.

We can find that in 4 steps:

Step 1: compute the cell mid-radius r31{3,3,7}.

    The reference paper "Regular Honeycombs In Hyperbolic Space" by HSM Coxeter
    doesn't give a direct formula for cell mid-radius r31,
    but it gives a formula for the cell in-radius r32.
    From that, we can use the identity
        sin(A)/sinh(a) = sin(B)/sinh(b) = sin(C)/sinh(c)
    on the right hyperbolic triangle
    formed by the cell center, face center, and edge center:
          sin(pi/2)/sinh(r31) = sin(pi/r)/sinh(r32)
    i.e.          1/sinh(r31) = sin(pi/r)/sinh(r32)
    i.e.            r31 = asinh(sinh(r32)/sin(pi/r))
    So, plug in the formula for r32 from the paper:
        r32{p,q,r} = acosh(sin(pi/p)*cos(pi/r)/sqrt(1-cos(pi/p)^2-cos(pi/q)^2))
    and now we have a formula for r31{p,q,r}, the cell mid-radius:
        r31{p,q,r} = asinh(sinh(acosh(sin(pi/p)*cos(pi/r)/sqrt(1-cos(pi/p)^2-cos(pi/q)^2)))/sin(pi/r))
    (This can certainly be simplified, using the identity
    sinh(acosh(x)) = sqrt(x^2-1), and probably further simplifications,
    but I'm not bothering here.)
    (And note, I'm not positive I got all of the above exactly right,
    but the method should be sound.)
    Substitute p=3,q=3,r=7 to get the desired cell-mid-radius of {3,3,7}.

Step 2: from r31, compute the euclidean distance
        from the origin to an edge of the center cell in the poincare ball model.

    If I recall correctly, that will be tanh(r31/2).

Step 3: from that, compute the actual coords of the two endpoints-at-infinity
    of one edge of the center cell.

    For definiteness, align the center cell
    with the regular euclidean tetrahedron
    with verts:
        (-1,-1,1)
        (-1,1,-1)
        (1,-1,-1)
        (1,1,1)
    The center of the cell's edge closest to joining -1,-1,1 to 1,1,1
    lies on the +z axis, so by Step 2 this edge center is:
        (0,0,tanh(r31/2))
    The two endpoints-at-infinity of that edge
    will be (-sqrt(1/2),-sqrt(1/2),0) and (sqrt(1/2),sqrt(1/2),0)
    "translated by (0,0,tanh(r31/2))",
    i.e.  transformed by the translation
    that takes the origin to that edge center (0,0,tanh(r31/2)).

    Recall that for any points p,t in the poincare ball
    (of any number of dimensions), p translated by t is:
        ((1 + 2*p.t + p.p)*t + (1-t.t)*p) / (1 + 2*p.t + p.p * t.t)
    where "." denotes dot product.  (Hope I wrote that down right.)
    So plug in:
        t = (0,0,tanh(r31/2))
        p = (sqrt(1/2),sqrt(1/2),0)
    (a bunch of terms simplify and drop out since p.p=1 and p.t=0, but whatever);
    The resulting endpoint coords are (a,a,b) for some a,b
    (then the other endpoint is (-a,-a,b), but we don't need that at this point).

Step 4: rotate one of those endpoints-at-infinity
    around the appropriate axis
    to get the other two vertices of the little spherical triangle.
    The three spherical triangle vertices are:
        (a,a,b)
        (b,a,a)
        (a,b,a)
    (where a,b are the result of step 3).

=========================================================================
So now we have one little spherical triangle.
Now, choose a set of 3 generators for the rotational symmetry group
of {3,3,7}, and use them repeatedly to send the triangle everywhere.
There are lots of choices of 3 generators; here's one:
    A: plain old euclidean rotation by 120 degrees about the vector (1,1,1)
    B: plain old euclidean rotation by 120 degrees about the vector (1,1,-1)
    C: rotation by 2*pi/7 about an edge of {3,3,7}.
       for this, we can use the composition of:
           translate the edge center to the origin
                (i.e. translate the origin to minus the edge center)
           followed by plain old euclidean rotation of 2*pi/7 about this edge-through-the-origin
           followed by translating the origin back to the original edge center
       A specific edge center, and the translation formula,
       can be found in Step 3 above.

Don


'''

import sys
import os

from cmath import pi, sqrt, cos,sin,tan, cosh,sinh,acosh,asinh,tanh


# length of edge of characteristic simplex.
# actually returns its cosh^2, its cosh and its value.
# If i0,i1 is:
#       0,1 -> vertex to edge center      (i.e. half edge length, i.e. dual cell in-radius)
#       0,2 -> vertex to face center      (i.e. dual cell mid-radius)
#       0,3 -> vertex to cell center      (i.e. cell circum-radius, i.e. dual cell circum-radius)
#       1,2 -> edge center to face center
#       1,3 -> edge center to cell center (i.e. cell mid-radius)
#       2,3 -> face center to cell center (i.e. cell in-radius, i.e. dual half edge length)
def measure(schlafli, i0,i1):

    if i0 == i1:
        return 0.,0.
    if i0 > i1:
        i0,i1 = i1,i0

    if len(schlafli) == 0:
        # 0 dimensional surface, {}
        # can't get here-- i0,i1 must be 0,0 which was handled above
        assert False
    if len(schlafli) == 1:
        # 1 dimensional surface, {p}
        p = schlafli
        if (i0,i1) == (0,1):
            assert False # I don't think this is well-defined... maybe infinite?
        else:
            assert False
    elif len(schlafli) == 2:
        # 2 dimensional surface, {p,q}
        p,q = schlafli
        if (i0,i1) == (0,1):
            # half edge length
            coshValue = cos(pi/p)/sin(pi/q)
        elif (i0,i1) == (1,2):
            # face in-radius
            coshValue = cos(pi/q)/sin(pi/p)
        elif (i0,i1) == (0,2):
            # face circum-radius
            coshValue = 1/(tan(pi/p)*tan(pi/q))
        else:
            assert False
    elif len(schlafli) == 3:
        # 3 dimensional surface, {p,q,r}
        p,q,r = schlafli
        def sin_pi_over_h(p,q):
            # = sqrt(1 - cos^2 (pi / h(p,q)))
            # = sqrt(1 - (cos(pi/p)^2 + cos(pi/q)^2)
            return sqrt(1 - (cos(pi/p)**2 + cos(pi/q)**2))
        if (i0,i1) == (0,1):
            # half edge length
            coshValue = cos(pi/p)*sin(pi/r)/sin_pi_over_h(q,r)
        elif (i0,i1) == (2,3):
            # cell in-radius
            coshValue = sin(pi/p)*cos(pi/r)/sin_pi_over_h(p,q)
        elif (i0,i1) == (0,3):
            # cell circum-radius
            coshValue = cos(pi/p)*cos(pi/q)*cos(pi/r)/(sin_pi_over_h(p,q)*sin_pi_over_h(q,r))
        elif (i0,i1) == (0,2):
            # 2d face circum-radius
            cosh2_r01,cosh_r01,r01 = measure(schlafli,0,1)
            sinh_r01 = sqrt(cosh2_r01-1)
            sinhValue = sinh_r01/sin(pi/p)
            coshValue = sqrt(1+sinhValue**2)
        elif (i0,i1) == (1,3):
            # cell mid-radius
            return measure([r,q,p], 0,2)
        elif (i0,i1) == (1,2):
            # 2d face in-radius
            # We can calculate this in one of two ways,
            # using the hyperbolic right triangle identities:
            #   cos A = tanh b / tanh c
            #   sinh a / sin A = sinh c / 1
            # => sinh a = sinh c * sqrt(1 - (tanh b / tanh c)^2)
            # (should try to simplify)

            cosh2_b,cosh_b,b = measure(schlafli, 0,1)
            cosh2_c,cosh_c,c = measure(schlafli, 0,2)
            sinh_a = sinh(c) * sqrt(1 - (tanh(b)/tanh(c))**2)
            do('sinh_a')

            cosh2_b,cosh_b,b = measure(schlafli, 2,3)
            cosh2_c,cosh_c,c = measure(schlafli, 1,3)
            sinh_a = sinh(c) * sqrt(1 - (tanh(b)/tanh(c))**2)
            do('sinh_a')

            # Trying to simplify the former...

            r01 = acosh(cos(pi/p)*sin(pi/r)/sin_pi_over_h(q,r))

            cosh2_r01 = (cos(pi/p)*sin(pi/r)/sin_pi_over_h(q,r))**2

            sinh_r02 = sqrt(cosh2_r01-1)/sin(pi/p)
            tanh_r02 = 1/sqrt(1+1/sinh_r02**2)
            r02 = asinh(sinh_r02)

            sinh_a = sinh_r02 * sqrt(1 - (tanh(r01)/tanh_r02)**2)
            do('sinh_a')

            coshValue = sqrt(1+sinh_a**2)
        else:
            assert False # illegal
    elif len(schlafli) == 4:
        # 4 dimensional surface, {p,q,r,s}
        p,q,r,s = schlafli
        if (i0,i1) == (0,1):
            # half edge length
            assert False # unimplemented
        elif (i0,i1) == (3,4):
            # facet in-radius
            assert False # unimplemented
        elif (i0,i1) == (0,4):
            # facet circum-radius
            assert False # unimplemented
        else:
            assert False # illegal
    else:
        assert False # unimplemented

    return coshValue**2, coshValue,acosh(coshValue)



# Little test program
if __name__ == '__main__':

    def do(s):
        import inspect
        answer = eval(s, globals(), inspect.currentframe().f_back.f_locals)
        print '            '+s+' = '+`answer`

    tau = (sqrt(5)+1)/2

    do('measure([7,3],0,1)')
    do('measure([7,3],0,2)')
    do('measure([7,3],1,2)')

    do('measure([7/2.,7],0,1)')
    do('measure([7/2.,7],0,2)')
    do('measure([7/2.,7],1,2)')
    do('cos(2*pi/7)/sin(pi/7)')     # 0,1 from book
    do('1/(tan(pi/7)*tan(2*pi/7))') # 0,3 from book
    do('1/(2*sin(pi/7))')           # 2,3 from book


    do('measure([5,3,4],0,1)')
    do('measure([5,3,4],0,3)')
    do('measure([5,3,4],2,3)')
    do('.5*tau**2') # 0,1 from book
    do('.5*tau**4') # 0,3 from book
    do('.5*sqrt(5)*tau') # 2,3 from book

    do('measure([5,3,5],0,1)')
    do('measure([5,3,5],0,3)')
    do('measure([5,3,5],2,3)')
    do('.25*sqrt(5)*tau**3') # 0,1 and 2,3 from book
    do('.25*tau**8')         # 0,3 from book

    do('measure([6,3,3],0,1)')
    do('measure([6,3,3],0,3)')
    do('measure([6,3,3],2,3)')

    do('measure([3,6,3],0,1)')
    do('measure([3,6,3],0,3)')
    do('measure([3,6,3],2,3)')

    do('measure([4,4,3],0,1)')
    do('measure([4,4,3],0,3)')
    do('measure([4,4,3],2,3)')

    do('measure([7,3,3],0,1)')
    do('measure([7,3,3],0,3)')
    do('measure([7,3,3],2,3)')

    do('measure([7,3],0,1)')
    do('measure([7,3,2],0,1)')
    do('measure([7,3,3],0,1)')
    do('measure([7,3,4],0,1)')
    do('measure([7,3,5],0,1)')
    do('measure([7,3,6],0,1)')
    do('measure([7,3,7],0,1)')
    do('measure([7,3,8],0,1)')

    def coshHalfEdgeLength(p,q,r):
        return cos(pi/p)*sin(pi/r)/sqrt(1-cos(pi/q)**2-cos(pi/r)**2)
    def halfEdgeLength(p,q,r):
        return acosh(cos(pi/p)*sin(pi/r)/sqrt(1-cos(pi/q)**2-cos(pi/r)**2))

    do('halfEdgeLength(7,3,2)')
    do('halfEdgeLength(7,3,3)')
    do('halfEdgeLength(7,3,4)')
    do('halfEdgeLength(7,3,5)')
    do('halfEdgeLength(7,3,6)')
    do('halfEdgeLength(7,3,7)')
    do('halfEdgeLength(7,3,8)')
    for n in [2,3,4,5,6,7,8,9]:
        p = 7
        q = 3
        r = n
        print "    {"+`p`+","+`q`+","+`r`+"} -> acosh("+`coshHalfEdgeLength(p,q,r)`+") = "+`halfEdgeLength(p,q,r)`+""
    do('halfEdgeLength(2,3,7)')

    do('measure([3,3,7], 0,1)')
    do('measure([3,3,7], 0,2)')
    do('measure([3,3,7], 0,3)')
    #do('measure([3,3,7], 1,2)')
    do('measure([3,3,7], 1,3)')
    do('measure([3,3,7], 2,3)')

    do('measure([5,3,4],0,1)')
    do('measure([5,3,4],0,2)')
    do('measure([5,3,4],0,3)')
    do('measure([5,3,4],1,2)')
    do('measure([5,3,4],1,3)')
    do('measure([5,3,4],2,3)')







