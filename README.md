XXX there are more notes in net.c++
XXX conjectures about polyhedra with all face planes unit distance from origin


NOTES ON SHEPHARD'S CONJECTURE

Trying to organize all the conjectures and counterexamples I know
related to Shephard's conjecture, roughly in order of strength,
strongest (false) to weakest (true).
I'm mainly interested in trivalent polyhedra at this point,
so I'll stick to those.


CONJECTURES ABOUT SPECIFIC STRONGER UNFOLDINGS THAT FOLLOW A PATTERN:

* For every trivalent convex polyhedron,
  the sharpest-dihedral-sum spanning cut tree is the cut tree of an unfolding
  [false, since it implies the next one which is false]

* Every trivalent convex polyhedron has an unfolding in which
  the sharpest edge at every vertex is a cut.
  [false. counterexample: netless9, i.e. hexes spiral]

* For every trivalent convex polyhedron,
  the dullest-dihedral spanning fold tree is a good unfolding
  [false, since it implies the next one which is false]

* Every trivalent convex polyhedron has an unfolding in which
  the dullest edge at every face is a fold.
  [false, counterexample: CannedThingHexesSpiralOther]

* Every trivalent convex polyhedron has an unfolding in which
  the dullest edge at every vertex is a fold.
  [false, easy counterexample: just take a regular pentagonal prism
  (jittered slightly so trivalent),
  or a slightlys squashed regular tetrahedron,
  or a regular 7-gon in the plane with edges going straight out
  from the center-- those spokes can't all be folds.]

* Every trivalent convex polyhedron has an unfolding in which
  the sharpest edge at every face is a cut.
  [false, easy counterexample: take a finely tesselated sphere,
  and intersect it with a half-space that retains about half of it.
  the edges around the big face can't all be cuts]

[similar conjectures about unfoldings which cut dullest edges
or fold sharpest edges omitted, since they seem unlikely]

[similar conjectures about unfoldings that cut shortest edges
or fold longest edges should be included, though]

[also some sort of weighted combinations of longest-dullest
or shortest-sharpest criteria would be interesting]


CONJECTURES ABOUT STRONGER UNFOLDINGS INVOLVING SWEEPS:

* For every trivalent convex polyhedron,
  for every sweep direction, there is an unfolding
  in which the cut tree follows the sweep direction
  [false, counterexample: sweepKiller, or sweepKiller1]

* For every trivalent convex polyhedron,
  there is a sweep direction and an unfolding
  in which the cut tree follows the sweep direction
  [false, counterexample: something with lots of little sweepKillers
  at different angles, to thwart every possible sweep direction.
  angle delta just needs to be small enough so that turning that much
  doesn't move any edge past horizontal.)

* For every trivalent convex polyhedron,
  for every sweep direction, there is an unfolding
  in which the faces and folds follow the sweep direction
  [unknown, implies the following]

* For every trivalent convex polyhedron,
  there is a sweep direction and an unfolding
  in which the faces and folds follow the sweep direction
  [unknown]





CONJECTURES ABOUT REFINEMENTS:

* Given any unfolding F of a trivalent convex polyhedron P,
  and a trivalent convex polyhedron P' which refines P
  (i.e. is obtained from P by intersecting with one or more additional
  face plane halfspaces),
  there is an unfolding of F' of P' that refines F
  (i.e. for each edge e of P and e' of P' such that e' is part of e,
  e' is a cut in F' iff e is a cut in F).
  [false, counterexample: netless8]

* There is a function f from trivalent convex polyhedra to unfoldings,
  such that for every trivalent convex polyhedron P, f(P) is an unfolding
  of P, and for every P and P' such that P' is a refinement of P
  (i.e. formed by intersecting with more face places),
  f(P') is a refinement of f(P).
  [unknown, implies planar-trivalent-shephard's.
   Note that blindly picking an arbitrary unfolding that refines
   all previously chosen unfoldings doesn't work,
   since it leads to a dead end, as shown in previous conjecture]


CONJECTURES ABOUT ALGORITHMS:

* There is an algorithm for producing an unfolding
  of every trivalent convex polyhedron whenever one exists,
  that only makes local decisions (i.e. the decision of whether
  to cut an edge depends on examining only a fixed bounded graph distance
  away)
  [unknown]

* There is a polynomial time algorithm
  for producing an unfolding of every trivalent convex polyhedron
  whenever one exists.
  [unknown]

* There is a nondeterministic polynomial time algorithm
  for producing an unfolding of every trivalent convex polyhedron
  whenever one exists.
  [true, since a proposed unfolding can be checked in polynomial time]

* There is a polynomial time algorithm
  for proving there is no unfolding of a given trivalent convex polyhedron,
  when there isn't
  [unknown, trivalent shephard's implies this trivially]

* There is a nondeterministic polynomial time algorithm
  for proving there is no unfolding of a given trivalent convex polyhedron,
  when there isn't
  [unknown, trivalent shephard's implies this trivially]


* The problem of finding an unfolding of a trivalent convex polyhedron
  if one exists is NP-hard (which is equivalent to saying it's NP-complete,
  since the problem is obviously in NP)
  [unknown]
  [XXX generally I've tried to state all conjectures as
  "for all polyhedra there exists an unfolding..."
  or "there exists an algorithm", however this one is backwards...
  maybe this conjecture should be replaced with its negation]


CONJECTURES ABOUT SINGLE CRACKS:

[XXX need to explain exactly what I mean by infinite planar polyhedron
 and crack]

* In any trivalent "infinite planar" polyhedron,
  for every vertex v and edge e incident on v,
  there is an infinite crack whose head is v
  and whose first edge is e.
  [false, counterexample: crackkiller]

* In any trivalent "infinite planar" polyhedron,
  For every pair of vertices v0,v1,
  there is an infinite tree with at most one fork
  whose heads are at most v0 and v1
  and which contains the sharpest edge at v0 and v1 respectively.
  [unknown, implies the following]

* In any trivalent "infinite planar" polyhedron,
  For every vertex v, there is an infinite crack
  whose head is v and whose first edge is the sharpest
  edge at v.
  [unknown]

* In any trivalent "infinite planar" polyhedron,
  For every pair of vertices v0,v1,
  there is an infinite tree with at most one fork
  whose heads are at most v0 and v1.
  [unknown, implies the following]

* In any trivalent "infinite planar" polyhedron,
  For every vertex v, there is an infinite crack
  whose head is v.
  [unknown]

* In any trivalent "infinite planar" polyhedron,
  For every pair of vertices v0,v1,
  there is an infinite crack
  containing v0 and v1.
  [unknown, probably false, probably not very useful, implies the following]

* In any trivalent "infinite planar" polyhedron,
  For every pair of vertices v0,v1,
  there is an infinite tree with at most one fork
  containing v0 and v1.
  [unknown, implies the following]

* In any trivalent "infinite planar" polyhedron,
  For every vertex v, there is an infinite crack
  containing v.
  [unknown]


CONJECTURES ABOUT UNFOLDINGS ON INFINITE PLANAR POLYHEDRA WITH 3 INFINITE EDGES:

* For every trivalent planar polyhedron with exactly 3 infinite edges,
  for each of those infinite edges, there is an unfolding
  in which that infinite edge is a cut and the other two are folds
  [false, counterexample: netless8]

* Every trivalent planar polyhedron with exactly 3 infinite edges
  has an unfolding in which its sharpest infinite edge
  is a cut and the other two infinite edges are folds
  [unknown, implies the next one which implies planar-trivalent-shephard's]

* Every trivalant planar polyhedron with exactly 3 infinite edges
  spaced regularly
  has an unfolding in which exactly 1 of the 3 infinite edges
  is a cut
  [unknown, implies planar-trivalent-shephard's]

OTHER CONJECTURES ABOUT INFINITE PLANAR POLYHEDRA:
* Every trivalent planar polyhedron has an unfolding
  in which every subtree has a center of curvature
  no higher than the vertex it hangs from, if the edge it hangs from
  is oriented vertically
  [trivalent shephard's is equivalent to this,
  I believe, although definitions need to be nailed down]
  [see observations, down below]






SHEPHARD'S CONJECTURE:

* Shephard's Conjecture: every convex polyhedron has an unfolding.

* "Trivalent Shephard's Conjecture":
  Every trivalent convex polyhedron has an unfolding.
  (This is apparently weaker than Shephard's conjecture.
   But if shephard's is false and this is true,
   that would mean there is a non-trivalent polyhedron
   that is not unfoldable, but every infinitesimal perterbation of it
   into a trivalent polyhedron is unfoldable-- which means
   every unfolding of such a perturbation of it
   must have a fold on one of those infinitesimal introduced edges,
   which seems unlikely.)

- Every convex polyhedron with planes all unit distance from origin
  has an unfolding.
  (in plane, these correspond to voronoi diagrams)

- Every trivalent convex polyhedron with planes all unit distance
  from origin has an unfolding.
  (in plane, these correspond to non-degenerate voronoi diagrams)



CONJECTURES ABOUT HIGHER DIMENSIONS:

* 4d shephard's: Every convex polychoron (4d polytope) has an unfolding.

* 4d tetravalent shephard's: Every tetravalent convex polychoron (4d polytope)
  has an unfolding.
  [unknown - I think I had the impression at one time
   that this was easy though--
   obviously true, or there is an easy counterexample?]






============================================================================

OBSERVATIONS ABOUT INFINITE PLANAR POLYHEDRA (TILINGS):

I believe that deciding Shepard's conjecture
boils down to a question about planar tilings,
(i.e. certain certain degenerate convex polyhedra).

That is, I believe that if Shephard's conjecture in the plane
is shown to be true using an algorithm, then that algorithm can be made
to work on actual polyhedra,
and if Shephard's conjecture in the plane is false,
then a counterexample to it can be easily modified
(by projecting onto a sufficiently large sphere or paraboloid) to form
a polyhedron counterexample to Shephard's conjecture.

Maybe that conjecture can be proved formally, independently
of any of the others, that is:
    If Shephard's conjecture is false,
    i.e. if there's a netless convex polyhedron,
    then there's a netless convex polyhedron
    where the impossible part is arbitrarily close to flat.
    (Maybe even with same structure as the original?)

Restricting our attention to planar tilings
simplifies some details so that there is less to think about,
and we get some nice duality properties.

As usual, I'm restricting my attention to trivalent tilings.
Furthermore when I say "trivalent planar tiling", I mean one
that is the limit of a process in which
part of the surface of a polyhedron becomes flatter and flatter.
So this excludes planar tilings which don't arise as such a limit,
e.g. a regular polygon with all the spokes emanating from it
twisted slighly counterclockwise
(more on the impossibility of this case in (6) below).

[XXX to be precise, I want to restrict attention
to tilings of the whole plane, consisting of a finite number of vertices,
edges, and convex cells, with some of the edges and cells infinite.
Need to give this class of tilings a good clear name.]

[XXX and note that when I say cut tree or forest,
I mean a cut tree or forest that produces an unfolding tree,
overlapping or not...
i.e. the cut forest must be a spanning forest, and every tree in it has exactly one
infinite edge (less than one: unfolding would be cyclic;
more than one: unfolding would be disconnected)
Think whether there is a better crystal clear definition for this.
Show examples of the various cases and near misses.]

The following observations are stated without proof;
they become evident from playing with planar tilings:

(1) Every infinite trivalent planar tiling has a dual triangulation,
    which is unique up to translation and scale
    (obtained as the limit of the spherical-reciprocal polyhedra
    of the sequence of polyhedra whose limit is the tiling),
    although many trivalent planar tilings map to the same dual triangulation.
    The original tiling is fully specified by its dual triangulation
    and an infinitesimal "height above the sphere"
    for each dual vertex (or, equivalently,
    an infinitesimal depth below the sphere
    for each face plane in the original tiling).

    When all these heights-above-the-sphere are zero,
    the tiling is a Voronoi diagram
    and the dual triangulation is a Delaunay triangulation,
    and in that case each Voronoi vertex is the circumcenter
    of the corresponding dual Delaunay triangle.
    This is the limiting case of the
    analagous condition for polyhedra,
    which is that the primal trivalent polyhedron has an insphere
    and the dual polyhedron (whose faces are all triangles) has a circumsphere.

    [XXX pictures of simple examples]

    Note that if we omit the trivalent condition,
    we do *not* necessarily get a unique dual.
    [XXX picture of simple example]
    Since uniqueness of the dual is what allows us to
    compute gaussian curvatures (as will be explained below),
    this is the main reason we avoid non-trivalent tilings.

(2) The edge curvatures (i.e. the infinitesimal values
    pi minus dihedral angle of an edge)
    are proportional to dual edge lengths in the dual triangulation
    (so they are independent of the height-above-the-sphere values;
    i.e. pushing face planes in and out without changing the structure or angles
    does not change dihedrals; this is true in general about polyhedra).
    (This is the limiting case of the statement
    that the dihedral angle of an edge of a non-degenerate polyhedron
    is equal to the arc length of the spherical arc
    obtained by projecting the dual edge onto the unit sphere.)

    This tells us that, at any (trivalent) vertex,
    we can recognize the sharpest of the three dihedral angles
    by any of the following equivalent criteria:
        * it has the longest of the three edges on the dual triangle
        * its opposite angle on the dual triangle is largest
        * its opposite interior angle in the primal tiling is smallest
    Similarly, the dullest of the three dihedrals is the one
    satisfying the exact opposite criteria.

    (Similar criteria hold for non-degenerate polyhedra, as well.)

            +
            |   .     sharpest
            |       . /
            |        /  .
   dullest ---------+       .
            |       |           .
            +-------|---------------+
                    |
                        


(3) The vertex (gaussian) curvatures (i.e. the infinitesimal values 2 pi minus
    sum of interior angles at a vertex)
    are proportional to the respective triangle areas in the dual triangulation
    (so they are independent of the height-above-the-sphere values;
    i.e. pushing face planes in and out without changing the structure or angles
    does not change vertex curvatures; this is true in general about polyhedra).
    (This is the limiting case of the statement
    that the gaussian curvature of a vertex of a non-degenerate polyhedron
    is equal to the area of the spherical triangle obtained by projecting
    the dual face onto the unit sphere.)

(4) Define the total curvature of any region of a tiling
    to be the sum of the curvatures of all the vertices in the region
    (i.e. sum of the respective dual triangle areas).
    Define the center of curvature of a region to be
    the weighted average of the positions of the vertices in the region,
    using each vertex's curvature as its weight.
    Then it's immediately clear that total curvature
    and center of curvature of regions are additive
    in the obvious way; that is, given a disjoint union of regions,
    the curvature of the union is the sum of the curvatures,
    and the center-of-curvature of the union is the weighted average
    of the centers-of-curvature of the individual regions.
    Furthermore, total curvature and center-of-curvature of a region are robust,
    in the sense that truncating or un-truncating by introducing, removing,
    or adjusting face planes within a region never changes the curvature
    or center-of-curvature of the region.
    (Robustness of total curvature is fairly obvious, by noticing
    that it is a statement about sums of triangle areas in the dual
    triangulation, which doesn't change under any of the listed operations;
    robustness of center-of-curvature is not obvious at all, to me.)

    [XXX REF: I made web pages proving the nontrivial inductive step:
        - circumCenterProof1.svg: swapping quad diagonal doesn't change center-of-curvature
        - circumCenterProof2.svg: removing a trivalent vertex, or adding a vertex
                in the middle of a triangle, doesn't change center-of-curvature]

    Note that total curvature clearly maps to a property
    of the dual triangulation, namely total area of the dual triangles.
    However, unfortunately, center-of-curvature doesn't seem to
    map so nicely to any clear property in the dual triangulation.
    An exception is the Voronoi/Delaunay case (see (1) above),
    in which the center of curvature of each triangle is its circum-center.
    In this case the center of curvature of any union of triangles
    is the "generalized circum-center" of the union polygon.
    (It turns out, surprisingly, that this is a robust concept
    even for non-cocircular polygons;
    that is, we can define the generalized circum-center
    of any polygon to be the area-weighted average of the circumcenters
    of the triangles in any triangulation of it,
    and we get a consistent answer regardless of the particular triangulation.)

    [XXX example, of a single trivalent vertex
     and then finely truncated... same curvature and center-of-curvature]

(5) We can always calculate the center-of-curvature
    of a set of vertices in a trivalent planar tiling
    by drawing the dual triangulation and using the dual triangle areas
    as weights to find the weighted average of the vertices.
    But if we have two vertices v0,v1 joined by an edge,
    the following method is often a more direct way of locating their
    center of curvature.
    Label the four other edges incident on the edge v0v1
    A,B,C,D in counterclockwise order, so that A,B are incident on v0
    and C,D are incident on v1:

          A          D
           \         |
            \        |
             v0-----v1
            /         \
           /           \
          B             C

    We know that the center-of-curvature is some non-negative-weighted
    average of v0 (intersection of A and B) and v1 (intersection of C and D),
    i.e. it lies somewhere on the edge joining them.
    By the previous robustness statement about center-of-curvature,
    the center of curvature doesn't depend on the specific structure
    of what's going on inside a region, so we can think of a region
    around the edge v0v1 as a black box:
          A          D
           \         |
          ..............
          :            :
          :............:
           /           \
          B             C
    And we know that the center-of-curvature lies on the segment joining
    the intersection of A and B with the intersection of C and D.
    By symmetry, we might expect that the center-of-curvature
    will also lie on the segment joining the intersection of B and C
    with the intersection of A and D,  It turns out that that is
    in fact the case, and this allows us to locate the center-of-curvature
    exactly, by simply intersecting lines and connecting intersection points.

    Extend A and D so that they meet at a point p,
    and extend B and C so that they meet at a point q:

                 q 
          A     /.\  D
           \   / . \ |
            \ /  .  \|
             v0---.-v1
            / \   .  |\
           /   \   . | \
          B     \  . |  C
                 \ . |
                  \ .|
                   \.|
                    \|
                     p

    Then the center of curvater of v0 and v1
    is the intersection of the lines v0v1 and pq.

    [XXX transcribe into a nicer picture]

(6) Given a polygon in a trivalent planar tiling,
                  | 
         ----+----+
            /      \ /
        ---+        +
           |        |
           +        +---
          / \      /
             +----+
             |    ???
   [XXX transcribe into a nicer picture]
   and given the directions of all but one of the spokes
   emanating from that polygon, the direction of the last spoke
   is uniquely determined.
   That's easy to see by drawing the dual triangulation:
   if we've drawn all but one of the dual triangles incident on the center,
   then the last one is determined.

   In particular, consider the case when the polygon is regular,
   and all but one of the spokes is tilted slightly counterclockwise.
   Drawing the dual triangulation, it's evident that the triangles
   get larger and larger as we travel around the polygon clockwise,
   but the final/largest dual triangle
   (corresponding to the final vertex of the polygon)
   will be adjacent to the first/smallest dual triangle,
   which means the final spoke must be tilted clockwise
   instead of counterclockwise.
   I.e. the spokes emanating from a regular
   polygon can't all spiral out in the same direction
   (counterclockwise or clockwise).

(7) Given a cut tree (or forest),
    and a vertex v and edge e incident on v
    such that the subtree consisting of all nodes and cuts
    closer in the cut tree to v than to e is finite,
    call v,e "unbalanced" if the center of curvature c of that subtree
    is strictly "higher" than v (i.e. c on the e side
    of the line through v perpendicular to e),

    (Note, in the definition of unbalanced, sometimes it's more convenient
    to exclude v from the center of curvature calculation.  Whether we choose
    to include or exclude v has no effect on whether the result is
    higher than v or not, so the definition ends up the same either way.)

    If there is an unbalanced v,e, then the unfolding has an (infinitesimal)
    overlap.

                        |       ........
                        |e      :  \ / :
                        |       : c +  :
                        v       :  /   :
              ....     / \  ....: /    :
              :  :..........:    /     :
              :      /\   /\    /      :
              :             +--+       :
              :                 \      :
              .........................:

    This is the planar/degenerate version
    of Alpha/Beta Rule 1 of Alex Benton's dissertation,
    namely that beta_Q >= pi/2, where beta_Q is the angle formed
    by c,v,e. c is the Virtual Root of the subtree.
    (In the non-degenerate case, the Alpha/Beta rule condition is that
    beta_Q >= (pi-alpha_P)/2, but in the degenerate case
    alpha_P is infinitesimal so the rule becomes simply beta_Q >= pi/2.)

    I'm pretty sure the converse is true as well;
    that is, if a cut tree is "completely balanced"
    (i.e. there are no unbalanced v,e)
    then the resulting unfolding is non-overlapping.
    I certainly can't think of any way in which this could fail.

    I believe this will be the key to solving Shephard's conjecture:
    that is, if Shephard's conjecture is false,
    then the simplest counterexample will be
    a planar tiling such that every cut tree
    has an unbalanced v,e (such a counterexample leads easily to
    an actual nondegenerate polyhedral counterexample, by projecting
    onto the surface of a large sphere or paraboloid).
    And, if Shephard's conjecture is true, then I think it can be proved
    by first proving that every planar tiling has a completely balanced
    cut tree, and by then using the same technique to come up with
    an algorithm for the messier details of the non-degenerate case.

(8) Given any cut tree and any vertex,edge v,e with v incident on e,
    such that the subtree consisting of all nodes and cuts
    closer in the cut tree to v than to e is finite,
    call that finite subtree the "lagoon" formed by removing e
    from the cut tree.

    The final (conjectured) algorithm of Alex Benton's dissertation
    proceeds by taking an unfolding and repairing it
    by repeatedly removing a bad edge
    and replacing it with a good exit from the resulting lagoon.
    This relies on some unproven properties of lagoons and good exits.

    It is easy to prove that every lagoon
    has at least one "good exit",
    that is, an edge e' with one of its vertices v' in the lagoon
    and the other v'' out of it, such that v',e' is balanced.
    (v' can be taken to be any vertex in the lagoon
    at maximal distance from the center of curvature of the lagoon).

    Conjecture: there are always at least *two* good exits
    from any lagoon formed in this way.

    Series of weaker conjectures:
        STRONGEST (known false)
        every lagoon has at least three good exits
                (counterexample: one edge, with a bad exit out of each endpoint)
        <=> false
         => every lagoon has at least two good exits  (MAIN CONJECTURE)
         => every convex lagoon as at least two good exits
                (where "convex" means it includes every vertex in its convex hull)
         => every lagoon that's a *subset* of the verts of one polygon has at least two good exits
         => true
        <=> every lagoon consisting of the verts of one polygon has at least two good exits
                (This is the dudley/priscilla proof, done but needs polishing in SpiralApplet/README.html)
        <=> every left-right symmetric lagoon consisting of the verts of one polygon has at least two good exits
        <=> every lagoon has at least one good exit
                (easily true, since there's a good exit out of any vertex at maximum distance from center)
        WEAKEST (known true)
    or, to phrase the series in reverse:
        STRONGEST (known false)
        there's a lagoon with no good exit
                (easily false, since there's a good exit out of any vertex at maximum distance from center)
        <=> there's a left-right symmetric lagoon consisting of the verts of one polygon with at most one good exit
        <=> there's a lagoon consisting of the verts of one polygon with at most one good exit
        <=> false
         => there's a lagoon that's a subset of the verts of one polygon with at most one good exit
         => there's a convex lagoon with at most one good exit
         => there's a lagoon with at most one good exit  (NEGATION OF MAIN CONJECTURE)
         => true
        <=> there's a lagoon with only two good exits
                (example: one edge, with a bad exit out of each endpoint)

    Related statements (might fit into the above heirarchies somehow,
    or might be helpful in proving them):
        - for every polygon, for every point in interior, there's at least two
            verts at local maximum distance (FALSE)
        - for every polygon, there's at least two verts at local maximum distance from cc
            (UNKNOWN, implies every polygon has at least two good exits)
        ...
        - for every lagoon, there's at least two verts at local maximum dist from cc
            (UNKNOWN, implies previous)
        - add "all faces tangent to sphere" to any/all of above conjectures (weaker)
        - replace "lagoon/polygon" with "left-right symmetric lagoon/polygon" in any/all of above conjectures (weaker)
        - call an edge within a lagoon "reversible" or "doublegood" if it's good from the point of view
          of both sublagoons.  Various conjectures possible, e.g.
          - there's an unfolding such that, for every lagoon in it, every edge in the lagoon is reversible?
          - maybe even a stronger condition such as: every connected subgraph has the same property

-----------------------------------------------
Required reading:

2006 Brendan Lucier's master's dissertion "Unfolding and Reconstructing Polyhedra"
Aug 20 2007 Benton/O'Rourke "Unfolding Polyhedra via Cut-Tree Truncation" http://www.dremel.com/en-us/tools/Pages/ToolDetail.aspx?pid=1100-N%2f25
        also http://cs.smith.edu/~orourke/Papers/trunc.pdf
Feb 5, 2008 O'Rourke "Band Unfoldings and Prismatoids: A Counterexample" http://arxiv.org/pdf/0710.0811.pdf
Feb 4, 2008 Benton/O'Rourke "A Class of Convex Polyhedra with Few Edge Unfoldings" http://arxiv.org/pdf/0801.4019.pdf
        also shorter (?) version later, august: http://cs.smith.edu/~orourke/Papers/geodome.pdf
Sep 7 2008 Alex Benton's dissertation "Unfolding Polyhedra" http://bentonian.com/Papers/Dissertation/Dissertation.pdf
6 Oct 2008  Alexey S Tarasov "Existence of a polyhedron which does not have a non-overlapping pseudo-edge unfolding" http://arxiv.org/abs/0806.2360 http://arxiv.org/pdf/0806.2360v3.pdf
21 Apr 2014 Mohammad Ghomi "Affine unfoldings of convex polyhedra" http://arxiv.org/abs/1305.3231 http://arxiv.org/pdf/1305.3231v2.pdf
31 Jan 2015 Mohammad Ghomi "Affine unfoldings of convex polyhedra: progress on Durer’s problem¨ http://people.math.gatech.edu/~ghomi/Papers/durerPF.pdf

Q: I noticed a mistake in one of the papers by benton&orourke-- it claimed to have proved something
which isn't true.  Something about volcano unfoldings?  What was it?  Doesn't seem to be any of the ones above, so far.  Bleah!  Was it a paper that also said "this proves there's no proof of shephard's conjecture based on a random argument"

