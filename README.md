#
some collision detection for use in microlensing simulation


## usage notes
The search field is so small that the trig functions are all approximated as linear.
As in, latitude and longitude are coordinates in a cartesian 2d space, and it checks to see if some circles cross.
Those circles would be the einstein radius of the lens and the angular size of the star.
Some coordinate conversion needed to apply the bodies' velocities before converting back to spherical.

(ffp and pbh are used interchangeably in the code, both are the possible lenses)
There are 2 equivalent functions:
- processPatch_quad_starTree
- processPatch_quad_ffpTree

the difference being which of the populations fills out the kdtree. The larger population can go in the tree (which is fast),
and the smaller population can be iterated over. It wouldn't be very much work to combine the functions into one and
dynamically choose which population goes into the tree.

the distance to search in the tree around each iterated body, 'maxSphRadius', is set by hand based on maximum total displacement.
I think it's overkill to have it be so large, but originally it didn't matter. You might lose some edge cases
by reducing it, but it could speed things up considerably to do so.


The function 'computeEvents' takes in a file containing both populations and processes each patch inside it.
This could be parallelized since each patch contains all the necessary information. I think in the popsycle
code they start combining patches in case some source body interacts with a lens in a neighboring patch, but this
seems like a ludicrous edge case so it's ignored here.

any hardcoded 1825 was the original duration in days.
