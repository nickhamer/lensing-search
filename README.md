#
Collision detection code for use in microlensing simulation

## basic idea
Instead of iterating over timesteps we just compute the contour for source and lens trajectory, then find the times at which they
cross analytically. Takes hdf5 output from PopSyCLE and performs search. Edit line 585 to specify different hdf5 file.
Outputs details of each event to json.

Example hdf5 file can be found at https://drive.google.com/file/d/1dyVHS0E8rKYQTyzBAHWpgYUVdLl6T86Z/view?usp=sharing_eil_m&ts=65aebb68

## usage notes
The search field is small so the trig functions are all approximated as linear.
As in, latitude and longitude are coordinates in a cartesian 2d space, and it checks to see if stretched circles cross.
Those circles are be the einstein radius of the lens and the angular size of the source.
Some coordinate conversion is needed to apply the bodies' velocities before converting back to spherical, see code.

("ffp" and "pbh" are used interchangeably in the code, both labels just mean the list of lenses)
There are 2 equivalent functions:
- processPatch_quad_starTree
- processPatch_quad_ffpTree
the difference being which of the populations fills out the kdtree. The larger population can go in the tree (which is fast),
and the smaller population can be iterated over. It wouldn't be very much work to combine the functions into one and
dynamically choose which population goes into the tree. starTree puts sources in the kdtree 
(good when lots of sources in comparison to lenses), ffpTree puts lenses in kdtree (vice versa)

The distance to search in the tree around each iterated body, 'maxSphRadius', is set by hand based on maximum total displacement.
I think it's overkill to have it be so large, but originally it didn't matter. Ideally, this should be set automatically.

The function 'computeEvents' takes in a file containing both populations and processes each patch inside it.
This could be parallelized since each patch contains all the necessary information.

N.B. any hardcoded 1825 is the Roman survey duration in days.
