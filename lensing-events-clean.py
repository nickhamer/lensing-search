import numpy as np
import h5py
from scipy.spatial import KDTree
import time, re, json
from numpyencoder import NumpyEncoder
from mpmath import mp, mpf, nstr
from mpmath import isnan as mp_isnan
from astropy.table import Table

##########
# Conversions
##########
masyr_to_degday = 1.0 * (1.0e-3 / 3600.0) * (1.0 / 365.25)
kms_to_kpcs = 1.0 * (3.086 * 10**16) ** -1
kms_to_kpcday = 1.0 * (3.086 * 10**16) ** -1 * 86400.0
au_to_kpc = 4.848 * 10**-9

transit15minSq = mpf('3.25786e-11')

# set precision
mp.dps = 30

# region coordinate transformations

def GLat_exact(px, py, pz):
    return np.arcsin(pz / np.sqrt(px**2 + py**2 + pz**2))

def GLon_exact(px, py, pz):
    return np.arctan(py / px)

def spherical_exact(px, py, pz):
    return(
        np.sqrt(px**2 + py**2 + pz**2),
        GLat_exact(px, py, pz),
        GLon_exact(px, py, pz)
    )

def cartesian_exact(rad, glat, glon):
    return(
        rad * np.cos(glat * np.pi / 180) * np.cos(glon * np.pi / 180),        
        rad * np.cos(glat * np.pi / 180) * np.sin(glon * np.pi / 180),
        rad * np.sin(glat * np.pi / 180)
    )

#endregion

# this holds all the information about some circle with a linear trajectory
# returns (circle1, circle2, rectangle points)
def RRectPath(r, p1, v, t):
    p2 = p1 + v * t

    if (v[0] == 0 and v[1] == 0):
        return ((p1, r), (p1, r), (p1, p1, p1, p1))


    uPerp = np.array([-v[1], v[0]]) / np.sqrt(np.dot(v,v))

    rectPoints = (p1 + uPerp * r, p2 + uPerp * r, p2 - uPerp * r, p1 - uPerp * r)

    return ((p1, r), (p2, r), rectPoints)

# after writing many functions, had to start using precision numbers.
# this is a decorator to map all the arguments of a function f(x1,x2...) --> f(mpf(x1),mpf(x2)...)
def make_precise(func):
    def inner(*args):
        return(func(*map(mpf, args)))

    return inner

# the squared difference between  the two solutions of the quadratic equation
# formulas here were solved analytically in mathematica and copy/pasted
@make_precise
def tSolSqDiff(r1, r2, x1, y1, vx1, vy1, x2, y2, vx2, vy2):
    # -r1 -r2 + sqrt(...) < 0 condition
    if (np.power(r1 + r2,2) <= (np.power(vy1*x1 - vy2*x1 - vy1*x2 + vy2*x2 - vx1*y1 + vx2*y1 + vx1*y2 - 
        vx2*y2,2)/
        (np.power(vx1,2) - 2*vx1*vx2 + np.power(vx2,2) + np.power(vy1,2) - 2*vy1*vy2 + 
        np.power(vy2,2)))):
        # no bueno
        return(np.nan)
    

    return(4*(-4*r1*r2*vx1*vx2 - 4*r1*r2*vy1*vy2 - 4*vy1*vy2*x1*x2 + 2*vx1*vy1*x1*y1 - 2*vx2*vy1*x1*y1 - 2*vx1*vy2*x1*y1 + 
     2*vx2*vy2*x1*y1 - 2*vx1*vy1*x2*y1 + 2*vx2*vy1*x2*y1 + 2*vx1*vy2*x2*y1 - 2*vx2*vy2*x2*y1 - 2*vx1*vy1*x1*y2 + 
     2*vx2*vy1*x1*y2 + 2*vx1*vy2*x1*y2 - 2*vx2*vy2*x1*y2 + 2*vx1*vy1*x2*y2 - 2*vx2*vy1*x2*y2 - 2*vx1*vy2*x2*y2 + 
     2*vx2*vy2*x2*y2 - 4*vx1*vx2*y1*y2 - 2*vx1*vx2*np.power(r1,2) - 2*vy1*vy2*np.power(r1,2) - 2*vx1*vx2*np.power(r2,2) - 
     2*vy1*vy2*np.power(r2,2) + 2*r1*r2*np.power(vx1,2) + 2*y1*y2*np.power(vx1,2) + np.power(r1,2)*np.power(vx1,2) + 
     np.power(r2,2)*np.power(vx1,2) + 2*r1*r2*np.power(vx2,2) + 2*y1*y2*np.power(vx2,2) + 
     np.power(r1,2)*np.power(vx2,2) + np.power(r2,2)*np.power(vx2,2) + 2*r1*r2*np.power(vy1,2) + 
     2*x1*x2*np.power(vy1,2) + np.power(r1,2)*np.power(vy1,2) + np.power(r2,2)*np.power(vy1,2) + 
     2*r1*r2*np.power(vy2,2) + 2*x1*x2*np.power(vy2,2) + np.power(r1,2)*np.power(vy2,2) + 
     np.power(r2,2)*np.power(vy2,2) + 2*vy1*vy2*np.power(x1,2) - np.power(vy1,2)*np.power(x1,2) - 
     np.power(vy2,2)*np.power(x1,2) + 2*vy1*vy2*np.power(x2,2) - np.power(vy1,2)*np.power(x2,2) - 
     np.power(vy2,2)*np.power(x2,2) + 2*vx1*vx2*np.power(y1,2) - np.power(vx1,2)*np.power(y1,2) - 
     np.power(vx2,2)*np.power(y1,2) + 2*vx1*vx2*np.power(y2,2) - np.power(vx1,2)*np.power(y2,2) - 
     np.power(vx2,2)*np.power(y2,2))*np.power(-2*vx1*vx2 - 2*vy1*vy2 + np.power(vx1,2) + np.power(vx2,2) + 
     np.power(vy1,2) + np.power(vy2,2),-2))

# solve for the value of t when the circles are exactly touching
@make_precise
def tSol(solutionIndex, r1, r2, x1, y1, vx1, vy1, x2, y2, vx2, vy2):

    if (solutionIndex == 1):
        sqrtSign = -1
    elif(solutionIndex == 2):
        sqrtSign = 1
    else:
        raise Exception('skill issue')

    # check condition for solution existing

    cond1 = r1 + r2 + np.sqrt(np.power(vy1*x1 - vy2*x1 - vy1*x2 + vy2*x2 - vx1*y1 + vx2*y1 + vx1*y2 - 
        vx2*y2,2)/
        (np.power(vx1,2) - 2*vx1*vx2 + np.power(vx2,2) + np.power(vy1,2) - 2*vy1*vy2 + 
        np.power(vy2,2)))

    cond2 = -r1 - r2 + np.sqrt(np.power(vy1*x1 - vy2*x1 - vy1*x2 + vy2*x2 - vx1*y1 + vx2*y1 + vx1*y2 - 
        vx2*y2,2)/
        (np.power(vx1,2) - 2*vx1*vx2 + np.power(vx2,2) + np.power(vy1,2) - 2*vy1*vy2 + 
        np.power(vy2,2)))

    if (cond1 > 0 and cond2 > 0):
        return(np.nan)
        

    return(-2*vx1*x1 + 2*vx2*x1 + 2*vx1*x2 - 
		2*vx2*x2 - 2*vy1*y1 + 
		2*vy2*y1 + 2*vy1*y2 - 
		2*vy2*y2 + 
		sqrtSign * np.sqrt(np.power(2*vx1*x1 - 
		2*vx2*x1 - 2*vx1*x2 + 
		2*vx2*x2 + 2*vy1*y1 - 
		2*vy2*y1 - 2*vy1*y2 + 
		2*vy2*y2,2) - 
		4*(np.power(vx1,2) - 
		2*vx1*vx2 + 
		np.power(vx2,2) + 
		np.power(vy1,2) - 
		2*vy1*vy2 + np.power(vy2,2))*
		(-np.power(r1,2) - 2*r1*r2 - 
		np.power(r2,2) + 
		np.power(x1,2) - 2*x1*x2 + 
		np.power(x2,2) + 
		np.power(y1,2) - 2*y1*y2 + 
		np.power(y2,2))))/(
		2.*(np.power(vx1,2) - 2*vx1*vx2 + 
		np.power(vx2,2) + 
		np.power(vy1,2) - 2*vy1*vy2 + 
		np.power(vy2,2))
	)

# solve for the times t1,t2 where two circles overlap given two RRectPaths
# only done after comparing the square differences
def rrQuadSolve(rr1, rr2):

	d1 = rr1[1][0] - rr1[0][0]
	d2 = rr2[1][0] - rr2[0][0]

	t1 = tSol(1, rr1[0][1], rr2[0][1], *rr1[0][0], *d1, *rr2[0][0], *d2)

	t2 = tSol(2, rr1[0][1], rr2[0][1],
		*rr1[0][0], *d1,
		*rr2[0][0], *d2
	)
	return((t1, t2))


# xxx: does this implement a t = -T/2 start?
def rrQuadDiff(rr1, rr2):

	d1 = rr1[1][0] - rr1[0][0]
	d2 = rr2[1][0] - rr2[0][0]
        
	return(tSolSqDiff(
        rr1[0][1], rr2[0][1],
		*rr1[0][0], *d1,
		*rr2[0][0], *d2
	))

# pretend all stars have the radius of the sun
# returns angular size in radians
solar_radius_kpc = 2.26E-11
def star_size(rad):
    return solar_radius_kpc / rad

# mass in solar masses, distances in kpc (all galaxia defaults)
# returns angular in radians
def einstein_radius(d_lens, d_source, mass):
    # https://en.wikipedia.org/wiki/Einstein_radius

    d_LS = np.abs(d_source - d_lens)

    theta_E_arcsec = np.sqrt(mass/np.power(10, 11.09)) / np.sqrt(d_lens * d_source / (d_LS * 1E6))

    if(np.isnan(theta_E_arcsec)):
        print(d_LS)
        raise Exception('fucky wucky')

    return(np.pi/180 * theta_E_arcsec/3600)

# take a patch as made in popsycle, make the kdtree from the stars, iterate over the ffps
def processPatch_quad_starTree(patch, duration):

    stars = patch[np.where(patch['rem_id'] == 0)]
    ffps  = patch[np.where(patch['rem_id'] == 104)] # 104 for pbhs 105 for ffps

    print('num stars: %d, ffps: %d' % (len(stars), len(ffps)))
    
    def end_movement_spherical_noCartesian(pVec, vVec):
        return spherical_exact(
            cartesian_exact(pVec['rad'], pVec['glat'], pVec['glon'])[0] + vVec['vx'] * kms_to_kpcday * duration,
            cartesian_exact(pVec['rad'], pVec['glat'], pVec['glon'])[1] + vVec['vy'] * kms_to_kpcday * duration,
            cartesian_exact(pVec['rad'], pVec['glat'], pVec['glon'])[2] + vVec['vz'] * kms_to_kpcday * duration
        )

    endPosSph = np.asarray(end_movement_spherical_noCartesian(stars[['rad','glat','glon']], stars[['vx', 'vy', 'vz']])).T
    endPosSph_ffps = np.asarray(end_movement_spherical_noCartesian(ffps[['rad','glat','glon']], ffps[['vx', 'vy', 'vz']])).T

    def to_radians(rgg):
        return (rgg['rad'], rgg['glat']*np.pi/180, rgg['glon']*np.pi/180)

    sphCoords = np.asarray(to_radians(stars[['rad','glat','glon']])).T
    sphCoords_ffps = np.asarray(to_radians(ffps[['rad','glat','glon']])).T
    

    print('==== max displacements ====')    
    print('stars: %s' % np.sqrt(np.max((sphCoords - endPosSph)[:, 1]**2 + (sphCoords - endPosSph)[:, 2]**2)))

    print('stars: %s' % np.sqrt(np.max((sphCoords_ffps - endPosSph_ffps)[:, 1]**2 + (sphCoords_ffps - endPosSph_ffps)[:, 2]**2)))
    print('use these numbers to set the spherical search radius in the kdtree')
    print('============================')
    



    maxSphRadius = 1e-6
    kdt = KDTree(sphCoords[:, 1:3])
    print('number of guys in the kdtree: %d' % len(sphCoords[:, 1:3]))

    startTime = time.time()
    totalCloseOnes = 0
    lonelyBoys = 0
    totalLensingEvents = 0

    resultData = {'transitData': []}

    for i, ffp in enumerate(ffps):
        # if (i%1000 == 0):
        #     print('i at %d' % i)

        results = kdt.query_ball_point( (ffp['glat']*np.pi/180, ffp['glon']*np.pi/180), maxSphRadius)

        
        
        for res in results:
            # res is the id in the original list of the star
            if(ffp['rad'] < sphCoords[res][0]):
                # lens is nearer than the source
                totalCloseOnes += 1

                # starting coordiantes
                ffpCoord  = (ffp['glat']*np.pi/180, ffp['glon']*np.pi/180)
                starCoord = sphCoords[res][1:3]

                ffpCoord_end  = endPosSph_ffps[i, 1:3]
                starCoord_end = endPosSph[res][1:3]

                d_ffp =  ffpCoord_end - ffpCoord
                d_star = starCoord_end - starCoord

                # this is probably here since the time in popsycle might go from t=-1/2 to t=1/2
                # so for purposes of comparing our output to theirs, make this the same.
                ffpCoord = ffpCoord - d_ffp/2
                starCoord = starCoord - d_star/2


                rr_ffp = RRectPath(
                    einstein_radius(ffp['rad'], sphCoords[res][0], ffp['mass']),
                    np.array(ffpCoord),
                    np.array(d_ffp),
                1)

                rr_star = RRectPath(
                    star_size(sphCoords[res][0]),
                    np.array(starCoord),
                    np.array(d_star),
                1)

                deltaTSq = rrQuadDiff(rr_ffp, rr_star)

                # some better code would remove the need for this try catch
                # but it works fine, so i'm leaving it in
                try:
                    if(np.isnan(deltaTSq)):
                        # no collision
                        continue
                except:
                    # chill
                    pass

                if (deltaTSq < transit15minSq):
                    # too short
                    continue
                
                # possible collision, now solve for the exact times where the circles are touching.
                (t1, t2) = rrQuadSolve(rr_ffp, rr_star)

                if((t1 >= 0 and t1 <= 1) or (t2 >= 0 and t2 <= 1)):

                    print('comparing delta t:   ',t2 - t1, np.sqrt(deltaTSq))

                    totalLensingEvents +=1
                    '''
                    print('======== lensing event ========')
                    print('start/end time of transit: %s, %s' % (t1, t2))
                    #print('source id: %s' % stars[res]['obj_id'])
                    #print('lens id: %s' % ffp['obj_id'])
                    #print('ffp index: %s' % i)
                    #print('star index: %s' % res)
                    print('einstein radius: %s' % einstein_radius(ffp['rad'], sphCoords[res][0], ffp['mass']))
                    print('star radius: %s' % star_size(sphCoords[res][0]))
                    print('star coord: %s' % starCoord)
                    print('dStar: %s' % d_star)
                    '''
                    
                    times_np = np.array(mp.matrix([t1,t2]).tolist(), dtype=np.float32)

                    resultData['transitData'].append({
                        'id_ffp': i,
                        'id_star': res,
                        'star obj_id': stars[res]['obj_id'],
                        'ffp obj_id': ffp['obj_id'],
                        't1': times_np[0],
                        't2': times_np[1],
                        'm_ffp': ffp['mass'],
                        'm_star': stars[res]['mass']
                    })

                    if(resultData['transitData'][-1]['id_star'] > len(stars)):
                        print('=!==!==!==!==!==!==!==!==!==!=')
                        print(resultData['transitData'][-1])
                        raise Exception('somehow star id larger than # of stars')

        if len(results) == 0:
            lonelyBoys += 1

    endTime = time.time()
    print('search radius is %s rad' % maxSphRadius)
    print('did search for transits in %s s' % (endTime-startTime))
    print('total close stars found: %d. total lonely ffps found: %d' % (totalCloseOnes, lonelyBoys))
    print('total lensing events: %s' % totalLensingEvents)

    resultData['starCoordsStart'] = sphCoords[ [tr['id_star'] for tr in resultData['transitData']] ]
    resultData['starCoordsEnd'] = endPosSph[ [tr['id_star'] for tr in resultData['transitData']] ]

    resultData['ffpCoordsStart'] = sphCoords_ffps[ [tr['id_ffp'] for tr in resultData['transitData']] ]
    resultData['ffpCoordsEnd'] = endPosSph_ffps[ [tr['id_ffp'] for tr in resultData['transitData']] ]

    return(resultData)

def processPatch_quad_ffpTree(patch, duration):

    stars = patch[np.where(patch['rem_id'] == 0)]
    ffps  = patch[np.where(patch['rem_id'] == 104)] # 105 for ffps

    print('num stars: %d, ffps: %d' % (len(stars), len(ffps)))
    
    def end_movement_spherical_noCartesian(pVec, vVec):
        return spherical_exact(
            cartesian_exact(pVec['rad'], pVec['glat'], pVec['glon'])[0] + vVec['vx'] * kms_to_kpcday * duration,
            cartesian_exact(pVec['rad'], pVec['glat'], pVec['glon'])[1] + vVec['vy'] * kms_to_kpcday * duration,
            cartesian_exact(pVec['rad'], pVec['glat'], pVec['glon'])[2] + vVec['vz'] * kms_to_kpcday * duration
        )

    endPosSph = np.asarray(end_movement_spherical_noCartesian(stars[['rad','glat','glon']], stars[['vx', 'vy', 'vz']])).T
    endPosSph_ffps = np.asarray(end_movement_spherical_noCartesian(ffps[['rad','glat','glon']], ffps[['vx', 'vy', 'vz']])).T

    def to_radians(rgg):
        return (rgg['rad'], rgg['glat']*np.pi/180, rgg['glon']*np.pi/180)

    sphCoords = np.asarray(to_radians(stars[['rad','glat','glon']])).T
    sphCoords_ffps = np.asarray(to_radians(ffps[['rad','glat','glon']])).T
    
    print('==== max displacement ====')    
    print('stars: %s' % np.sqrt(np.max((sphCoords - endPosSph)[:, 1]**2 + (sphCoords - endPosSph)[:, 2]**2)))

    print('stars: %s' % np.sqrt(np.max((sphCoords_ffps - endPosSph_ffps)[:, 1]**2 + (sphCoords_ffps - endPosSph_ffps)[:, 2]**2)))
    print('use these numbers to set the spherical search radius in the kdtree')
    print('============================')
    



    maxSphRadius = 1e-6
    kdt = KDTree(sphCoords_ffps[:, 1:3])
    print('number of guys in the kdtree: %d' % len(sphCoords_ffps[:, 1:3]))

    startTime = time.time()
    totalCloseOnes = 0
    lonelyBoys = 0
    totalLensingEvents = 0

    resultData = {'transitData': []}

    for i, star in enumerate(stars):
        # if (i%1000 == 0):
        #     print('i at %d' % i)
        '''
        print('star coords from the list')
        print((star['glat']*np.pi/180, star['glon']*np.pi/180))

        print('star coords from sphCoords')
        print(sphCoords[i,1:3])

        raise Exception('done')
        '''
        results = kdt.query_ball_point(sphCoords[i,1:3], maxSphRadius)

        
        
        for res in results:
            # res is the id in the original list of the ffps
            if(star['rad'] > sphCoords_ffps[res][0]):
                # lens is nearer than the source
                totalCloseOnes += 1

                # starting coordiantes
                ffpCoord  = sphCoords_ffps[res, 1:3]
                ffpCoord_end  = endPosSph_ffps[res, 1:3]

                starCoord = sphCoords[i, 1:3]                
                starCoord_end = endPosSph[i, 1:3]

                d_ffp =  ffpCoord_end - ffpCoord
                d_star = starCoord_end - starCoord

                ffpCoord = ffpCoord - d_ffp/2
                starCoord = starCoord - d_star/2


                rr_ffp = RRectPath(
                    einstein_radius(sphCoords_ffps[res, 0], sphCoords[i, 0], ffps[res]['mass']),
                    np.array(ffpCoord),
                    np.array(d_ffp),
                1)

                rr_star = RRectPath(
                    star_size(sphCoords[i][0]),
                    np.array(starCoord),
                    np.array(d_star),
                1)

                deltaTSq = rrQuadDiff(rr_ffp, rr_star)

                if(mp_isnan(deltaTSq)):
                        # no collision
                        continue

                try:
                    if(np.isnan(deltaTSq)):
                        # no collision
                        continue
                except:
                    # chill
                    pass

                if (deltaTSq < transit15minSq):
                    # too short
                    continue
                
                (t1, t2) = rrQuadSolve(rr_ffp, rr_star)

                try:
                    if((t1 >= 0 and t1 <= 1) or (t2 >= 0 and t2 <= 1)):
                        print('hit')
                except TypeError as e:
                    print('this shouldn\'t happen')
                    print(e, t1, t2, deltaTSq)
                    continue
                

                if(t1 >= 0 and t2 <= 1):

                    print('comparing delta t:   ',t2 - t1, np.sqrt(deltaTSq))

                    totalLensingEvents +=1
                    '''
                    print('======== lensing event ========')
                    print('start/end time of transit: %s, %s' % (t1, t2))
                    #print('source id: %s' % stars[res]['obj_id'])
                    #print('lens id: %s' % ffp['obj_id'])
                    #print('ffp index: %s' % i)
                    #print('star index: %s' % res)
                    print('einstein radius: %s' % einstein_radius(ffp['rad'], sphCoords[res][0], ffp['mass']))
                    print('star radius: %s' % star_size(sphCoords[res][0]))
                    print('star coord: %s' % starCoord)
                    print('dStar: %s' % d_star)
                    '''
                    
                    times_np = np.array(mp.matrix([t1,t2]).tolist(), dtype=np.float32)

                    resultData['transitData'].append({
                        'id_ffp': res,
                        'id_star': i,
                        'star obj_id': star['obj_id'],
                        'ffp obj_id': ffps[res]['obj_id'],
                        't1': times_np[0],
                        't2': times_np[1],
                        'm_ffp': ffps[res]['mass'],
                        'm_star': star['mass']
                    })

                    if(resultData['transitData'][-1]['id_star'] > len(stars)):
                        print('=!==!==!==!==!==!==!==!==!==!=')
                        print(resultData['transitData'][-1])
                        raise Exception('somehow star id larger than # of stars')

        if len(results) == 0:
            lonelyBoys += 1

    endTime = time.time()
    print('search radius is %s rad' % maxSphRadius)
    print('did search for transits in %s s' % (endTime-startTime))
    print('total close stars found: %d. total lonely ffps found: %d' % (totalCloseOnes, lonelyBoys))
    print('total lensing events: %s' % totalLensingEvents)

    resultData['starCoordsStart'] = sphCoords[ [tr['id_star'] for tr in resultData['transitData']] ]
    resultData['starCoordsEnd'] = endPosSph[ [tr['id_star'] for tr in resultData['transitData']] ]

    resultData['ffpCoordsStart'] = sphCoords_ffps[ [tr['id_ffp'] for tr in resultData['transitData']] ]
    resultData['ffpCoordsEnd'] = endPosSph_ffps[ [tr['id_ffp'] for tr in resultData['transitData']] ]

    return(resultData)

def computeEvents(hdf5_filename):
    hf = h5py.File(hdf5_filename, "r")

    binRE = re.compile(r'l(\d{1,2})b(\d{1,2})')
    patchKeys = list(filter(binRE.match, hf.keys()))
    print('patches in this file are %s' % patchKeys)

    results = {}
    for k in [patchKeys[0]]:
        print('processing patch %s' % k)
        patchResult = processPatch_quad_ffpTree(hf[k], 1825)
        results[k] = patchResult
    
    jsonFilename = '%s-mk2.json' % hdf5_filename[:-3]
    print('writing output to file: %s' % jsonFilename)

    with open(jsonFilename, 'w') as f:
        json.dump(results, f, cls=NumpyEncoder)

computeEvents('pbh-binl9b9_earth_01_fdm_5_long.h5')
#computeEvents('pbh-binl2b2.h5')


# not really sure what this was for. possibly for running this code on
# the output of popsycle's lensing search to see if it finds the same.
def convertFits(fn, relevantIdx=104):
    duration = 1825
    
    t = Table.read(fn)

    goodIdx = np.where(t['rem_id_L'] == relevantIdx)[0]
    goodRows = t[goodIdx]
    
    def end_movement_spherical_noCartesian(pVec, vVec, suffix):
        def s(str):
            return(str+suffix)

        return spherical_exact(
            cartesian_exact(pVec[s('rad')], pVec[s('glat')], pVec[s('glon')])[0] + vVec[s('vx')] * kms_to_kpcday * duration,
            cartesian_exact(pVec[s('rad')], pVec[s('glat')], pVec[s('glon')])[1] + vVec[s('vy')] * kms_to_kpcday * duration,
            cartesian_exact(pVec[s('rad')], pVec[s('glat')], pVec[s('glon')])[2] + vVec[s('vz')] * kms_to_kpcday * duration
        )

    endPosSph = np.asarray(end_movement_spherical_noCartesian(goodRows[['rad_S','glat_S','glon_S']], goodRows[['vx_S', 'vy_S', 'vz_S']], '_S')).T
    endPosSph_ffps = np.asarray(end_movement_spherical_noCartesian(goodRows[['rad_L','glat_L','glon_L']], goodRows[['vx_L', 'vy_L', 'vz_L']], '_L')).T

    def to_radians(rgg, suffix):
        return (rgg['rad%s' % suffix], rgg['glat%s' % suffix]*np.pi/180, rgg['glon%s' % suffix]*np.pi/180)

    sphCoords = np.asarray(to_radians(goodRows[['rad_S','glat_S','glon_S']], '_S')).T
    sphCoords_ffps = np.asarray(to_radians(goodRows[['rad_L','glat_L','glon_L']], '_L')).T

    totalEvents = 0
    resultData = {
        'transitData': [],
        'good lens obj_ids': [],
        'pop t_E values': [],        
        'my t_E values': []
        }

    for i in range(len(goodRows)):
        # starting coordiantes
        ffpCoord  = sphCoords_ffps[i, 1:3]
        ffpCoord_end  = endPosSph_ffps[i, 1:3]

        starCoord = sphCoords[i, 1:3]                
        starCoord_end = endPosSph[i, 1:3]

        d_ffp =  ffpCoord_end - ffpCoord
        d_star = starCoord_end - starCoord

        ffpCoord = ffpCoord - d_ffp/2
        starCoord = starCoord - d_star/2

        #d_lens, d_source, mass

        rr_ffp = RRectPath(
            einstein_radius(sphCoords_ffps[i, 0], sphCoords[i, 0], goodRows[i]['mass_L']),
            np.array(ffpCoord),
            np.array(d_ffp),
        1)

        rr_star = RRectPath(
            star_size(sphCoords[i][0]),
            np.array(starCoord),
            np.array(d_star),
        1)

        deltaTSq = rrQuadDiff(rr_ffp, rr_star)

        resultData['transitData'].append({
            'star obj_id': goodRows[i]['obj_id_S'],
            'ffp obj_id': goodRows[i]['obj_id_L'],
            'm_ffp': goodRows[i]['mass_L'],
            'm_star': goodRows[i]['mass_S'],
            'deltaTSq': nstr(deltaTSq, n=13)
        })

        resultData['starCoordsStart'] = sphCoords
        resultData['starCoordsEnd'] = endPosSph

        resultData['ffpCoordsStart'] = sphCoords_ffps
        resultData['ffpCoordsEnd'] = endPosSph_ffps


        if(mp_isnan(deltaTSq)):
            # no collision

            #print('nan -- radial distances L,S %s %s' % (sphCoords_ffps[i, 0],sphCoords[i, 0]))
            #print('nan -- einsten radius %s' % einstein_radius(sphCoords_ffps[i, 0], sphCoords[i, 0], goodRows[i]['mass_L']))
            #print('------------')
            continue

        vRel = (d_ffp - d_star) / duration
        speedRel = np.sqrt(vRel[0]**2 + vRel[1]**2)
        t_E = einstein_radius(sphCoords_ffps[i, 0], sphCoords[i, 0], goodRows[i]['mass_L']) / speedRel

        resultData['good lens obj_ids'].append(goodRows[i]['obj_id_L'])
        resultData['pop t_E values'].append(goodRows[i]['t_E'])
        resultData['my t_E values'].append(t_E)

        if (deltaTSq < transit15minSq):
            # too short
            continue
        
        (t1, t2) = rrQuadSolve(rr_ffp, rr_star)

        if((t1 >= 0 and t1 <= 1) or (t2 >= 0 and t2 <= 1)):
            print(1825*(t2-t1))
            totalEvents += 1

        
    print('total event count: %d' % totalEvents)

    jsonFilename = 'fits-data%s.json' % fn[:-5]
    print('writing output to file: %s' % jsonFilename)
    
    with open(jsonFilename, 'w') as f:
        json.dump(resultData, f, cls=NumpyEncoder)

#convertFits('30_mass_small.fits')