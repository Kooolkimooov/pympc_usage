#!/usr/bin/env python
# author: Vincent Hugel @ COSMER | UTLN

import matplotlib.pyplot as plt
import numpy as np
from math import sinh
import scipy.optimize as opt


def catenary( x, C ):
	y = (1.0 / C) * (np.cosh( C * x ) - 1.0)
	return y


def ma_fonction( xC, L, DH, dxAB ):
	y = xC * xC * (L * L - DH * DH) - 2.0 * (-1.0 + np.cosh( xC * dxAB ))
	return y


def funcs( a, L, DH, dxAB ):
	f = pow( L, 2 ) - pow( DH, 2 ) - pow( 2 * a * sinh( dxAB / (2 * a) ), 2 )
	return f


def funcc( a, L, DH, dxAB ):
	f = pow( L, 2 ) - pow( DH, 2 ) - pow( 2 * a * np.cosh( dxAB / (2 * a) ), 2 )
	return f


def ma_derivee( xC, L, DH, dxAB ):
	y = 2.0 * xC * (L * L - DH * DH) - 2.0 * dxAB * sinh( xC * dxAB )
	return y


def ma_double_derivee( xC, L, DH, dxAB ):
	y = 2.0 * (L * L - DH * DH) - 2.0 * dxAB ** 2 * np.cosh( xC * dxAB )
	return y


def TraceDroiteVerticale( x1, y1, ma_couleur ):
	tab_x = [ x1, x1 ]
	tab_y = [ 0, y1 ]
	plt.plot( tab_x, tab_y, color = ma_couleur )


def TraceDroite( x1, x2, a, b, ma_couleur ):
	tab_x = np.linspace( x1, x2, 100 )
	tab_y = a * tab_x + b
	plt.plot( tab_x, tab_y, color = ma_couleur )


def TraceTangeante( x1, x2, derive_en_x1, y1, ma_couleur ):
	a = derive_en_x1
	b = -derive_en_x1 * x1 + y1
	TraceDroite( x1, x2, a, b, ma_couleur )


def TraceFonction( xmax, ymax, L, DH, dxAB ):
	print( 'Trace fonction' )
	tab_x = np.linspace( 0, xmax, 5000 )
	tab_y = ma_fonction( tab_x, L, DH, dxAB )

	plt.figure()
	plt.plot( tab_x, tab_y, color = 'red' )
	plt.ylim( -ymax, ymax )
	plt.grid()


def AfficherTexte( x, n ):
	texte = 'x%s=%f' % (str( n ), x)
	plt.text( x, 0, texte )


def NewtonsMethod( x, L, DH, dxAB, epsilon = 0.000001 ):
	dif = 2 * epsilon
	while dif > epsilon:
		y = ma_fonction( x, L, DH, dxAB )
		y_prime = ma_derivee( x, L, DH, dxAB )
		xn = x - y / y_prime
		dif = abs( xn - x )

		# TraceDroiteVerticale(x, y, 'green')
		# TraceTangeante(x, xn, y_prime, y, 'green')
		x = xn

	return x


def NewtonsMethodForDeriv( x, L, DH, dxAB, epsilon = 0.000001 ):
	dif = 2 * epsilon
	while dif > epsilon:
		y = ma_derivee( x, L, DH, dxAB )
		y_prime = ma_double_derivee( x, L, DH, dxAB )
		xn = x - y / y_prime
		dif = abs( xn - x )
		x = xn

	return x


def get_catenary_param( DH, dxAB, L ):
	# Eq. of catenary : 	y = (1/C)*(cosh(C*x) - 1) with origin at lowest point
	# Newton to get the initial guess
	# val = 1 / dxAB * (L ** 2 - DH ** 2) / (dxAB ** 2)
	# if val > 1:
	# 	x_init = 2.0 * np.arccosh( 1 / dxAB * (L ** 2 - DH ** 2) / (dxAB ** 2) )
	# else:
	# 	x_init = 2.0
	#
	# (x_deriv_zero, niter1) = NewtonsMethodForDeriv( x_init, L, DH, dxAB, eps )
	# # print('x_deriv_zero = %f_gamma' % x_deriv_zero)
	# x_init = 2.0 * x_deriv_zero
	#
	# xmax = 2.0 * x_init
	# ymax = 1.5 * np.max(
	# 		[ np.abs( ma_fonction( x_deriv_zero, L, DH, dxAB ) ),
	# 			np.abs( ma_fonction( x_init, L, DH, dxAB ) ) ]
	# 		)
	# TraceFonction(xmax,ymax,L,DH,dxAB)

	# (C, niter2) = NewtonsMethod(x_init, L, DH, dxAB, eps)

	a = opt.brentq(
			funcs, -.01, 100., args = (L, DH, dxAB)
			)

	C = 1 / a

	# H is the positive solution of a 2nd degree equation A.H^2+B.H+C = 0 - see Juliette's doc
	interd = L ** 2 - DH ** 2
	A_eq = -4.0 * (C ** 2) * interd
	B_eq = -4.0 * C * DH * (C * (interd) - 2 * DH) - 8.0 * (L ** 2) * C
	C_eq = (C * interd - 2.0 * DH) ** 2
	H = (-B_eq - np.sqrt( B_eq ** 2 - 4.0 * A_eq * C_eq )) / (2.0 * A_eq)
	# print('H=', H)
	D = 1.0 / C * np.arccosh( C * H + 1.0 )
	# return parameter of catenary and nb of iterations
	return C, D, H


# get coord. points of markers along the cable
# inputs : coord. of 1st extremety point A, coord. of 2nd extremity point B, length L of cable,
# distance d between
# markers along the cable
# outputs : array of coord of marker points, and number of marker points, 1st and 2nd exty points
# included.
# N.B.: assumption => the z coordinate is vertical, parallel to gravitation field.
def get_coor_marker_points_ideal_catenary( xA, yA, zA, xB, yB, zB, L, d, d0 = 0., nmax = None ):
	dxAB = np.sqrt( (xB - xA) ** 2 + (yB - yA) ** 2 )
	DH = zB - zA
	# get parameter of catenary cable
	C, D, H = get_catenary_param( DH, dxAB, L )
	# get total number of points
	nbpts = 2 if d0 != 0. else 1
	l = d0
	while l < L:  # to prevent missing point from numerical error
		nbpts += 1
		l += d
	if nmax is not None and nbpts > nmax:
		nbpts = nmax
	tabcoord = np.zeros( (nbpts, 3) )
	# curvilinear abscissae
	s = 0.
	for i in range( nbpts - 1 ):
		# get x,z coord of points in catenary frame, centered at 1st point
		inter = C * s - np.sinh( C * D )
		x = 1.0 / C * np.arcsinh( inter ) + D
		z = 1.0 / C * (np.sqrt( 1.0 + inter ** 2 ) - 1.0) - H
		tabcoord[ i ][ 0 ] = xA + x * (xB - xA) / dxAB
		tabcoord[ i ][ 1 ] = yA + x * (yB - yA) / dxAB
		tabcoord[ i ][ 2 ] = zA + z
		s += d0 if i == 0 and d0 != 0. else d

	tabcoord[ nbpts - 1 ] = np.array( [ xB, yB, zB ] )

	return tabcoord, C, D, H


eps = 0.001
if __name__ == "__main__":
	# -----------------------------------------------------------------------------
	# Programme principal
	# -----------------------------------------------------------------------------

	# x     = float(input("Quelle est la valeur de depart ? : "))

	# parameters to define the catenary
	# cable length
	L = 3.0
	print( 'Linput =', L )
	# height of catenary with respect to first point A
	fracH_L = 0.35
	H = fracH_L * L
	print( 'Hinput =', H )
	# DH, differnece of height between attachments points
	fracDH_L = 0.1
	dH = -fracDH_L * L
	print( 'DHinput =', dH )

	# checks:
	if (H + dH < 0):
		print( 'H+DH<0, exiting' )
		exit()
	if (L < np.abs( dH ) or L < (2 * H + dH)):
		print( 'L<max(|DH|,2H+DH), exiting' )
		exit()

	# catenary coefficient
	C = 2 * (2 * H + dH + 2 * L * np.sqrt( H * (H + dH) / (L ** 2 - dH ** 2) )) / (
				L ** 2 - (2 * H + dH) ** 2)

	# coord. of extremity points A and B
	D = 1 / C * np.arccosh( C * H + 1 )
	DD = 1 / C * np.arccosh( C * (H + dH) + 1 ) - D

	xA = -D
	xB = D + DD
	yA = H
	yB = H + dH
	dH = yB - yA
	dxAB = xB - xA
	# print('dxAB=%f_gamma, DH=%f_gamma' %(dxAB,DH))
	# print('L**2-DH**2 = %f_gamma' %(L**2-DH**2))
	# print('1/dxAB*(L**2-DH**2)/(dxAB**2) = %f_gamma' %(1/dxAB*(L**2-DH**2)/(dxAB**2)))

	# OBJECTIVE: determine C with inputs L, DH, and dxAB, which is to be used with coordinates of
	# motion tracking
	# system.
	C_, D, H, n = get_catenary_param( dH, dxAB, L )

	# display results
	print( 'C=%f n=%d (C input = %f)' % (C_, n, C) )
	print( 'checking:' )
	print( 'ma_fonction(x,L,DH,dxAB) = %f_gamma', ma_fonction( C_, L, dH, dxAB ) )
	print( 'ma_fonction(C,L,DH,dxAB) = %f_gamma', ma_fonction( C, L, dH, dxAB ) )

	# to be done:
	# extract coordinates of all points in initial coord frame, given coord of 2 extremity points,
	# assuming z is
	# vertical
	# inputs: 3D coordinates of 2 extremity points of catenary, length of catenary, and distance
	# between markers
	# outputs: array of 3D coordinates of all points of catenary, separated by d cm.

	xA = 0.0
	yA = 0.0
	zA = 0.0
	xB = xA + dxAB
	yB = 0.0
	zB = dH
	d = 0.2

	#############################################
	tab, n = get_coor_marker_points_ideal_catenary( xA, yA, zA, xB, yB, zB, L, d )
	print( n )
	#############################################

	# check with print and display
	print( tab )
	fig = plt.figure()
	ax = plt.axes( projection = '3d' )
	xs = tab[ :, 0 ]
	ys = tab[ :, 1 ]
	zs = tab[ :, 2 ]
	ax.plot3D( xs, ys, zs, marker = 'o' )

	ax.set_xlabel( 'X Label' )
	ax.set_ylabel( 'Y Label' )
	ax.set_zlabel( 'Z Label' )

	# check distances...
	nb = np.size( xs )
	ux = xs[ nb - 1 ] - xs[ 0 ]
	uy = ys[ nb - 1 ] - ys[ 0 ]
	lg_vec = np.sqrt( ux ** 2 + uy ** 2 )
	ux = ux / lg_vec
	uy = uy / lg_vec

	# check distances between markers
	for i in range( nb - 1 ):
		print( np.sqrt( (xs[ i + 1 ] - xs[ i ]) ** 2 + (zs[ i + 1 ] - zs[ i ]) ** 2 ) )
		# get distance between markers for check
		x_u = ux * (xs[ i ] - xs[ 0 ]) + uy * (ys[ i ] - ys[ 0 ])
		x_u_plus = ux * (xs[ i + 1 ] - xs[ 0 ]) + uy * (ys[ i + 1 ] - ys[ 0 ])
		dm = 1.0 / C * (np.sinh( C * (x_u_plus - D) ) - np.sinh( C * (x_u - D) ))
		print( 'dist=', dm )

	plt.show()
