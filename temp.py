import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

from catenary import Catenary

np.set_printoptions( precision = 2, linewidth = 10000, suppress = True )

cat = Catenary( length = 3., get_parameter_method = 'precompute', reference_frame = 'ENU' )

def c_approximation_polynomial( data, p10, p11, p12, p13, p14, p20, p21, p22, p23, p24 ):

	p1 = [ p10, p11, p12, p13, p14 ]
	p2 = [ p20, p21, p22, p23, p24 ]
	two_D_plus_dD = data[ 0 ]
	dH = data[ 1 ]

	p1 = [ c * pow( dH, p ) for p, c in enumerate( p1 ) ]
	p2 = [ c * pow( two_D_plus_dD, p ) for p, c in enumerate( p2 ) ]

	return sum( p1 ) + sum( p2 )

def c_approximation_exponential( data, a0, b0, c0, d0, a1, b1, c1, d1 ):
	v0 = a0 * np.exp( b0 * (c0 + data[ 0 ]) ) + d0
	v1 = a1 * np.exp( b1 * (c1 + data[ 1 ]) ) + d1
	output = v0 + v1
	return output


xm, zm = np.meshgrid( cat._two_D_plus_dDs.copy(), cat._dHs.copy() )
_, (sp1, sp2, sp3) = plt.subplots( 1, 3, figsize = (15, 5), subplot_kw = { 'projection': '3d' } )
sp1.plot_surface( xm, zm, cat._Cs )

X = np.array( [ xm.flatten(), zm.flatten() ] )
Y = np.array( cat._Cs ).flatten()

with open('output.csv','w') as f:
	for (d, h), c in zip(X.T, Y):
		f.write(f'{d},{h},{c}\n')

exit()

try:
	params = sp.optimize.curve_fit( c_approximation_polynomial, X, Y, nan_policy = 'omit' )
	print( params[ 0 ] )
	C_poly = c_approximation_polynomial( np.array( [ xm, zm ] ), *params[ 0 ] )
	sp2.plot_surface( xm, zm, C_poly )
except Exception as e:
	print( e )

try:
	bounds = sp.optimize.Bounds( [ 0., -np.inf, -np.inf, -np.inf ] * 2, [ np.inf, 0, np.inf, np.inf ] * 2 )
	params = sp.optimize.curve_fit( c_approximation_exponential, X, Y, bounds=bounds, nan_policy = 'omit' )
	print( params[ 0 ] )
	C_exp = c_approximation_exponential( np.array( [ xm, zm ] ), *params[ 0 ] )
	sp3.plot_surface( xm, zm, C_exp )
except Exception as e:
	print( e )

plt.show()
