from copy import deepcopy

from numpy import arccosh, arcsinh, array, cosh, ndarray, sinh, sqrt, zeros
from numpy.f2py.crackfortran import get_parameters
from numpy.linalg import norm
from scipy.optimize import brentq


class Catenary:

	def __init__( self, length = 3., mass = 0. ):
		self.length = length
		self.mass = mass

		self.linear_mass = mass / length

		self.optimization_function = self.optimization_function_1

	def __call__( self, p1: ndarray, p2: ndarray ) -> ndarray:
		'''
		get the catenary's lowest point
		:param p1: one end of the catenary
		:param p2: second end of the catenary
		:return: the lowest point of the catenary
		'''

		distance = norm( p2 - p1 )
		if distance > self.length:
			raise RuntimeWarning( f'{p1} and {p2} to far apart for catenary of length {self.length}' )
		elif distance > .95 * self.length:
			return p1 if p1[ 2 ] >= p2[ 2 ] else p2
		elif distance < .05 * self.length:
			return p1 + array( [ 0, 0, self.length / 2 ] )

		_, H, dH, D, dD = self.get_parameters( p1, p2 )

		lowest_point = p1 + (D + dD) * (p2 - p1) / distance + array( [ 0, 0, H + dH ] )
		return lowest_point

	def get_perturbations( self, p1: ndarray, p2: ndarray ):
		C, H, dH, D, dD = self.get_parameters(p1, p2)

		horizontal_perturbation = self.linear_mass * 9.81 / C
		pass

	def discretize( self, p1, p2, n: int = 100 ) -> tuple[ ndarray, float, float, float, float, float ]:

		distance = norm( p2 - p1 )
		if distance > self.length:
			raise RuntimeWarning( f'{p1} and {p2} to far apart for catenary of length {self.length}' )
		elif distance > .95 * self.length:
			return array( [ p1, p2 ] ), -1., 0., p2[ 2 ] - p1[ 2 ], norm( p2[ :2 ] - p1[ :2 ] ), 0.
		elif distance < .05 * self.length:
			return array( [ p1, p2 ] ), -1., self.length / 2, p2[ 2 ] - p1[ 2 ], norm( p2[ :2 ] - p1[ :2 ] ), 0.

		C, H, dH, D, dD = self.get_parameters( p1, p2 )

		points = zeros( (100, 3) )

		s = 0.
		ds = self.length / n

		for i in range( n - 1 ):
			# get x,z coord of points in catenary frame, centered at 1st point
			inter = C * s - sinh( C * D )
			x = 1.0 / C * arcsinh( inter ) + D
			z = 1.0 / C * (sqrt( 1.0 + pow( inter, 2 ) ) - 1.0) - H
			points[ i ][ 0 ] = p1[ 0 ] + x * (p2[ 0 ] - p1[ 0 ]) / (2 * D + dD)
			points[ i ][ 1 ] = p1[ 1 ] - x * (p2[ 1 ] - p1[ 1 ]) / (2 * D + dD)
			points[ i ][ 2 ] = p1[ 2 ] - z
			s += ds

		points[ -1 ] = p2

		return points, C, H, dH, D, dD

	@staticmethod
	def optimization_function_1( C, length, dH, two_D_plus_dD ):
		return C * C * (length * length - dH * dH) - 2.0 * (-1.0 + cosh( C * two_D_plus_dD ))

	@staticmethod
	def optimization_function_2( C, length, dH, two_D_plus_dD ):
		return pow( length, 2 ) - pow( dH, 2 ) - pow( 2 * sinh( C * two_D_plus_dD / 2 ) / C, 2 )

	def get_parameters(
			self, p1: ndarray, p2: ndarray
			) -> tuple[ float, float, float, float, float ]:
		'''
		:param dH: vertical distance between the two attachment points (ΔH)
		:param two_D_plus_dD: horizontal distance between the two attachment points (2*D + ΔD)
		:return:
			- intrinsic parameter of the catenary (C)
			- vertical sag of the catenary (H)
			- (ΔH)
			- (D)
			- (ΔD)
		'''

		two_D_plus_dD = norm( p2[ :2 ] - p1[ :2 ] )
		dH = p2[ 2 ] - p1[ 2 ]

		C: float = brentq(
				self.optimization_function, -1e-2, 1e3, args = (self.length, dH, two_D_plus_dD), xtol = 1e-12
				)

		temp_var = pow( self.length, 2 ) - pow( dH, 2 )

		a_eq = -4. * pow( C, 2 ) * temp_var
		b_eq = -4. * C * dH * (C * temp_var - 2 * dH) - 8.0 * pow( self.length, 2 ) * C
		c_eq = pow( C * temp_var - 2. * dH, 2 )

		H = (-b_eq - sqrt( pow( b_eq, 2 ) - 4. * a_eq * c_eq )) / (2. * a_eq)
		D = arccosh( C * H + 1.0 ) / C
		dD = two_D_plus_dD - 2. * D

		return C, H, dH, D, dD


def test_1():
	'''
	to figure out the regions where the optimization over C fails and avoid them
	'''
	from numpy import linspace
	import matplotlib.pyplot as plt
	from tqdm import tqdm
	from warnings import simplefilter

	simplefilter( 'ignore', RuntimeWarning )

	cat = Catenary()
	lengths = linspace( 0., 100., 100 )

	ls = [ ]
	doverl = [ ]

	for l in tqdm( lengths ):
		distances = linspace( 0., l, 1000 )
		cat.length = l
		for d in distances:
			try:
				cat( array( [ 0, 0, 0 ] ), array( [ d, 0, 0 ] ) )
			except:
				ls += [ l ]
				doverl += [ d / l ]

	plt.scatter( ls, doverl )
	plt.show()


def test_2():
	'''
	to test good derivation of the catenary
	'''
	from numpy import linspace
	import matplotlib.pyplot as plt
	from tqdm import tqdm
	from warnings import simplefilter

	simplefilter( 'ignore', RuntimeWarning )

	cat = Catenary()
	X = linspace( 0., cat.length, 10 )
	Y = linspace( 0., cat.length, 10 )

	attachments = [ ]
	lowest_points = [ ]

	for x in tqdm( X ):
		for y in Y:
			try:
				p2 = array( [ x, y, 0 ] )
				lp = cat( array( [ 0, 0, 0 ] ), p2 )
				attachments += [ deepcopy( p2 ) ]
				lowest_points += [ deepcopy( lp ) ]
			except:
				pass

	attachments = array( attachments )
	lowest_points = array( lowest_points )

	ax = plt.subplot( projection = '3d' )
	ax.scatter( attachments[ :, 0 ], attachments[ :, 1 ], attachments[ :, 2 ], s = 1, color = 'b' )
	ax.scatter( lowest_points[ :, 0 ], lowest_points[ :, 1 ], lowest_points[ :, 2 ], s = 1, color = 'r' )
	# for i in range( attachments.shape[ 0 ] ):
	# 	points = cat.discretize( array( [ 0, 0, 0 ] ), attachments[ i ] )[ 0 ]
	# 	ax.plot( points[ :, 0 ], points[ :, 1 ], points[ :, 2 ] )
	# 	ax.plot(
	# 			[ attachments[ i, 0 ], lowest_points[ i, 0 ] ],
	# 			[ attachments[ i, 1 ], lowest_points[ i, 1 ] ],
	# 			[ attachments[ i, 2 ], lowest_points[ i, 2 ] ]
	# 			)
	ax.invert_zaxis()
	plt.show()


if __name__ == '__main__':
	test_1()

del test_1, test_2
