from numpy import arccosh, arcsinh, array, cosh, mean, ndarray, quantile, sinh, sqrt, std, zeros
from numpy.linalg import norm
from scipy.optimize import brentq

from utils import G


class Catenary:
	"""
	Catenary class with the NED convention meaning that the z axis is pointing downward
	"""

	def __init__( self, length = 3., linear_mass = 0. ):
		self.length = length
		self.linear_mass = linear_mass

		self.optimization_function = self._optimization_function_1
		self.get_parameters = self._get_parameters_optimization

	def __call__( self, p1: ndarray, p2: ndarray ):
		'''
		get all relevant data on the catenary of length self.length, linear mass self.linear_mass, and the given
		attachment points
		:param p1: first attachment point
		:param p2: second attachment point
		:return: tuple containing:
		- the parameters of the catenary:
			- the parameter of the catenary (C, set to None if out of safe search space);
			- vertical sag of the catenary (H, set to None if out of safe search space and 2D+ΔD > length);
			- vertical distance between attachment points (ΔH, set to None if out of safe search space and 2D+ΔD > length);
			- horizontal half-length (D, set to None if out of safe search space and 2D+ΔD > length);
			- horizontal asymmetric length (ΔD, set to None if out of safe search space and 2D+ΔD > length);
		- the lowest point (x, y, z) of the catenary;
		- the perturbations force on the two points in the form (perturbation_p1, perturbation_p2);
		- array of points for plotting (x, y, z) are on the second dimension of the array.
		'''
		C, H, dH, D, dD = self.get_parameters( p1, p2 )
		lowest_point = self._get_lowest_point( p1, p2, C, H, dH, D, dD )
		perturbations = self._get_perturbations( p1, p2, C, D, dD )
		points = self._discretize( p1, p2, C, H, D, dD )

		return (C, H, dH, D, dD), lowest_point, perturbations, points

	def get_lowest_point( self, p1: ndarray, p2: ndarray ) -> ndarray:
		'''
		get the catenary's lowest point
		:param p1: one end of the catenary
		:param p2: second end of the catenary
		:return: the lowest point (x, y, z) of the catenary
		'''
		C, H, dH, D, dD = self.get_parameters( p1, p2 )
		return self._get_lowest_point( p1, p2, C, H, dH, D, dD )

	def get_perturbations( self, p1: ndarray, p2: ndarray ):
		'''
		get the perturbations of the two points
		:param p1: one end of the catenary
		:param p2: second end of the catenary
		:return: tuple containing the perturbations force on the two points in the form (perturbation_p1, perturbation_p2)
		'''
		C, H, dH, D, dD = self.get_parameters( p1, p2 )
		return self._get_perturbations( p1, p2, C, D, dD )

	def discretize( self, p1: ndarray, p2: ndarray, n: int = 100 ) -> ndarray:
		'''
		discretize the catenary, if the optimization fails, the catenary is approximated by a straight line
		:param p1: one end of the catenary
		:param p2: second end of the catenary
		:param n: number of point to discretize
		:return: array of points of the catenary points are on the second dimension of the array
		'''
		C, H, dH, D, dD = self.get_parameters( p1, p2 )
		return self._discretize( p1, p2, C, H, D, dD, n )

	@staticmethod
	def optimization_function( C, length, dH, two_D_plus_dD ):
		raise NotImplementedError( 'optimization_function method should have been implemented in __init__' )

	def get_parameters( self, p1: ndarray, p2: ndarray ) -> tuple[ float, float, float, float, float ]:
		"""
		:param p1: first attachment point
		:param p2: second attachment point
		:return: tuple containing:
		- the parameter of the catenary (C, set to None if out of safe search space);
		- vertical sag of the catenary (H, set to None if out of safe search space and 2D+ΔD > length);
		- vertical distance between attachment points (ΔH, set to None if out of safe search space and 2D+ΔD > length);
		- horizontal half-length (D, set to None if out of safe search space and 2D+ΔD > length);
		- horizontal asymmetric length (ΔD, set to None if out of safe search space and 2D+ΔD > length)
		"""
		raise NotImplementedError( 'get_parameters method should have been implemented in __init__' )

	def _get_parameters_optimization( self, p1: ndarray, p2: ndarray ) -> tuple[ float, float, float, float, float ]:
		'''
		implementation of get_parameters using optimization
		'''

		dH = p2[ 2 ] - p1[ 2 ]
		two_D_plus_dD = norm( p2[ :2 ] - p1[ :2 ] )

		if norm( p2 - p1 ) > 0.99 * self.length:
			return None, None, None, None, None
		elif norm( p2[ :2 ] - p1[ :2 ] ) < .01 * self.length:
			return None, (self.length - dH) / 2, dH, two_D_plus_dD / 2, 0.

		C = brentq(
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

	def _get_lowest_point( self, p1: ndarray, p2: ndarray, C: float, H: float, dH: float, D: float, dD: float ):
		if (C is None) and (H is not None):
			return p1 + array( [ 0, 0, self.length / 2 ] )
		elif C is None:
			return p1 if p1[ 2 ] >= p2[ 2 ] else p2

		lowest_point = zeros( (3,) )
		lowest_point[ :2 ] = p1[ :2 ] + (D + dD) * (p2[ :2 ] - p1[ :2 ]) / norm( p2[ :2 ] - p1[ :2 ] )
		lowest_point[ 2 ] = p1[ 2 ] + H + dH
		return lowest_point

	def _get_perturbations( self, p1: ndarray, p2: ndarray, C: float, D: float, dD: float ) -> tuple[ ndarray, ndarray ]:

		if (C is None) and (D is not None):
			return array(
					[ 0., 0., self.linear_mass * self.length * G / 2 ]
					), array(
					[ 0., 0., self.linear_mass * self.length * G / 2 ]
					)
		elif C is None:
			return None, None

		horizontal_perturbation = self.linear_mass * G / C
		vertical_perturbation_1 = horizontal_perturbation * sinh( -C * (D + dD) )
		vertical_perturbation_2 = horizontal_perturbation * sinh( C * D )

		direction = (p2[ :2 ] - p1[ :2 ]) / norm( p2[ :2 ] - p1[ :2 ] )

		perturbation_p1, perturbation_p2 = zeros( (3,) ), zeros( (3,) )
		perturbation_p1[ :2 ] = direction * horizontal_perturbation
		perturbation_p1[ 2 ] = vertical_perturbation_1
		perturbation_p2[ :2 ] = -direction * horizontal_perturbation
		perturbation_p2[ 2 ] = -vertical_perturbation_2

		return perturbation_p1, perturbation_p2

	def _discretize( self, p1: ndarray, p2: ndarray, C: float, H: float, D: float, dD: float, n: int = 100 ) -> ndarray:

		if (C is None) and (D is not None):
			return array( [ p1, p1 + array( [ 0, 0, H ] ), p2 ] )
		elif C is None:
			return array( [ p1, p2 ] )

		points = zeros( (100, 3) )

		s = 0.
		ds = self.length / n

		for i in range( n - 1 ):
			# get x,z coord of points in catenary frame, centered at 1st point
			inter = C * s - sinh( C * D )
			x = 1. / C * arcsinh( inter ) + D
			z = 1. / C * (sqrt( 1. + pow( inter, 2 ) ) - 1.) - H
			points[ i ][ 0 ] = p2[ 0 ] - x * (p2[ 0 ] - p1[ 0 ]) / (2. * D + dD)
			points[ i ][ 1 ] = p2[ 1 ] - x * (p2[ 1 ] - p1[ 1 ]) / (2. * D + dD)
			points[ i ][ 2 ] = p2[ 2 ] - z
			s += ds

		points[ -1 ] = p1

		return points

	@staticmethod
	def _optimization_function_1( C, length, dH, two_D_plus_dD ):
		return C * C * (length * length - dH * dH) - 2.0 * (-1.0 + cosh( C * two_D_plus_dD ))

	@staticmethod
	def _optimization_function_2( C, length, dH, two_D_plus_dD ):
		return pow( length, 2 ) - pow( dH, 2 ) - pow( 2 * sinh( C * two_D_plus_dD / 2 ) / C, 2 )


def test_1():
	'''
	to figure out the regions where the optimization over C fails and avoid them
	'''
	from numpy import linspace
	import matplotlib.pyplot as plt
	from tqdm import tqdm
	from warnings import simplefilter
	from time import perf_counter

	simplefilter( 'ignore', RuntimeWarning )

	cat = Catenary()
	# cat.optimization_function = cat._optimization_function_1 # μ=0.0004014649499549705 σ=0.0002544858910793164
	cat.optimization_function = cat._optimization_function_2  # μ=0.00039219858003561965 σ=0.00023520437741971333
	lengths = linspace( 3., 3., 1 )

	ts = [ ]
	ls = [ ]
	xoverl = [ ]
	zoverl = [ ]

	for l in lengths:
		X = linspace( 0., l, 100 )
		Z = linspace( -l, l, 200 )
		cat.length = l
		with tqdm( X, desc = f'test 1 {l=:.2f}' ) as X:
			for x in X:
				for z in Z:
					X.display( f'{X}\t{x=:.2f}\t{z=:.2f}\tn={len( ls )}' )
					try:
						ti = perf_counter()
						cat( array( [ 0, 0, 0 ] ), array( [ x, 0, z ] ) )
						ts += [ perf_counter() - ti ]
					except:
						ls += [ l ]
						xoverl += [ x / l ]
						zoverl += [ z / l ]

	print( mean( ts ), std( ts ) )
	print( quantile( ts, [ 0.01, 0.25, 0.50, 0.75, 0.99 ] ) )

	plt.figure()
	plt.scatter( ls, xoverl )
	plt.axis( 'equal' )
	plt.grid()

	plt.figure()
	plt.scatter( xoverl, zoverl )
	plt.axis( 'equal' )
	plt.grid()

	plt.show()


def test_2():
	'''
	to test good derivation of the catenary
	'''
	from numpy import linspace
	import matplotlib.pyplot as plt
	from warnings import simplefilter

	simplefilter( 'ignore', RuntimeWarning )

	cat = Catenary( linear_mass = 1. / 3. )
	X = linspace( 0., cat.length, 20 )
	Z = linspace( -cat.length, cat.length, 4 )

	for z in Z:
		for x in X:
			try:
				p1 = array( [ 0, 0, 0 ] )
				p2 = array( [ x, 0, z ] )

				plt.scatter( *p1[ ::2 ], s = 50 )
				plt.scatter( *p2[ ::2 ], s = 50 )

				out = cat( p1, p2 )
				C, H, dH, D, dD = out[ 0 ]
				lowest_point = out[ 1 ]
				perturbations = out[ 2 ]
				points = out[ 3 ]

				plt.scatter( *lowest_point[ ::2 ], s = 50 )

				plt.quiver( *p1[ ::2 ], *perturbations[ 0 ][ ::2 ], scale = 50 )
				plt.quiver( *p2[ ::2 ], *perturbations[ 1 ][ ::2 ], scale = 50 )

				plt.plot( points[ :, 0 ], points[ :, 2 ] )

				plt.plot( [ 0, dD ], [ 0, 0 ] )
				plt.plot( [ 0, 0 ], [ 0, dH ] )
				plt.plot( [ dD, dD ], [ 0, dH ] )
				plt.plot( [ 0, dD ], [ dH, dH ] )
				plt.plot( [ 0, dD ], [ 0, dH ], ':', linewidth = 5 )

				plt.plot( [ dD, dD ], [ dH, H + dH ] )
				plt.plot( [ dD, D + dD ], [ dH, dH ] )
				plt.plot( [ D + dD, D + dD ], [ H + dH, dH ] )
				plt.plot( [ dD, D + dD ], [ H + dH, H + dH ] )
				plt.plot( [ dD, D + dD ], [ dH, H + dH ], ':', linewidth = 5 )

				plt.plot( [ D + dD, 2 * D + dD ], [ dH, dH ] )
				plt.plot( [ D + dD, 2 * D + dD ], [ H + dH, H + dH ] )
				plt.plot( [ 2 * D + dD, 2 * D + dD ], [ dH, H + dH ] )

				plt.plot( [ D + dD, 2 * D + dD ], [ H + dH, dH ], ':', linewidth = 5 )

				plt.title( f'{p2=}' )
				print( f'{p2=}' )
				plt.gca().invert_yaxis()
				plt.axis( 'equal' )
				plt.show()

			except:
				pass


if __name__ == '__main__':
	test_1()
	# test_2()
	pass

del test_1, test_2
