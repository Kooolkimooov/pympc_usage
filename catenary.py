from json import dump, load
from pathlib import Path

from numpy import (
	arccosh,
	arcsinh,
	array,
	cosh,
	isnan,
	linspace,
	log10,
	logspace,
	meshgrid,
	ndarray,
	set_printoptions,
	sinh,
	sqrt,
	zeros,
	)
from numpy.linalg import norm
from scipy.optimize import brentq
from tqdm import tqdm

from utils import check, G


class Catenary:
	"""
	Catenary class with the NED convention meaning that the z axis is pointing downward
	"""

	GET_PARAMETER_METHOD = [ 'runtime', 'precompute' ]

	def __init__(
			self, length = 3., linear_mass = 1., get_parameter_method: str = 'runtime'
			):

		self.length = length
		self.linear_mass = linear_mass
		self.optimization_function = self._optimization_function_0

		match get_parameter_method:
			case 'runtime':
				self.get_parameters = self._get_parameters_runtime
			case 'precompute':
				self._precompute()
				self.get_parameters = self._get_parameters_precompute
			case _:
				raise ValueError( f'get_parameter_method must be one of {self.GET_PARAMETER_METHOD}' )

	def __call__( self, p0: ndarray, p1: ndarray ):
		"""
		get all relevant data on the catenary of length self.length, linear mass self.linear_mass, and the given
		attachment points
		:param p0: first attachment point
		:param p1: second attachment point
		:return: tuple containing:
		- the parameters of the catenary:
			- the parameter of the catenary (C, set to None if out of safe search space);
			- vertical sag of the catenary (H, set to None if out of safe search space and 2D+ΔD > length);
			- vertical distance between attachment points (ΔH, set to None if out of safe search space and 2D+ΔD >
			length);
			- horizontal half-length (D, set to None if out of safe search space and 2D+ΔD > length);
			- horizontal asymmetric length (ΔD, set to None if out of safe search space and 2D+ΔD > length);
		- the lowest point (x, y, z) of the catenary;
		- the perturbations force on the two points in the form (perturbation_p0, perturbation_p1);
		- array of points for plotting (x, y, z) are on the second dimension of the array.
		"""
		C, H, dH, D, dD = self.get_parameters( p0, p1 )
		lowest_point = self._get_lowest_point( p0, p1, C, H, dH, D, dD )
		perturbations = self._get_perturbations( p0, p1, C, H, dH, D, dD )
		points = self._discretize( p0, p1, C, H, D, dD )

		return (C, H, dH, D, dD), lowest_point, perturbations, points

	def get_lowest_point( self, p0: ndarray, p1: ndarray ) -> ndarray:
		"""
		get the catenary's lowest point
		:param p0: one end of the catenary
		:param p1: second end of the catenary
		:return: the lowest point (x, y, z) of the catenary
		"""
		C, H, dH, D, dD = self.get_parameters( p0, p1 )
		return self._get_lowest_point( p0, p1, C, H, dH, D, dD )

	def get_perturbations( self, p0: ndarray, p1: ndarray ):
		"""
		get the perturbations of the two points
		:param p0: one end of the catenary
		:param p1: second end of the catenary
		:return: tuple containing the perturbations force on the two points in the form (perturbation_p1, perturbation_p2)
		"""
		C, H, dH, D, dD = self.get_parameters( p0, p1 )
		return self._get_perturbations( p0, p1, C, H, dH, D, dD )

	def discretize( self, p0: ndarray, p1: ndarray, n: int = 100 ) -> ndarray:
		"""
		discretize the catenary, if the optimization fails, the catenary is approximated by a straight line
		:param p0: one end of the catenary
		:param p1: second end of the catenary
		:param n: number of point to discretize
		:return: array of points of the catenary points are on the second dimension of the array
		"""
		C, H, dH, D, dD = self.get_parameters( p0, p1 )
		return self._discretize( p0, p1, C, H, D, dD, n )

	@staticmethod
	def optimization_function( C, length, dH, two_D_plus_dD ):
		raise NotImplementedError( 'optimization_function method should have been implemented in __init__' )

	def get_parameters( self, p0: ndarray, p1: ndarray ) -> tuple[ float, float, float, float, float ]:
		"""
		:param p0: first attachment point
		:param p1: second attachment point
		:return: tuple containing:
		- the parameter of the catenary (C, set to None if out of safe search space);
		- vertical sag of the catenary (H, set to None if out of safe search space and 2D+ΔD > length);
		- vertical distance between attachment points (ΔH, set to None if out of safe search space and 2D+ΔD > length);
		- horizontal half-length (D, set to None if out of safe search space and 2D+ΔD > length);
		- horizontal asymmetric length (ΔD, set to None if out of safe search space and 2D+ΔD > length)
		"""
		raise NotImplementedError( 'get_parameters method should have been implemented in __init__' )

	def _get_parameters_runtime( self, p0: ndarray, p1: ndarray ) -> tuple[ float, float, float, float, float ]:
		"""
		implementation of get_parameters using optimization
		"""

		dH = p1[ 2 ] - p0[ 2 ]
		two_D_plus_dD = norm( p1[ :2 ] - p0[ :2 ] )

		if norm( p1 - p0 ) > 0.99 * self.length or any( isnan( p0 ) ) or any( isnan( p1 ) ):
			return None, None, dH, None, None
		elif two_D_plus_dD < .01 * self.length:
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

	def _precompute( self ):

		self._dHs = linspace( 0., self.length, 1000 )
		self._two_D_plus_dDs = self.length * logspace( -2, 0, 1000 )

		check( Path( f'./cache' ), prompt = False )
		check( Path( f'./cache/Catenary' ), prompt = False )
		if len( list( Path( f'./cache/Catenary' ).glob( f'{self.length}*' ) ) ):
			with open( Path( f'./cache/Catenary/{self.length}.json' ) ) as file:
				self._Cs = array( load( file ) )
				return

		X, Z = meshgrid( self._two_D_plus_dDs, self._dHs )
		self._Cs = zeros( X.shape )

		for i, xr in enumerate( tqdm( X, desc = 'precomputing values of C' ) ):
			for j, x in enumerate( xr ):
				z = Z[ i, j ]
				p1 = array( [ 0., 0., 0. ] )
				p2 = array( [ x, 0., z ] )
				self._Cs[ i, j ], _, _, _, _ = self._get_parameters_runtime( p1, p2 )

		with open( Path( f'./cache/Catenary/{self.length}.json' ), 'w' ) as file:
			dump( self._Cs.tolist(), file )

	def _get_parameters_precompute( self, p0: ndarray, p1: ndarray ) -> tuple[ float, float, float, float, float ]:
		"""
		implementation of get_parameters using optimization
		"""
		dH = p1[ 2 ] - p0[ 2 ]
		two_D_plus_dD = norm( p1[ :2 ] - p0[ :2 ] )

		if norm( p1 - p0 ) > 0.99 * self.length or any( isnan( p0 ) ) or any( isnan( p1 ) ):
			return None, None, dH, None, None
		elif two_D_plus_dD < .01 * self.length:
			return None, (self.length - dH) / 2, dH, two_D_plus_dD / 2, 0.

		i = int( round( (1000 - 1) * abs( dH ) / self.length, 0 ) )
		j = int( round( (1000 - 1) * (log10( abs( two_D_plus_dD ) / self.length ) - (-2)) / (0 - (-2)), 0 ) )
		if (0 < i and not self._dHs[ i - 1 ] < abs(dH)) or (i < 999 and not abs(dH) < self._dHs[ i + 1 ]):
			raise ValueError()
		if (0 < j and not self._two_D_plus_dDs[ j - 1 ] < abs(two_D_plus_dD)) or (
				j < 999 and not abs(two_D_plus_dD) < self._two_D_plus_dDs[ j + 1 ]):
			raise ValueError()

		C = self._Cs[ i, j ]

		if isnan( C ):
			return None, None, dH, None, None

		temp_var = pow( self.length, 2 ) - pow( dH, 2 )

		a_eq = -4. * pow( C, 2 ) * temp_var
		b_eq = -4. * C * dH * (C * temp_var - 2 * dH) - 8.0 * pow( self.length, 2 ) * C
		c_eq = pow( C * temp_var - 2. * dH, 2 )

		H = (-b_eq - sqrt( pow( b_eq, 2 ) - 4. * a_eq * c_eq )) / (2. * a_eq)
		D = arccosh( C * H + 1.0 ) / C
		dD = two_D_plus_dD - 2. * D

		return C, H, dH, D, dD

	def _get_lowest_point( self, p0: ndarray, p1: ndarray, C: float, H: float, dH: float, D: float, dD: float ):

		# case where horizontal distance is too small
		if (C is None) and (H is not None):
			return p0 + array( [ 0, 0, H + dH ] )
		# case where cable is taunt
		elif C is None:
			return p0 if p0[ 2 ] >= p1[ 2 ] else p1

		lowest_point = zeros( (3,) )
		lowest_point[ :2 ] = p0[ :2 ] + (D + dD) * (p1[ :2 ] - p0[ :2 ]) / norm( p1[ :2 ] - p0[ :2 ] )
		lowest_point[ 2 ] = p0[ 2 ] + H + dH
		return lowest_point

	def _get_perturbations(
			self, p0: ndarray, p1: ndarray, C: float, H: float, dH: float, D: float, dD: float
			) -> tuple[ ndarray, ndarray ]:

		# case where horizontal distance is too small
		if (C is None) and (D is not None):
			return array(
					[ 0., 0., self.linear_mass * G * (H + dH) ]
					), array(
					[ 0., 0., self.linear_mass * G * H ]
					)
		# case where cable is taunt
		elif C is None:
			return None, None

		horizontal_perturbation = self.linear_mass * G / C
		vertical_perturbation_0 = horizontal_perturbation * sinh( -C * (D + dD) )
		vertical_perturbation_1 = horizontal_perturbation * sinh( C * D )

		direction = (p1[ :2 ] - p0[ :2 ]) / norm( p1[ :2 ] - p0[ :2 ] )

		perturbation_p0, perturbation_p1 = zeros( (3,) ), zeros( (3,) )
		perturbation_p0[ :2 ] = direction * horizontal_perturbation
		perturbation_p0[ 2 ] = -vertical_perturbation_0
		perturbation_p1[ :2 ] = -direction * horizontal_perturbation
		perturbation_p1[ 2 ] = vertical_perturbation_1

		return perturbation_p0, perturbation_p1

	def _discretize( self, p0: ndarray, p1: ndarray, C: float, H: float, D: float, dD: float, n: int = 100 ) -> ndarray:

		# case where ΔH is too small
		if (C is None) and (D is not None):
			return array( [ p0, p0 + array( [ 0, 0, H ] ), p1 ] )
		# case where cable is taunt
		elif C is None:
			return array( [ p0, p1 ] )

		points = zeros( (100, 3) )

		s = 0.
		ds = self.length / n

		for i in range( n - 1 ):
			# get x,z coord of points in catenary frame, centered at 1st point
			inter = C * s - sinh( C * D )
			x = 1. / C * arcsinh( inter ) + D
			z = 1. / C * (sqrt( 1. + pow( inter, 2 ) ) - 1.) - H
			points[ i, 0 ] = p1[ 0 ] - x * (p1[ 0 ] - p0[ 0 ]) / (2. * D + dD)
			points[ i, 1 ] = p1[ 1 ] - x * (p1[ 1 ] - p0[ 1 ]) / (2. * D + dD)
			points[ i, 2 ] = p1[ 2 ] - z
			s += ds

		points[ -1 ] = p0

		return points

	@staticmethod
	def _optimization_function_0( C, length, dH, two_D_plus_dD ):
		return C * C * (length * length - dH * dH) - 2.0 * (-1.0 + cosh( C * two_D_plus_dD ))

	@staticmethod
	def _optimization_function_1( C, length, dH, two_D_plus_dD ):
		return pow( length, 2 ) - pow( dH, 2 ) - pow( 2 * sinh( C * two_D_plus_dD / 2 ) / C, 2 )


def test_1():
	"""
	to figure out the regions where the optimization over C fails and avoid them
	"""
	from numpy import linspace, nanmean, nanstd, nanquantile, nan
	import matplotlib.pyplot as plt
	from tqdm import tqdm
	from warnings import simplefilter
	from time import perf_counter

	simplefilter( 'ignore', RuntimeWarning )

	cat = Catenary()
	# cat.optimization_function = cat._optimization_function_0 # μ=0.0004014649499549705 σ=0.0002544858910793164
	# cat.optimization_function = cat._optimization_function_1  # μ=0.00039219858003561965 σ=0.00023520437741971333
	lengths = linspace( 3., 10., 1 )

	xlpisp = [ ]
	zlpisp = [ ]
	cs = [ ]
	xs = [ ]
	zs = [ ]
	ts = [ ]
	ls = [ ]
	xoverl = [ ]
	zoverl = [ ]

	for l in lengths:
		X = linspace( 0., l, 200 )
		# Z = linspace( -l, l, 400 )
		Z = linspace( 0, 0, 1 )
		cat.length = l
		with tqdm( X, desc = f'test 1 {l=:.2f}' ) as X:
			for x in X:
				for z in Z:
					X.display( f'{X}\t{x=:.2f}\t{z=:.2f}\tn={len( ls )}' )
					p1 = array( [ 0, 0, 0 ] )
					p2 = array( [ x, 0, z ] )
					try:
						ti = perf_counter()
						# (C, H, dH, D, dD), lowest_point, perturbations, points
						(c, _, _, _, _), lp, _, _ = cat( p1, p2 )
						ts += [ perf_counter() - ti ]
						cs += [ c ] if c is not None else [ nan ]
						xs += [ x / l ]
						zs += [ z / l ]
						if norm( lp - p1 ) < 1e-3 or norm( lp - p2 ) < 1e-3:
							xlpisp += [ x / l ]
							zlpisp += [ z / l ]
					except:
						ls += [ l ]
						xoverl += [ x / l ]
						zoverl += [ z / l ]

	print( 'times' )
	print( nanmean( ts ), nanstd( ts ) )
	print( nanquantile( ts, [ 0.01, 0.25, 0.50, 0.75, 0.99 ] ) )
	print( 'C' )
	print( nanmean( cs ), nanstd( cs ) )
	print( nanquantile( cs, [ 0.01, 0.25, 0.50, 0.75, 0.99 ] ) )

	plt.figure()
	plt.scatter( xs, cs, s = .1 )
	plt.xlabel( r'$(2D + \Delta D) / L$' )
	plt.ylabel( r'$C$' )

	plt.figure()
	plt.scatter( zs, cs, s = .1 )
	plt.xlabel( r'$\Delta H / L$' )
	plt.ylabel( r'$C$' )

	plt.figure()
	plt.scatter( [ sqrt( pow( x, 2 ) + pow( z, 2 ) ) for x, z in zip( xs, zs ) ], cs, s = .1 )
	plt.xlabel( r'$\Delta H / L$' )
	plt.ylabel( r'$C$' )

	plt.figure()
	plt.hist( cs, bins = 100 )
	plt.xlabel( 'value of C' )
	plt.ylabel( 'occurrences' )

	plt.figure()
	plt.scatter( xoverl, zoverl, s = .1 )
	plt.xlabel( r'$(2D + \Delta D) / L$' )
	plt.ylabel( r'$\Delta H / L$' )
	plt.axis( 'equal' )
	plt.legend( [ 'failed optimization' ] )
	plt.grid()

	plt.figure()
	plt.scatter( xlpisp, zlpisp, s = .1 )
	plt.xlabel( r'$(2D + \Delta D) / L$' )
	plt.ylabel( r'$\Delta H / L$' )
	plt.axis( 'equal' )
	plt.legend( [ 'lowest point is p1 or p2' ] )
	plt.grid()

	plt.show()


def test_2():
	"""
	to test good derivation of the catenary
	"""
	from numpy import linspace
	import matplotlib.pyplot as plt
	from warnings import simplefilter

	simplefilter( 'ignore', RuntimeWarning )

	cat = Catenary( linear_mass = 1. )
	X = linspace( -cat.length, cat.length, 7 )
	Z = linspace( 0, 0, 1 )

	for z in Z:
		for x in X:
			try:
				p1 = array( [ 0, 0, 0 ] )
				p2 = array( [ x, 0, z ] )

				out = cat( p1, p2 )

				C, H, dH, D, dD = out[ 0 ]
				lowest_point = out[ 1 ]
				perturbations = out[ 2 ]
				points = out[ 3 ]

				plt.figure()
				plt.gca().set_xlim([-3, 3])
				plt.gca().set_ylim([-3, 3])
				plt.gca().invert_yaxis()

				plt.scatter( *p1[ ::2 ], s = 50 )
				plt.scatter( *p2[ ::2 ], s = 50 )
				plt.scatter( *lowest_point[ ::2 ], s = 50 )

				plt.quiver( *p1[ ::2 ], *perturbations[ 0 ][ ::2 ], angles='xy', scale = 50 )
				plt.quiver( *p2[ ::2 ], *perturbations[ 1 ][ ::2 ], angles='xy', scale = 50 )

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
				print( perturbations )
				plt.show()

			except:
				pass


def test_3():
	from numpy import linspace, array
	from warnings import simplefilter
	import matplotlib.pyplot as plt
	from tqdm import tqdm

	simplefilter( 'ignore', RuntimeWarning )

	cat = Catenary()

	ds = [ ]
	t1s, t2s = [ ], [ ]

	X = linspace( -cat.length, cat.length, 500 )
	Z = linspace( -cat.length, cat.length, 500 )
	X, Z = meshgrid( X, Z )
	P1, P2 = zeros( X.shape ), zeros( X.shape )

	for i, xr in enumerate( tqdm( X ) ):
		for j, x in enumerate( xr ):
			z = Z[ i, j ]
			p1 = array( [ 0, 0, 0 ] )
			p2 = array( [ x, 0, z ] )
			t1, t2 = cat.get_perturbations( p1, p2 )
			if t1 is not None:
				t1, t2 = norm( t1 ), norm( t2 )
				P1[ i, j ] = t1
				P2[ i, j ] = t2

	_, (ax1, ax2) = plt.subplots( 2, 1, subplot_kw = { 'projection': '3d' } )

	ax1.plot_surface( X, Z, P1 )
	ax2.plot_surface( X, Z, P2 )
	plt.show()


def test_4():
	from numpy import linspace, meshgrid, logspace
	from warnings import simplefilter
	import matplotlib.pyplot as plt
	from tqdm import tqdm

	simplefilter( 'ignore', RuntimeWarning )

	# strange case
	dx = 2.781439881986825
	dz = -2.480654570646834e-16
	c = -1.2166747574150463e-08

	cat = Catenary()
	X = cat.length * logspace( -6, 0, 1000 )
	Z = linspace( 0., cat.length, 1000 )
	X, Z = meshgrid( X, Z )
	C = zeros( X.shape )

	for i, xr in enumerate( tqdm( X ) ):
		for j, x in enumerate( xr ):
			z = Z[ i, j ]
			p1 = array( [ 0., 0., 0. ] )
			p2 = array( [ x, 0., z ] )
			c, _, _, _, _ = cat.get_parameters( p1, p2 )
			C[ i, j ] = c

	ax = plt.subplot( projection = '3d' )
	ax.plot_surface( X, Z, C )
	ax.set_xlabel( r'horizontal distance $2D+\Delta D$' )
	ax.set_ylabel( r'vertical distance $\Delta H$' )
	ax.set_zlabel( r'value of $C$' )
	plt.show()


def test_5():
	L = 3.
	e0 = -6
	en = 0
	n = 69

	logvs = L * logspace( e0, en, n )
	linvs = linspace( 0, L, n )

	print( 'log' )
	for i, v in enumerate( logvs ):
		print( v, end = '\t' )
		print( log10( v / L ), end = '\t' )
		print( int( round( (n - 1) * (log10( v / L ) - e0) / (en - e0), 0 ) ), end = '\t' )
		print( i, end = '\t' )
		print( i == round( (n - 1) * (log10( v / L ) - e0) / (en - e0), 0 ), end = '\t' )
		print()

	print( 'lin' )
	for i, v in enumerate( linvs ):
		print( v, end = '\t' )
		print( int( round( (n - 1) * abs( v ) / L, 0 ) ), end = '\t' )
		print( i, end = '\t' )
		print( i == int( round( (n - 1) * abs( v ) / L, 0 ) ), end = '\t' )
		print()


if __name__ == '__main__':
	set_printoptions( precision = 5, linewidth = 10000, suppress = True )
	# test_1()
	test_2()
	# test_3()
	# test_4()
	# test_5()
	pass

del test_1, test_2, test_3, test_4, test_5
