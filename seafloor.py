from inspect import signature

from numpy import array, ndarray
from numpy.linalg import norm


class Seafloor:
	def get_distance_to_seafloor( self, point: ndarray ) -> float:
		raise NotImplementedError()


class SeafloorFromFunction( Seafloor ):
	def __init__( self, function: callable ):
		assert list( signature( function ).parameters ) == [ 'x', 'y' ]
		self.seafloor_function = function

	def get_distance_to_seafloor( self, point: ndarray ) -> float:
		return norm(point[2] - self.seafloor_function( *(point[ :2 ]) ))


if __name__ == '__main__':
	from numpy import sin, linspace, meshgrid, zeros, exp
	import matplotlib.pyplot as plt

	point = array( [ 1, 2, 3 ] )

	def seafloor_function( x, y ):
		z = 4.
		z += 1. * sin( x / 4 )
		z += 1. * sin( y / 3 )
		z += .05 * sin( 3 * (x * y) )
		z -= 2 * exp( - pow( (x - 4), 2 ) - pow( y, 2 ) )
		return z


	X = linspace( -5, 5, 1000 )
	Y = linspace( -5, 5, 1000 )

	Xm, Ym = meshgrid( X, Y )
	Zm = zeros( Xm.shape )
	WS = zeros( Xm.shape )
	for i, x in enumerate( X ):
		for j, y in enumerate( Y ):
			Zm[ i, j ] = seafloor_function( x, y )

	ax = plt.subplot( projection = '3d' )
	ax.set_xlabel( r"$\mathbf{x}_w$-axis" )
	ax.set_ylabel( r"$\mathbf{y}_w$-axis" )
	ax.set_zlabel( r"$\mathbf{z}_w$-axis" )
	ax.invert_yaxis()
	ax.invert_zaxis()
	ax.plot_surface( Xm, Ym, Zm )
	ax.plot_surface( Xm, Ym, WS )
	ax.scatter(*point)
	plt.show()

	seafloor: Seafloor = SeafloorFromFunction( seafloor_function )
	distance = seafloor.get_distance_to_seafloor( point )
	print( distance )
