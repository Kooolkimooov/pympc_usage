from inspect import signature

from numpy import array, ndarray


class Seafloor:
	def get_distance_to_seafloor( self, point: ndarray ) -> float:
		raise NotImplementedError()


class SeafloorFromFunction( Seafloor ):
	def __init__( self, function: callable ):
		assert list( signature( function ).parameters ) == [ 'x', 'y' ]
		self.seafloor_function = function

	def get_distance_to_seafloor( self, point: ndarray ) -> float:
		return self.seafloor_function( *(point[ :2 ]) ) - point[ 2 ]


if __name__ == '__main__':
	from numpy import linspace, meshgrid, zeros, ones
	import matplotlib.pyplot as plt

	point = array( [ -4, 0, 0 ] )


	def seafloor_function( x, y ):
		from numpy import sin, exp
		z = 4.5
		z += 1. * sin( y / 4 )
		z += .5 * sin( x / 3 )
		# peak at (-3, 0)
		z -= 2.5 * exp( -8 * (pow( (x - (-3)), 2 ) + pow( (y - 0), 2 )) )
		return z

	X = linspace( -6.5, 4.5, 1000 )
	Y = linspace( -6.5, 4.5, 1000 )

	Xm, Ym = meshgrid( X, Y )
	Zws = zeros( Xm.shape )
	Zcsf = ones( Xm.shape ) * 4.
	Zsf = zeros( Xm.shape )

	for i, y in enumerate(Y):
		for j, x in enumerate(X):
			Zsf[i,j] = seafloor_function(x, y)

	ax = plt.subplot( projection = '3d' )
	ax.set_xlabel( r"$\mathbf{x}_w$-axis" )
	ax.set_ylabel( r"$\mathbf{y}_w$-axis" )
	ax.set_zlabel( r"$\mathbf{z}_w$-axis" )
	ax.invert_yaxis()
	ax.invert_zaxis()
	ax.plot_surface( Xm, Ym, Zws )
	ax.plot_surface( Xm, Ym, Zcsf )
	ax.plot_surface( Xm, Ym, Zsf )
	ax.scatter( *point )
	plt.show()

	seafloor: Seafloor = SeafloorFromFunction( seafloor_function )
	distance = seafloor.get_distance_to_seafloor( point )
	print( distance )
