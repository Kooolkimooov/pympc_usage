from copy import deepcopy
from time import perf_counter
from warnings import simplefilter

from matplotlib import pyplot as plt
from numpy import any, array, dot, isnan, linspace, ndarray, r_, set_printoptions, zeros
from numpy.linalg import norm
from tqdm import tqdm

from bluerov import Bluerov
from catenary import Catenary
from model import Model

simplefilter( 'ignore', RuntimeWarning )


class ChainOf2:

	state_size = 2 * Bluerov.state_size
	actuation_size = 2 * Bluerov.actuation_size

	def __init__(
			self,
			water_surface_z: float = 0.,
			water_current: ndarray = None,
			cables_lenght: float = 3.,
			cables_linear_mass: float = 0.,
			get_cable_parameter_method = 'runtime'
			):

		self.br_0 = Bluerov( water_surface_z, water_current )
		self.c_01 = Catenary( cables_lenght, cables_linear_mass, get_cable_parameter_method )
		self.br_1 = Bluerov( water_surface_z, water_current )

		self.last_perturbation_0 = zeros( (Bluerov.state_size // 2,) )
		self.last_perturbation_1 = zeros( (Bluerov.state_size // 2,) )

		self.br_0_pose = slice( 0, 6 )
		self.br_0_position = slice( 0, 3 )
		self.br_0_xy = slice( 0, 2 )
		self.br_0_z = 2
		self.br_0_orientation = slice( 3, 6 )

		self.br_1_pose = slice( 6, 12 )
		self.br_1_position = slice( 6, 9 )
		self.br_1_xy = slice( 6, 8 )
		self.br_1_z = 8
		self.br_1_orientation = slice( 9, 12 )

		self.br_0_speed = slice( 12, 18 )
		self.br_0_linear_speed = slice( 12, 15 )
		self.br_0_angular_speed = slice( 15, 18 )

		self.br_1_speed = slice( 18, 24 )
		self.br_1_linear_speed = slice( 18, 21 )
		self.br_1_angular_speed = slice( 21, 24 )

		self.br_0_actuation_start = 0
		self.br_0_actuation = slice( 0, 6 )
		self.br_0_linear_actuation = slice( 0, 3 )
		self.br_0_angular_actuation = slice( 3, 6 )

		self.br_1_actuation_start = 6
		self.br_1_actuation = slice( 6, 12 )
		self.br_1_linear_actuation = slice( 6, 9 )
		self.br_1_angular_actuation = slice( 9, 12 )

		self.br_0_state = r_[ self.br_0_pose, self.br_0_speed ]
		self.br_1_state = r_[ self.br_1_pose, self.br_1_speed ]

	def __call__( self, state: ndarray, actuation: ndarray ) -> ndarray:
		"""
		evalutes the dynamics of each robot of the chain
		:param state: current state of the system
		:param actuation: current actuation of the system
		:return: state derivative of the system
		"""
		state_derivative = zeros( state.shape )

		perturbation_01_0, perturbation_01_1 = self.c_01.get_perturbations(
				state[ self.br_0_position ], state[ self.br_1_position ]
				)

		# if the cable is taunt the perturbation is None
		# here we should consider any pair with a taunt cable as a single body
		if perturbation_01_0 is not None:
			self.last_perturbation_0[ :3 ] = perturbation_01_0
			self.last_perturbation_1[ :3 ] = perturbation_01_1
		else:
			perturbation_01_0, perturbation_01_1 = self.get_taunt_cable_perturbations( state, actuation, 0 )

		perturbation_01_0.resize( (Bluerov.actuation_size,), refcheck = False )
		perturbation_01_1.resize( (Bluerov.actuation_size,), refcheck = False )

		# perturbation is in world frame, should be applied robot frame instead
		br_0_transformation_matrix = self.br_0.build_transformation_matrix( *state[ self.br_0_orientation ] )
		br_1_transformation_matrix = self.br_1.build_transformation_matrix( *state[ self.br_1_orientation ] )
		perturbation_01_0 = br_0_transformation_matrix.T @ perturbation_01_0
		perturbation_01_1 = br_1_transformation_matrix.T @ perturbation_01_1

		state_derivative[ self.br_0_state ] = self.br_0(
				state[ self.br_0_state ], actuation[ self.br_0_actuation ], perturbation_01_0
				)
		state_derivative[ self.br_1_state ] = self.br_1(
				state[ self.br_1_state ], actuation[ self.br_1_actuation ], perturbation_01_1
				)

		return state_derivative

	def get_taunt_cable_perturbations( self, state: ndarray, actuation: ndarray, pair: int ) -> tuple[ ndarray,
	ndarray ]:
		match pair:
			case 0:
				br_0 = self.br_0
				c_01 = self.c_01
				br_1 = self.br_1
				br_0_state = self.br_0_state
				br_0_position = self.br_0_position
				br_0_orientation = self.br_0_orientation
				br_0_actuation = self.br_0_actuation
				br_0_perturbation = self.last_perturbation_0
				br_1_state = self.br_1_state
				br_1_position = self.br_1_position
				br_1_orientation = self.br_1_orientation
				br_1_actuation = self.br_1_actuation
				br_1_perturbation = self.last_perturbation_1
			case _:
				raise RuntimeError( f'unknown {pair=}' )

		# from br_0 to br_1
		direction = state[ br_1_position ] - state[ br_0_position ]
		direction /= norm( direction )

		null = zeros( (br_0.state_size // 2,) )

		br_0_transformation_matrix = br_0.build_transformation_matrix( *state[ br_0_orientation ] )
		br_1_transformation_matrix = br_1.build_transformation_matrix( *state[ br_1_orientation ] )

		# in robot frame
		br_0_acceleration = br_0( state[ br_0_state ], actuation[ br_0_actuation ], null )[ 6: ]
		br_0_forces = (br_0.inertial_matrix[ :3, :3 ] @ br_0_acceleration[ :3 ])

		br_1_acceleration = br_1( state[ br_1_state ], actuation[ br_1_actuation ], null )[ 6: ]
		br_1_forces = (br_1.inertial_matrix[ :3, :3 ] @ br_1_acceleration[ :3 ])

		all_forces = dot( br_0_transformation_matrix[ :3, :3 ] @ br_0_forces, -direction )
		all_forces += dot( br_1_transformation_matrix[ :3, :3 ] @ br_1_forces, direction )
		all_forces += dot( br_0_perturbation[ :3 ], direction )
		all_forces += dot( br_1_perturbation[ :3 ], -direction )

		perturbation = direction * all_forces

		# in world frame
		return perturbation, -perturbation


if __name__ == "__main__":
	set_printoptions( precision = 5, linewidth = 10000, suppress = True )

	ti = perf_counter()

	n_frames = 10000
	time_step = 0.01

	dynamics = ChainOf2( cables_linear_mass = .045, get_cable_parameter_method = 'precompute' )

	initial_state = zeros( (dynamics.state_size,) )
	initial_actuation = zeros( (dynamics.actuation_size,) )

	model = Model( dynamics, time_step, initial_state, initial_actuation )

	x0s = [ ]
	x1s = [ ]

	ds = [ ]
	t0s = [ ]
	t1s = [ ]
	l = 2.5

	model.actuation[ dynamics.br_0_actuation ][ 0 ] = 50

	for frame in tqdm( range( n_frames ) ):
		initial_state = deepcopy( model.state )
		model.step()

		if any( isnan( model.state ) ):
			print( f'nan at {frame=}' )
			break

		t0, t1 = dynamics.c_01.get_perturbations(
				model.state[ dynamics.br_0_position ], model.state[ dynamics.br_1_position ]
				)
		if t0 is None:
			t0, t1 = dynamics.get_taunt_cable_perturbations( model.state, model.actuation, 0 )

		ds += [ norm( model.state[ dynamics.br_1_position ] - model.state[ dynamics.br_0_position ] ) ]
		t0s += [ norm( t0 ) ]
		t1s += [ norm( t1 ) ]

		x0s += [ model.state[ dynamics.br_0_position ] ]
		x1s += [ model.state[ dynamics.br_1_position ] ]

	x0s, x1s = array( x0s ), array( x1s )
	T = [ i * time_step for i in range( len( x0s ) ) ]

	fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots( 2, 2, figsize = (16, 9) )
	fig.suptitle( f'{model.actuation[ dynamics.br_0_linear_actuation ]=}' )
	plt.subplots_adjust( hspace = 0 )

	line = ax1.scatter( ds, t0s, c = linspace( 0., time_step * len( ds ), len( ds ) ), cmap = 'summer' )
	plt.colorbar( line, location = 'top', label = 'time [s]' )
	ax1.set_ylabel( 'force du câble sur br_0 [N]' )
	ax1.axvline( 3. )

	ax2.scatter( ds, t1s, c = linspace( 0., time_step * len( ds ), len( ds ) ), cmap = 'summer' )
	ax2.set_xlabel( 'distances entre br_0 et br_1 [m]' )
	ax2.set_ylabel( 'force du câble sur br_1 [N]' )
	ax2.axvline( 3. )

	ax3.plot( T, x0s[ :, 0 ], color = 'r' )
	ax3.plot( T, x1s[ :, 0 ], color = 'b' )
	ax3.set_ylabel( 'position sur $x_w$' )
	ax3.legend( [ 'br_0', 'br_1' ] )

	ax4.plot( T, [ norm( x1 - x0 ) for x0, x1 in zip( x0s, x1s ) ] )
	ax4.axhline( 3. )
	ax4.set_xlabel( 'time [s]' )
	ax4.set_ylabel( 'distance entre br_0 et br_1' )

	plt.show()
