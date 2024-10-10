from copy import deepcopy
from time import perf_counter
from warnings import simplefilter

from matplotlib import pyplot as plt
from numpy import dot, isnan, linspace, ndarray, r_, set_printoptions, zeros, any
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
		if perturbation_01_0 is None:
			perturbation_01_0, perturbation_01_1 = self.get_taunt_cable_perturbations( state, actuation, 0 )
			perturbation_01_0.resize( (Bluerov.actuation_size,) )
			perturbation_01_1.resize( (Bluerov.actuation_size,) )
		else:
			perturbation_01_0.resize( (Bluerov.actuation_size,) )
			perturbation_01_1.resize( (Bluerov.actuation_size,) )
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
				br_1_state = self.br_1_state
				br_1_position = self.br_1_position
				br_1_orientation = self.br_1_orientation
				br_1_actuation = self.br_1_actuation
			case _:
				raise RuntimeError( f'unknown {pair=}' )

		direction = state[ br_1_position ] - state[ br_0_position ]
		direction /= norm( direction )

		null_perturbation = zeros( (br_0.state_size // 2,) )

		br_0_transformation_matrix = br_0.build_transformation_matrix( *state[ br_0_orientation ] )
		br_1_transformation_matrix = br_1.build_transformation_matrix( *state[ br_1_orientation ] )

		br_0_acceleration = br_0( state[ br_0_state ], state[ br_0_actuation ], null_perturbation )[ 6: ]
		br_0_forces = (br_0.inertial_matrix[ :3, :3 ] @ br_0_acceleration[ :3 ])
		br_0_forces += br_0_acceleration[ :3 ] * c_01.length * c_01.linear_mass / 2

		br_1_acceleration = br_1( state[ br_1_state ], state[ br_1_actuation ], null_perturbation )[ 6: ]
		br_1_forces = (br_1.inertial_matrix[ :3, :3 ] @ br_1_acceleration[ :3 ])
		br_1_forces += br_1_acceleration[ :3 ] * c_01.length * c_01.linear_mass / 2

		perturbation_01_0 = br_0_transformation_matrix[ :3, :3 ].T @ (direction * dot(
				br_1_transformation_matrix[ :3, :3 ] @ br_1_forces, direction
				))
		perturbation_01_1 = br_1_transformation_matrix[ :3, :3 ].T @ (direction * dot(
				br_0_transformation_matrix[ :3, :3 ] @ br_0_forces, direction
				))

		return perturbation_01_0, perturbation_01_1


if __name__ == "__main__":
	set_printoptions( precision = 3, linewidth = 10000, suppress = True )

	ti = perf_counter()

	n_frames = 10000
	time_step = 0.01

	dynamics = ChainOf2( cables_linear_mass = .045, get_cable_parameter_method = 'precompute' )

	initial_state = zeros( (dynamics.state_size,) )
	initial_actuation = zeros( (dynamics.actuation_size,) )

	model = Model( dynamics, time_step, initial_state, initial_actuation )

	_, ((ax1, ax2), (ax3, ax4)) = plt.subplots( 2, 2 )

	x0s = [ ]
	x1s = [ ]

	ds = [ ]
	t1s = [ ]
	t2s = [ ]

	model.actuation[ dynamics.br_0_actuation ][ 0 ] = 1.
	for frame in tqdm( range( n_frames ) ):
		initial_state = deepcopy(model.state)
		model.step()

		if any(isnan(model.state)):
			print(f'nan at {frame=}')
			model.state = deepcopy(initial_state)
			model.step()
			break

		t1, t2 = dynamics.c_01.get_perturbations(
				model.state[ dynamics.br_0_position ], model.state[ dynamics.br_1_position ]
				)
		if t1 is None:
			t1, t2 = dynamics.get_taunt_cable_perturbations( model.state, model.actuation, 0 )

		ds += [ norm( model.state[ dynamics.br_1_position ] - model.state[ dynamics.br_0_position ] ) ]
		t1s += [ t1[ 0 ] ]
		t2s += [ t2[ 0 ] ]

		x0s += [ model.state[ dynamics.br_0_position ][ 0 ] ]
		x1s += [ model.state[ dynamics.br_1_position ][ 0 ] ]

	ax1.scatter( ds, t1s, c = linspace( 0., 1., len( ds ) ), cmap = 'summer' )
	ax2.scatter( ds, t2s, c = linspace( 0., 1., len( ds ) ), cmap = 'summer' )

	ax3.plot( x0s )
	ax4.plot( x1s )

	plt.show()
