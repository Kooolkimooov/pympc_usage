from json import dump, load
from os.path import join, split
from time import perf_counter, time
from warnings import simplefilter

from numpy import array, cos, diff, dot, exp, eye, inf, ndarray, pi, r_, set_printoptions, sin, zeros
from numpy.linalg import norm
from scipy.optimize import NonlinearConstraint

from bluerov import Bluerov, USV
from catenary import Catenary
from model import Model
from mpc import MPC
from seafloor import Seafloor, SeafloorFromFunction
from utils import check, generate_trajectory, get_computer_info, Logger, print_dict, serialize_others

simplefilter( 'ignore', RuntimeWarning )


class ChainOf4:

	state_size = 4 * Bluerov.state_size
	actuation_size = 3 * Bluerov.actuation_size + USV.actuation_size

	def __init__(
			self,
			water_surface_depth: float,
			water_current: ndarray,
			seafloor: Seafloor,
			cables_length: float,
			cables_linear_mass: float,
			get_cable_parameter_method
			):

		self.br_0 = Bluerov( water_surface_depth, water_current )
		self.c_01 = Catenary( cables_length, cables_linear_mass, get_cable_parameter_method )
		self.br_1 = Bluerov( water_surface_depth, water_current )
		self.c_12 = Catenary( cables_length, cables_linear_mass, get_cable_parameter_method )
		self.br_2 = Bluerov( water_surface_depth, water_current )
		self.c_23 = Catenary( cables_length, cables_linear_mass, get_cable_parameter_method )
		self.br_3 = USV( water_surface_depth )

		self.sf = seafloor

		self.last_perturbation_01_0 = zeros( (Bluerov.state_size // 2,) )
		self.last_perturbation_01_1 = zeros( (Bluerov.state_size // 2,) )
		self.last_perturbation_12_1 = zeros( (Bluerov.state_size // 2,) )
		self.last_perturbation_12_2 = zeros( (Bluerov.state_size // 2,) )
		self.last_perturbation_23_2 = zeros( (Bluerov.state_size // 2,) )
		self.last_perturbation_23_3 = zeros( (Bluerov.state_size // 2,) )

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

		self.br_2_pose = slice( 12, 18 )
		self.br_2_position = slice( 12, 15 )
		self.br_2_xy = slice( 12, 14 )
		self.br_2_z = 14
		self.br_2_orientation = slice( 15, 18 )

		self.br_3_pose = slice( 18, 24 )
		self.br_3_position = slice( 18, 21 )
		self.br_3_xy = slice( 18, 20 )
		self.br_3_z = 20
		self.br_3_orientation = slice( 21, 24 )

		self.br_0_speed = slice( 24, 30 )
		self.br_0_linear_speed = slice( 24, 27 )
		self.br_0_angular_speed = slice( 27, 30 )

		self.br_1_speed = slice( 30, 36 )
		self.br_1_linear_speed = slice( 30, 33 )
		self.br_1_angular_speed = slice( 33, 36 )

		self.br_2_speed = slice( 36, 42 )
		self.br_2_linear_speed = slice( 36, 39 )
		self.br_2_angular_speed = slice( 39, 42 )

		self.br_3_speed = slice( 42, 48 )
		self.br_3_linear_speed = slice( 42, 45 )
		self.br_3_angular_speed = slice( 45, 48 )

		self.br_0_actuation_start = 0
		self.br_0_actuation = slice( 0, 6 )
		self.br_0_linear_actuation = slice( 0, 3 )
		self.br_0_angular_actuation = slice( 3, 6 )

		self.br_1_actuation_start = 6
		self.br_1_actuation = slice( 6, 12 )
		self.br_1_linear_actuation = slice( 6, 9 )
		self.br_1_angular_actuation = slice( 9, 12 )

		self.br_2_actuation = slice( 12, 18 )
		self.br_2_actuation_start = 12
		self.br_2_linear_actuation = slice( 12, 15 )
		self.br_2_angular_actuation = slice( 15, 18 )

		self.br_3_actuation = slice( 18, 20 )
		self.br_3_actuation_start = 18
		self.br_3_linear_actuation = 18
		self.br_3_angular_actuation = 19

		self.br_0_state = r_[ self.br_0_pose, self.br_0_speed ]
		self.br_1_state = r_[ self.br_1_pose, self.br_1_speed ]
		self.br_2_state = r_[ self.br_2_pose, self.br_2_speed ]
		self.br_3_state = r_[ self.br_3_pose, self.br_3_speed ]

	def __call__( self, state: ndarray, actuation: ndarray ) -> ndarray:
		"""
		evaluates the dynamics of each robot of the chain
		:param state: current state of the system
		:param actuation: current actuation of the system
		:return: state derivative of the system
		"""
		state_derivative = zeros( state.shape )

		perturbation_01_0, perturbation_01_1 = self.c_01.get_perturbations(
				state[ self.br_0_position ], state[ self.br_1_position ]
				)
		perturbation_12_1, perturbation_12_2 = self.c_12.get_perturbations(
				state[ self.br_1_position ], state[ self.br_2_position ]
				)
		perturbation_23_2, perturbation_23_3 = self.c_23.get_perturbations(
				state[ self.br_2_position ], state[ self.br_3_position ]
				)

		# if the cable is taunt the perturbation is None
		# here we should consider any pair with a taunt cable as a single body
		if perturbation_01_0 is not None:
			self.last_perturbation_01_0[ :3 ] = perturbation_01_0
			self.last_perturbation_01_1[ :3 ] = perturbation_01_1
		else:
			perturbation_01_0, perturbation_01_1 = self.get_taunt_cable_perturbations( state, actuation, 0 )

		if perturbation_12_1 is not None:
			self.last_perturbation_12_1[ :3 ] = perturbation_12_1
			self.last_perturbation_12_2[ :3 ] = perturbation_12_2
		else:
			perturbation_12_1, perturbation_12_2 = self.get_taunt_cable_perturbations( state, actuation, 1 )

		if perturbation_23_2 is not None:
			self.last_perturbation_23_2[ :3 ] = perturbation_23_2
			self.last_perturbation_23_3[ :3 ] = perturbation_23_3
		else:
			perturbation_23_2, perturbation_23_3 = self.get_taunt_cable_perturbations( state, actuation, 2 )

		perturbation_01_0.resize( (Bluerov.actuation_size,), refcheck = False )
		perturbation_01_1.resize( (Bluerov.actuation_size,), refcheck = False )
		perturbation_12_1.resize( (Bluerov.actuation_size,), refcheck = False )
		perturbation_12_2.resize( (Bluerov.actuation_size,), refcheck = False )
		perturbation_23_2.resize( (Bluerov.actuation_size,), refcheck = False )
		perturbation_23_3.resize( (Bluerov.actuation_size,), refcheck = False )

		# perturbation is in world frame, should be applied robot frame instead
		br_0_transformation_matrix = self.br_0.build_transformation_matrix( *state[ self.br_0_orientation ] )
		br_1_transformation_matrix = self.br_1.build_transformation_matrix( *state[ self.br_1_orientation ] )
		br_2_transformation_matrix = self.br_2.build_transformation_matrix( *state[ self.br_2_orientation ] )
		br_3_transformation_matrix = self.br_3.build_transformation_matrix( *state[ self.br_3_orientation ] )

		perturbation_01_0 = br_0_transformation_matrix.T @ perturbation_01_0
		perturbation_01_1 = br_1_transformation_matrix.T @ perturbation_01_1
		perturbation_12_1 = br_1_transformation_matrix.T @ perturbation_12_1
		perturbation_12_2 = br_2_transformation_matrix.T @ perturbation_12_2
		perturbation_23_2 = br_2_transformation_matrix.T @ perturbation_23_2
		perturbation_23_3 = br_3_transformation_matrix.T @ perturbation_23_3

		state_derivative[ self.br_0_state ] = self.br_0(
				state[ self.br_0_state ], actuation[ self.br_0_actuation ], perturbation_01_0
				)
		state_derivative[ self.br_1_state ] = self.br_1(
				state[ self.br_1_state ], actuation[ self.br_1_actuation ], perturbation_01_1 + perturbation_12_1
				)
		state_derivative[ self.br_2_state ] = self.br_2(
				state[ self.br_2_state ], actuation[ self.br_2_actuation ], perturbation_12_2 + perturbation_23_2
				)
		state_derivative[ self.br_3_state ] = self.br_3(
				state[ self.br_3_state ], actuation[ self.br_3_actuation ], perturbation_23_3
				)

		return state_derivative

	def get_taunt_cable_perturbations(
			self, state: ndarray, actuation: ndarray, pair: int
			) -> tuple[ ndarray, ndarray ]:
		match pair:
			case 0:
				br_0 = self.br_0
				br_1 = self.br_1
				br_0_state = self.br_0_state
				br_0_position = self.br_0_position
				br_0_orientation = self.br_0_orientation
				br_0_actuation = self.br_0_actuation
				br_0_perturbation = self.last_perturbation_01_0
				br_1_state = self.br_1_state
				br_1_position = self.br_1_position
				br_1_orientation = self.br_1_orientation
				br_1_actuation = self.br_1_actuation
				br_1_perturbation = self.last_perturbation_01_1
			case 1:
				br_0 = self.br_1
				br_1 = self.br_2
				br_0_state = self.br_1_state
				br_0_position = self.br_1_position
				br_0_orientation = self.br_1_orientation
				br_0_actuation = self.br_1_actuation
				br_0_perturbation = self.last_perturbation_12_1
				br_1_state = self.br_2_state
				br_1_position = self.br_2_position
				br_1_orientation = self.br_2_orientation
				br_1_actuation = self.br_2_actuation
				br_1_perturbation = self.last_perturbation_12_2
			case 2:
				br_0 = self.br_2
				br_1 = self.br_3
				br_0_state = self.br_2_state
				br_0_position = self.br_2_position
				br_0_orientation = self.br_2_orientation
				br_0_actuation = self.br_2_actuation
				br_0_perturbation = self.last_perturbation_23_2
				br_1_state = self.br_3_state
				br_1_position = self.br_3_position
				br_1_orientation = self.br_3_orientation
				br_1_actuation = self.br_3_actuation
				br_1_perturbation = self.last_perturbation_23_3
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


def chain_of_4_constraints( self: MPC, candidate ):

	chain: ChainOf4 = self.model.dynamics

	actuation, _ = self.get_actuation( candidate )

	prediction = self.predict( actuation )
	prediction = prediction[ :, 0 ]

	# 3 constraints on cables (lowest points to seafloor),
	# 6 on inter robot_distance (3 horizontal, 2 3d),
	n_constraints = 3 + 6
	constraints = zeros( (self.horizon, n_constraints) )

	for i, state in enumerate( prediction ):
		constraints[ i, 0 ] = chain.sf.get_distance_to_seafloor(
				chain.c_01.get_lowest_point(
						state[ chain.br_0_position ], state[ chain.br_1_position ]
						)
				)
		constraints[ i, 1 ] = chain.sf.get_distance_to_seafloor(
				chain.c_12.get_lowest_point(
						state[ chain.br_1_position ], state[ chain.br_2_position ]
						)
				)
		constraints[ i, 2 ] = chain.sf.get_distance_to_seafloor(
				chain.c_23.get_lowest_point(
						state[ chain.br_2_position ], state[ chain.br_3_position ]
						)
				)

	# horizontal distance between consecutive robots
	constraints[ :, 3 ] = norm(
			prediction[ :, chain.br_1_xy ] - prediction[ :, chain.br_0_xy ], axis = 1
			)
	constraints[ :, 4 ] = norm(
			prediction[ :, chain.br_2_xy ] - prediction[ :, chain.br_1_xy ], axis = 1
			)
	constraints[ :, 5 ] = norm(
			prediction[ :, chain.br_3_xy ] - prediction[ :, chain.br_2_xy ], axis = 1
			)

	# distance between consecutive robots
	constraints[ :, 6 ] = norm(
			prediction[ :, chain.br_1_position ] - prediction[ :, chain.br_0_position ], axis = 1
			)
	constraints[ :, 7 ] = norm(
			prediction[ :, chain.br_2_position ] - prediction[ :, chain.br_1_position ], axis = 1
			)
	constraints[ :, 8 ] = norm(
			prediction[ :, chain.br_3_position ] - prediction[ :, chain.br_2_position ], axis = 1
			)

	return constraints.flatten()


def chain_of_4_objective( self: MPC, prediction: ndarray, actuation: ndarray ):

	chain: ChainOf4 = self.model.dynamics
	desired_distance = chain.c_01.length / 2

	objective = 0.

	# objective += pow( norm( prediction[ :, 0, chain.br_0_speed ], axis = 1 ).sum(), 2 )
	objective += pow( norm( prediction[ :, 0, chain.br_1_speed ], axis = 1 ).sum(), 2 )
	objective += pow( norm( prediction[ :, 0, chain.br_2_speed ], axis = 1 ).sum(), 2 )
	objective += pow( norm( prediction[ :, 0, chain.br_3_speed ], axis = 1 ).sum(), 2 )

	objective += abs(
			norm(
					prediction[ :, 0, chain.br_0_position ] - prediction[ :, 0, chain.br_1_position ], axis = 1
					) - desired_distance
			).sum()
	objective += abs(
			norm(
					prediction[ :, 0, chain.br_1_position ] - prediction[ :, 0, chain.br_2_position ], axis = 1
					) - desired_distance
			).sum()
	objective += abs(
			norm(
					prediction[ :, 0, chain.br_2_position ] - prediction[ :, 0, chain.br_3_position ], axis = 1
					) - desired_distance
			).sum()

	return objective


def seafloor_function( x, y ):
	z = 4.
	z += 1. * sin( x / 4 )
	z += 1. * sin( y / 3 )
	z += .05 * sin( 3 * (x * y) )
	z -= 2 * exp( - pow( (x - 4), 2 ) - pow( y, 2 ) )
	return z


if __name__ == "__main__":
	set_printoptions( precision = 2, linewidth = 10000, suppress = True )

	ti = perf_counter()

	n_frames = 200
	tolerance = 1e-6
	max_number_of_iteration = 100
	time_step = 0.1

	seafloor = SeafloorFromFunction( seafloor_function )

	dynamics = ChainOf4(
			water_surface_depth = 0.,
			water_current = array( [ .5, .5, 0. ] ),
			seafloor = seafloor,
			cables_length = 3.,
			cables_linear_mass = 0.01,
			get_cable_parameter_method = 'precompute'
			)

	initial_state = zeros( (dynamics.state_size,) )
	initial_state[ dynamics.br_0_position ][ 0 ] = 2.
	initial_state[ dynamics.br_0_position ][ 2 ] = 1.
	initial_state[ dynamics.br_1_position ][ 0 ] = 2.5
	initial_state[ dynamics.br_1_position ][ 2 ] = 1.
	initial_state[ dynamics.br_2_position ][ 0 ] = 3.
	initial_state[ dynamics.br_2_position ][ 2 ] = 1.
	initial_state[ dynamics.br_3_position ][ 0 ] = 3.5
	initial_state[ dynamics.br_3_orientation ][ 2 ] = pi / 2

	initial_actuation = zeros( (dynamics.actuation_size,) )

	horizon = 5
	time_steps_per_actuation = 5
	time_step_prediction_factor = 2
	assert time_step_prediction_factor * horizon < n_frames

	key_frames = [ (0., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 18), (.5, [ -5., 0., 0., 0., 0., 0. ] + [ 0. ] * 18),
								 (1., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 18), (2., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 18) ]

	trajectory = generate_trajectory( key_frames, 2 * n_frames )
	trajectory[ :, 0, dynamics.br_0_z ] = 1.5 * cos(
			1.25 * (trajectory[ :, 0, dynamics.br_0_position ][ :, 0 ] - 2) + pi
			) + 2.5

	max_required_speed = (max( norm( diff( trajectory[ :, 0, :3 ], axis = 0 ), axis = 1 ) ) / time_step)

	pose_weight_matrix = eye( initial_state.shape[ 0 ] // 2 )
	pose_weight_matrix[ dynamics.br_0_position, dynamics.br_0_position ] *= 10.
	pose_weight_matrix[ dynamics.br_0_orientation, dynamics.br_0_orientation ] *= 1.
	pose_weight_matrix[ dynamics.br_1_position, dynamics.br_1_position ] *= 0.
	pose_weight_matrix[ dynamics.br_1_orientation, dynamics.br_1_orientation ] *= 1.
	pose_weight_matrix[ dynamics.br_2_position, dynamics.br_2_position ] *= 0.
	pose_weight_matrix[ dynamics.br_2_orientation, dynamics.br_2_orientation ] *= 1.
	pose_weight_matrix[ dynamics.br_3_position, dynamics.br_3_position ] *= 0.
	pose_weight_matrix[ dynamics.br_3_orientation, dynamics.br_3_orientation ] *= 0.

	actuation_weight_matrix = eye( initial_actuation.shape[ 0 ] )
	actuation_weight_matrix[ dynamics.br_0_linear_actuation, dynamics.br_0_linear_actuation ] *= 0.
	actuation_weight_matrix[ dynamics.br_0_angular_actuation, dynamics.br_0_angular_actuation ] *= 1.
	actuation_weight_matrix[ dynamics.br_1_linear_actuation, dynamics.br_1_linear_actuation ] *= 0.
	actuation_weight_matrix[ dynamics.br_1_angular_actuation, dynamics.br_1_angular_actuation ] *= 1.
	actuation_weight_matrix[ dynamics.br_2_linear_actuation, dynamics.br_2_linear_actuation ] *= 0.
	actuation_weight_matrix[ dynamics.br_2_angular_actuation, dynamics.br_2_angular_actuation ] *= 1.
	actuation_weight_matrix[ dynamics.br_3_linear_actuation, dynamics.br_3_linear_actuation ] *= 0.
	actuation_weight_matrix[ dynamics.br_3_angular_actuation, dynamics.br_3_angular_actuation ] *= 0.

	final_cost_weight = 0.
	objective_weight = .01

	model = Model(
			dynamics = dynamics,
			time_step = time_step,
			initial_state = initial_state,
			initial_actuation = initial_actuation,
			record = True
			)

	mpc = MPC(
			model = model,
			horizon = horizon,
			target_trajectory = trajectory,
			optimize_on = 'actuation',
			objective_weight = objective_weight,
			time_step_prediction_factor = time_step_prediction_factor,
			tolerance = tolerance,
			max_number_of_iteration = max_number_of_iteration,
			time_steps_per_actuation = time_steps_per_actuation,
			pose_weight_matrix = pose_weight_matrix,
			actuation_derivative_weight_matrix = actuation_weight_matrix,
			final_weight = final_cost_weight,
			record = True,
			# verbose = True
			)

	# inject constraints and objective as member functions so that they may access self
	mpc.constraints_function = chain_of_4_constraints.__get__(
			mpc, MPC
			)

	mpc.objective = chain_of_4_objective.__get__(
			mpc, MPC
			)

	sf_lb = 0.2
	sf_ub = inf
	dp_lb = 0.2
	dp_ub = 2.8
	dr_lb = -inf
	dr_ub = 2.8

	constraints_values_labels = [ 'c_01_distance_to_seafloor', 'c_12_distance_to_seafloor', 'c_23_distance_to_seafloor',
																'br_0_br_1_horizontal_distance', 'br_1_br_2_horizontal_distance',
																'br_2_br_3_horizontal_distance', 'br_0_br_1_distance', 'br_1_br_2_distance',
																'br_2_br_3_distance' ]
	constraints_labels = [ 'seafloor', 'seafloor', 'seafloor', 'cable_length', 'cable_length', 'cable_length',
												 'cable_length', 'cable_length', 'cable_length' ]

	constraint_lb_base = [ sf_lb, sf_lb, sf_lb, dp_lb, dp_lb, dp_lb, dr_lb, dr_lb, dr_lb ]
	constraint_ub_base = [ sf_ub, sf_ub, sf_ub, dp_ub, dp_ub, dp_ub, dr_ub, dr_ub, dr_ub ]

	assert (len( constraint_lb_base ) == len( constraints_values_labels )) and (
			len( constraint_ub_base ) == len( constraints_values_labels ))

	lb = [ constraint_lb_base ] * horizon
	ub = [ constraint_ub_base ] * horizon

	constraint = NonlinearConstraint(
			mpc.constraints_function, array( lb ).flatten(), array( ub ).flatten()
			)

	constraint.value_labels = constraints_values_labels
	constraint.labels = constraints_labels

	mpc.constraints = (constraint,)

	previous_nfeval_record = [ 0 ]
	previous_H01_record = [ 0. ]
	previous_H12_record = [ 0. ]
	previous_H23_record = [ 0. ]

	folder = join(
			split( __file__ )[ 0 ], 'export', split( __file__ )[ 1 ].split( '.' )[ 0 ] + '_' + str( int( time() ) )
			)

	if check( folder ) + check( f'{folder}/data' ):
		exit()

	logger = Logger()

	with open( f'{folder}/config.json', 'w' ) as f:
		dump( mpc.__dict__ | get_computer_info(), f, default = serialize_others )

	with open( f'{folder}/config.json' ) as f:
		config = load( f )
		print_dict( config )

	if 'y' != input( 'continue ? (y/n) ' ):
		exit()

	for frame in range( n_frames ):

		mpc.target_trajectory = trajectory[ frame + 1: ]

		logger.log( f'frame {frame + 1}/{n_frames} starts at t={perf_counter() - ti:.2f}' )

		mpc.compute_actuation()
		mpc.apply_result()
		model.step()

		logger.log( f'ends at t={perf_counter() - ti:.2f}' )
		logger.log( f'{mpc.raw_result.message}' )

		# try to recover if the optimization failed
		if not mpc.raw_result.success and mpc.tolerance < 1:
			mpc.tolerance *= 10
			logger.log( f'increasing tolerance: {mpc.tolerance:.0e}' )
		elif mpc.raw_result.success and mpc.tolerance > 2 * tolerance:
			# *2 because of floating point error
			mpc.tolerance /= 10
			logger.log( f'decreasing tolerance: {mpc.tolerance:.0e}' )
		else:
			logger.log( f'keeping tolerance: {mpc.tolerance:.0e}' )

		objective_value = mpc.get_objective()
		logger.log( f'objective: {objective_value:.2f}' )

		constraints_values = mpc.constraints_function( mpc.raw_result.x )
		logger.log( f'constraints: {constraints_values[ :9 ]}' )

		logger.lognl( '' )
		logger.save_at( folder )

		# save simulation state
		with open( f'{folder}/data/{frame}.json', 'w' ) as f:
			dump( mpc.__dict__, f, default = serialize_others )
