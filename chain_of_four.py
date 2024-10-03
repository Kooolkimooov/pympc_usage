from json import dump, load
from time import perf_counter, time
from warnings import simplefilter

from numpy import array, cos, diff, dot, eye, inf, linspace, ndarray, pi, r_, set_printoptions, zeros
from numpy.linalg import norm
from scipy.optimize import NonlinearConstraint

from bluerov import Bluerov, USV
from catenary import Catenary
from model import Model
from mpc import MPC
from utils import check, generate_trajectory, get_computer_info, Logger, print_dict, serialize_others

simplefilter( 'ignore', RuntimeWarning )


class ChainOf4:

	state_size = 4 * Bluerov.state_size
	actuation_size = 3 * Bluerov.actuation_size + USV.actuation_size

	def __init__(
			self,
			water_surface_z: float = 0.,
			water_current: ndarray = None,
			cables_lenght: float = 3.,
			cables_linear_mass: float = 0.
			):

		self.br_0 = Bluerov( water_surface_z, water_current )
		self.c_01 = Catenary( cables_lenght, cables_linear_mass )
		self.br_1 = Bluerov( water_surface_z, water_current )
		self.c_12 = Catenary( cables_lenght, cables_linear_mass )
		self.br_2 = Bluerov( water_surface_z, water_current )
		self.c_23 = Catenary( cables_lenght, cables_linear_mass )
		self.br_3 = USV( water_surface_z )

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
		evalutes the dynamics of each robot of the chain
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
		if perturbation_01_0 is None:
			perturbation_01_0, perturbation_01_1 = self.get_taunt_cable_perturbations( state, actuation, 0 )

		if perturbation_12_1 is None:
			perturbation_12_1, perturbation_12_2 = self.get_taunt_cable_perturbations( state, actuation, 1 )

		if perturbation_23_2 is None:
			perturbation_23_2, perturbation_23_3 = self.get_taunt_cable_perturbations( state, actuation, 2 )

		perturbation_01_0.resize( (Bluerov.actuation_size,) )
		perturbation_01_1.resize( (Bluerov.actuation_size,) )
		perturbation_12_1.resize( (Bluerov.actuation_size,) )
		perturbation_12_2.resize( (Bluerov.actuation_size,) )
		perturbation_23_2.resize( (Bluerov.actuation_size,) )
		perturbation_23_3.resize( (Bluerov.actuation_size,) )

		state_derivative[ self.br_0_state ] = self.br_0(
				state[ self.br_0_state ], actuation[ self.br_0_actuation ], perturbation_01_0
				)
		state_derivative[ self.br_1_state ] = self.br_1(
				state[ self.br_1_state ],
				actuation[ self.br_1_actuation ],
				perturbation_01_1 + perturbation_12_1
				)
		state_derivative[ self.br_2_state ] = self.br_2(
				state[ self.br_2_state ],
				actuation[ self.br_2_actuation ],
				perturbation_12_2 + perturbation_23_2
				)
		state_derivative[ self.br_3_state ] = self.br_3(
				state[ self.br_3_state ], actuation[ self.br_3_actuation ], perturbation_23_3
				)

		return state_derivative

	def get_taunt_cable_perturbations( self, state: ndarray, actuation: ndarray, pair: int ) -> tuple[ ndarray,
	ndarray ]:
		match pair:
			case 0:
				br_0 = self.br_0
				br_1 = self.br_1
				br_0_state = self.br_0_state
				br_0_position = self.br_0_position
				br_0_orientation = self.br_0_orientation
				br_0_actuation = self.br_0_actuation
				br_1_state = self.br_1_state
				br_1_position = self.br_1_position
				br_1_orientation = self.br_1_orientation
				br_1_actuation = self.br_1_actuation
			case 1:
				br_0 = self.br_1
				br_1 = self.br_2
				br_0_state = self.br_1_state
				br_0_position = self.br_1_position
				br_0_orientation = self.br_1_orientation
				br_0_actuation = self.br_1_actuation
				br_1_state = self.br_2_state
				br_1_position = self.br_2_position
				br_1_orientation = self.br_2_orientation
				br_1_actuation = self.br_2_actuation
			case 2:
				br_0 = self.br_2
				br_1 = self.br_3
				br_0_state = self.br_2_state
				br_0_position = self.br_2_position
				br_0_orientation = self.br_2_orientation
				br_0_actuation = self.br_2_actuation
				br_1_state = self.br_3_state
				br_1_position = self.br_3_position
				br_1_orientation = self.br_3_orientation
				br_1_actuation = self.br_3_actuation
			case _:
				raise RuntimeError( f'unknown {pair=}' )

		direction = state[ br_1_position ] - state[ br_0_position ]
		direction /= norm( direction )

		br_0_transformation_matrix = br_0.build_transformation_matrix( *state[ br_0_orientation ] )
		br_1_transformation_matrix = br_1.build_transformation_matrix( *state[ br_1_orientation ] )

		br_0_acceleration = br_0( state[ br_0_state ], state[ br_0_actuation ], self.water_current_force )[ 6: ]
		br_1_acceleration = br_1( state[ br_1_state ], state[ br_1_actuation ], self.water_current_force )[ 6: ]
		br_0_forces = (br_0.inertial_matrix @ br_0_acceleration)[ :3 ]
		br_1_forces = (br_1.inertial_matrix @ br_1_acceleration)[ :3 ]

		perturbation_01_0 = br_0_transformation_matrix[ :3, :3 ] @ (direction * dot(
				br_1_transformation_matrix[ :3, :3 ] @ br_1_forces, direction
				))
		perturbation_01_1 = br_1_transformation_matrix[ :3, :3 ] @ (direction * dot(
				br_0_transformation_matrix[ :3, :3 ] @ br_0_forces, direction
				))

		return perturbation_01_0, perturbation_01_1


def chain_of_4_constraints( self: MPC, candidate ):

	chain: ChainOf4 = self.model.model_dynamics

	actuation, _ = self.get_actuation( candidate )

	prediction = self.predict( actuation )
	prediction = prediction[ :, 0 ]

	# 3 constraints on cables (lowest points.z),
	# 6 on inter robot_distance (3 horizontal, 2 3d),
	# 3 on robot speed
	n_constraints = 3 + 6 + 3
	constraints = zeros( (self.horizon, n_constraints) )

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

	# speed
	constraints[ :, 9 ] = norm( prediction[ :, chain.br_0_speed ], axis = 1 )
	constraints[ :, 10 ] = norm( prediction[ :, chain.br_1_speed ], axis = 1 )
	constraints[ :, 11 ] = norm( prediction[ :, chain.br_2_speed ], axis = 1 )

	for i, state in enumerate( prediction ):
		constraints[ i, 0 ] = chain.c_01.get_lowest_point(
				state[ chain.br_0_position ], state[ chain.br_1_position ]
				)[ 2 ]
		constraints[ i, 1 ] = chain.c_12.get_lowest_point(
				state[ chain.br_1_position ], state[ chain.br_2_position ]
				)[ 2 ]
		constraints[ i, 2 ] = chain.c_23.get_lowest_point(
				state[ chain.br_2_position ], state[ chain.br_3_position ]
				)[ 2 ]

	return constraints.flatten()


def chain_of_4_objective( self: MPC, prediction: ndarray, actuation: ndarray ):

	chain: ChainOf4 = self.model.model_dynamics

	objective = 0.

	objective += norm( prediction[ :, 0, chain.br_0_linear_speed ], axis = 1 ).sum()
	objective += norm( prediction[ :, 0, chain.br_1_linear_speed ], axis = 1 ).sum()
	objective += norm( prediction[ :, 0, chain.br_2_linear_speed ], axis = 1 ).sum()

	objective += abs(
			norm(
					prediction[ :, 0, chain.br_0_xy ] - prediction[ :, 0, chain.br_1_xy ], axis = 1
					) - 1.5
			).sum()
	objective += abs(
			norm(
					prediction[ :, 0, chain.br_1_xy ] - prediction[ :, 0, chain.br_2_xy ], axis = 1
					) - 1.5
			).sum()
	objective += abs(
			norm(
					prediction[ :, 0, chain.br_2_xy ] - prediction[ :, 0, chain.br_3_xy ], axis = 1
					) - 1.5
			).sum()

	return objective


if __name__ == "__main__":
	set_printoptions( precision = 2, linewidth = 10000, suppress = True )

	ti = perf_counter()

	n_frames = 200
	tolerance = 1e-6
	max_number_of_iteration = 100
	time_step = 0.1

	model = ChainOf4( water_current = array( [ .5, .5, 0. ] ) )

	initial_state = zeros( (model.state_size,) )
	initial_state[ model.br_0_position ][ 0 ] = 2.
	initial_state[ model.br_0_position ][ 2 ] = 1.
	initial_state[ model.br_1_position ][ 0 ] = 2.5
	initial_state[ model.br_1_position ][ 2 ] = 1.
	initial_state[ model.br_2_position ][ 0 ] = 3.
	initial_state[ model.br_2_position ][ 2 ] = 1.
	initial_state[ model.br_3_position ][ 0 ] = 3.5

	initial_actuation = zeros( (model.actuation_size,) )

	horizon = 5
	time_steps_per_actuation = 5

	key_frames = [ (0., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 18), (.5, [ -3., 0., 0., 0., 0., 0. ] + [ 0. ] * 18),
								 (1., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 18), (2., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 18) ]

	trajectory = generate_trajectory( key_frames, 2 * n_frames )
	times = linspace( 0, trajectory.shape[ 0 ] * time_step, trajectory.shape[ 0 ] )
	trajectory[ :, 0, model.br_0_z ] = 1.5 * cos(
			1.25 * (trajectory[ :, 0, model.br_0_position ][ :, 0 ] - 2) + pi
			) + 2.5
	# trajectory[ :, 0, model.br_3_z ] += -2.5 * sin( times / 6 )
	# trajectory[ :, 0, model.br_3_z ] += + .2 * sin( times )
	# trajectory[ :, 0, model.br_3_z ] += + .1 * sin( 3.3 * times )
	# trajectory[ :, 0, model.br_3_position ][ :, 0 ] = 3.5

	trajectory_derivative = diff( trajectory, append = trajectory[ :1, :, : ], axis = 0 ) / time_step

	max_required_speed = (max( norm( diff( trajectory[ :, 0, :3 ], axis = 0 ), axis = 1 ) ) / time_step)

	# import matplotlib.pyplot as plt
	# plt.plot( trajectory[:, 0, model.br_3_z] )
	# # plt.plot( norm(diff(trajectory[:, 0, :3], axis=0), axis=1) / time_step )
	# plt.show()
	# exit()

	pose_weight_matrix = eye( initial_state.shape[ 0 ] // 2 )
	pose_weight_matrix[ model.br_0_position, model.br_0_position ] *= 10.
	pose_weight_matrix[ model.br_0_orientation, model.br_0_orientation ] *= 1.
	pose_weight_matrix[ model.br_1_position, model.br_1_position ] *= 0.
	pose_weight_matrix[ model.br_1_orientation, model.br_1_orientation ] *= 1.
	pose_weight_matrix[ model.br_2_position, model.br_2_position ] *= 0.
	pose_weight_matrix[ model.br_2_orientation, model.br_2_orientation ] *= 1.
	pose_weight_matrix[ model.br_3_position, model.br_3_position ] *= 0.
	pose_weight_matrix[ model.br_3_orientation, model.br_3_orientation ] *= 0.

	actuation_weight_matrix = eye( initial_actuation.shape[ 0 ] )
	actuation_weight_matrix[ model.br_0_linear_actuation, model.br_0_linear_actuation ] *= 1e-12
	actuation_weight_matrix[ model.br_0_angular_actuation, model.br_0_angular_actuation ] *= 1.
	actuation_weight_matrix[ model.br_1_linear_actuation, model.br_1_linear_actuation ] *= 1e-12
	actuation_weight_matrix[ model.br_1_angular_actuation, model.br_1_angular_actuation ] *= 1.
	actuation_weight_matrix[ model.br_2_linear_actuation, model.br_2_linear_actuation ] *= 1e-12
	actuation_weight_matrix[ model.br_2_angular_actuation, model.br_2_angular_actuation ] *= 1.
	actuation_weight_matrix[ model.br_3_linear_actuation, model.br_3_linear_actuation ] *= 1e-3
	actuation_weight_matrix[ model.br_3_angular_actuation, model.br_3_angular_actuation ] *= 1.

	final_cost_weight = 0.
	objective_weight = .1

	chain_model = Model(
			model, time_step, initial_state, initial_actuation, record = True
			)

	chain_mpc = MPC(
			chain_model,
			horizon,
			trajectory,
			optimize_on = 'actuation',
			objective_weight = objective_weight,
			tolerance = tolerance,
			max_iter = max_number_of_iteration,
			time_steps_per_actuation = time_steps_per_actuation,
			pose_weight_matrix = pose_weight_matrix,
			actuation_derivative_weight_matrix = actuation_weight_matrix,
			final_weight = final_cost_weight,
			record = True,
			# verbose = True
			)

	# inject constraints and objective as member functions so that they may access self
	chain_mpc.constraints_function = chain_of_4_constraints.__get__(
			chain_mpc, MPC
			)

	chain_mpc.objective = chain_of_4_objective.__get__(
			chain_mpc, MPC
			)

	floor_depth = 4.00001
	# du_l_ub = 2000.
	# du_l_lb = -2000.
	# du_a_ub = .1
	# du_a_lb = -.1
	dp_lb = 0.4
	dp_ub = 2.8
	dr_lb = -inf
	dr_ub = 2.8
	v_lb = -inf
	v_ub = 1.5

	if v_ub < max_required_speed and 'y' == input(
			f'the trajectory requires {max_required_speed} but the speed limit is {v_ub}, upgrade ? (y/n)'
			):
		v_ub = int( max_required_speed ) + 1.

	# bounds_lb_base = [ du_l_lb, du_l_lb, du_l_lb, du_a_lb, du_a_lb, du_a_lb ]
	# bounds_ub_base = [ du_l_ub, du_l_ub, du_l_ub, du_a_ub, du_a_ub, du_a_ub ]

	# three_bluerov_chain_with_fixed_end_mpc.bounds = Bounds(
	# 		array( [ bounds_lb_base[ model.br0_actuation ] ] ).repeat( 3, axis = 0 ).flatten(),
	# 		array( [ bounds_ub_base[ model.br0_actuation ] ] ).repeat( 3, axis = 0 ).flatten()
	# 		)

	constraints_labels = [ '$z_0+H_{01}$', '$z_1+H_{12}$', '$z_2+H_{2fe}$', '$|P_0^{x,y}-P_1^{x,y}|$',
												 '$|P_1^{x,y}-P_2^{x,y}|$', '$|P_2^{x,y}-P_fe^{x,y}|$', '$|P_0^{x,y,z}-P_1^{x,y,z}|$',
												 '$|P_1^{x,y,z}-P_2^{x,y,z}|$', '$|P_2^{x,y,z}-P_fe^{x,y,z}|$', '$|V_0|$', '$|V_1|$',
												 '$|V_2|$' ]

	constraint_lb_base = [ -inf, -inf, -inf, dp_lb, dp_lb, dp_lb, dr_lb, dr_lb, dr_lb, v_lb, v_lb, v_lb ]
	constraint_ub_base = [ floor_depth, floor_depth, floor_depth, dp_ub, dp_ub, dp_ub, dr_ub, dr_ub, dr_ub, v_ub, v_ub,
												 v_ub ]

	assert (len( constraint_lb_base ) == len( constraints_labels )) and (
			len( constraint_ub_base ) == len( constraints_labels ))

	lb = [ constraint_lb_base ] * horizon
	ub = [ constraint_ub_base ] * horizon

	constraint = NonlinearConstraint(
			chain_mpc.constraints_function, array( lb ).flatten(), array( ub ).flatten()
			)
	constraint.labels = constraints_labels

	chain_mpc.constraints = (constraint,)

	previous_nfeval_record = [ 0 ]
	previous_H01_record = [ 0. ]
	previous_H12_record = [ 0. ]
	previous_H23_record = [ 0. ]

	folder = f'./export/{__file__.split( '\\' )[ -1 ].split( '.' )[ 0 ]}_{int( time() )}'

	if check( folder ) + check( f'{folder}/data' ):
		exit()

	logger = Logger()

	with open( f'{folder}/config.json', 'w' ) as f:
		dump( chain_mpc.__dict__ | get_computer_info(), f, default = serialize_others )

	with open( f'{folder}/config.json' ) as f:
		config = load( f )
		print_dict( config )

	if 'y' != input( 'continue ? (y/n) ' ):
		exit()

	for frame in range( n_frames ):

		chain_mpc.target_trajectory = trajectory[ frame + 1:frame + horizon + 1 ]

		logger.log( f'frame {frame + 1}/{n_frames} starts at t={perf_counter() - ti:.2f}' )

		chain_mpc.compute_actuation()
		chain_mpc.apply_result()
		chain_model.step()

		logger.log( f'ends at t={perf_counter() - ti:.2f}' )

		logger.log( f'{chain_mpc.raw_result.success}' )
		logger.log( f'{chain_mpc.raw_result.message}' )

		# try to recover if the optimization failed
		if not chain_mpc.raw_result.success and chain_mpc.tolerance < 1:
			chain_mpc.tolerance *= 10
			logger.log( f'increasing tolerance: {chain_mpc.tolerance:.0e}' )
		elif chain_mpc.raw_result.success and chain_mpc.tolerance > 2 * tolerance:
			# *2 because of floating point error
			chain_mpc.tolerance /= 10
			logger.log( f'decreasing tolerance: {chain_mpc.tolerance:.0e}' )
		else:
			logger.log( f'keeping tolerance: {chain_mpc.tolerance:.0e}' )

		constraints_values = chain_mpc.constraints_function( chain_mpc.raw_result.x )
		logger.log( f'constraints: {constraints_values[ :12 ]}' )

		logger.lognl( '' )
		logger.save_at( folder )

		# save simulation state
		with open( f'{folder}/data/{frame}.json', 'w' ) as f:
			dump( chain_mpc.__dict__, f, default = serialize_others )
