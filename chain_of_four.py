from json import dump
from time import perf_counter, time
from warnings import simplefilter

from numpy import array, cos, diff, eye, inf, linspace, ndarray, pi, r_, sin, zeros
from numpy.linalg import norm
from scipy.optimize import NonlinearConstraint

from bluerov import Bluerov
from catenary import Catenary
from model import Model
from mpc import MPC
from utils import check, generate_trajectory, Logger, serialize_others

simplefilter( 'ignore', RuntimeWarning )


class ChainOf4:

	state_size = 4 * Bluerov.state_size
	actuation_size = 4 * Bluerov.actuation_size

	def __init__( self ):
		self.br_0 = Bluerov()
		self.c_01 = Catenary()
		self.br_1 = Bluerov()
		self.c_12 = Catenary()
		self.br_2 = Bluerov()
		self.c_23 = Catenary()
		self.br_3 = Bluerov()

		self.water_current_force = array( [ 0, 0, 0, 0, 0, 0 ] )

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
		self.br_0_linear_actuation = slice( 0, 6 )
		self.br_0_angular_actuation = slice( 0, 0 )

		self.br_1_actuation_start = 6
		self.br_1_actuation = slice( 6, 12 )
		self.br_1_linear_actuation = slice( 6, 9 )
		self.br_1_angular_actuation = slice( 9, 12 )

		self.br_2_actuation = slice( 12, 18 )
		self.br_2_actuation_start = 12
		self.br_2_linear_actuation = slice( 12, 15 )
		self.br_2_angular_actuation = slice( 15, 18 )

		self.br_3_actuation = slice( 18, 24 )
		self.br_3_actuation_start = 18
		self.br_3_linear_actuation = slice( 18, 21 )
		self.br_3_angular_actuation = slice( 21, 24 )

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
			perturbation_01_0 = zeros( (3,) )
			perturbation_01_1 = zeros( (3,) )

		if perturbation_12_1 is None:
			perturbation_12_1 = zeros( (3,) )
			perturbation_12_2 = zeros( (3,) )

		if perturbation_23_2 is None:
			perturbation_23_2 = zeros( (3,) )
			perturbation_23_3 = zeros( (3,) )

		perturbation_01_0.resize( (Bluerov.actuation_size,) )
		perturbation_01_1.resize( (Bluerov.actuation_size,) )
		perturbation_12_1.resize( (Bluerov.actuation_size,) )
		perturbation_12_2.resize( (Bluerov.actuation_size,) )
		perturbation_23_2.resize( (Bluerov.actuation_size,) )
		perturbation_23_3.resize( (Bluerov.actuation_size,) )

		state_derivative[ self.br_0_state ] = self.br_0(
				state[ self.br_0_state ], actuation[ self.br_0_actuation ], perturbation_01_0 + self.water_current_force
				)
		state_derivative[ self.br_1_state ] = self.br_1(
				state[ self.br_1_state ],
				actuation[ self.br_1_actuation ],
				perturbation_01_1 + perturbation_12_1 + self.water_current_force
				)
		state_derivative[ self.br_2_state ] = self.br_2(
				state[ self.br_2_state ],
				actuation[ self.br_2_actuation ],
				perturbation_12_2 + perturbation_23_2 + self.water_current_force
				)
		state_derivative[ self.br_3_state ] = self.br_3(
				state[ self.br_3_state ], actuation[ self.br_3_actuation ], perturbation_23_3 + self.water_current_force
				)

		return state_derivative


def chain_of_4_constraints( self: MPC, candidate ):

	chain: ChainOf4 = self.model.model_dynamics

	actuation, actuation_derivatives = self.get_actuation( candidate )

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
					prediction[ :, 0, chain.br_0_position ] - prediction[ :, 0, chain.br_1_position ], axis = 1
					) - 1.5
			).sum()
	objective += abs(
			norm(
					prediction[ :, 0, chain.br_1_position ] - prediction[ :, 0, chain.br_2_position ], axis = 1
					) - 1.5
			).sum()
	objective += abs(
			norm(
					prediction[ :, 0, chain.br_2_position ] - prediction[ :, 0, chain.br_3_position ], axis = 1
					) - 1.5
			).sum()

	return objective


if __name__ == "__main__":

	ti = perf_counter()

	n_frames = 200
	tolerance = 1e-6
	max_number_of_iteration = 100
	time_step = 0.1

	model = ChainOf4()

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
	trajectory[ :, 0, model.br_3_z ] += -2. * sin( times / 6 )
	trajectory[ :, 0, model.br_3_z ] += + .2 * sin( times )
	trajectory[ :, 0, model.br_3_z ] += + .1 * sin( 3.3 * times )
	trajectory[ :, 0, model.br_3_position ][ :, 0 ] = 3.5

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
	pose_weight_matrix[ model.br_3_xy, model.br_3_xy ] *= 0.
	pose_weight_matrix[ model.br_3_z, model.br_3_z ] *= 0.
	pose_weight_matrix[ model.br_3_orientation, model.br_3_orientation ] *= 0.

	actuation_weight_matrix = eye( initial_actuation.shape[ 0 ] )
	actuation_weight_matrix[ model.br_0_linear_actuation, model.br_0_linear_actuation ] *= 1e-12
	actuation_weight_matrix[ model.br_0_angular_actuation, model.br_0_angular_actuation ] *= 1.
	actuation_weight_matrix[ model.br_1_linear_actuation, model.br_1_linear_actuation ] *= 1e-12
	actuation_weight_matrix[ model.br_1_angular_actuation, model.br_1_angular_actuation ] *= 1.
	actuation_weight_matrix[ model.br_2_linear_actuation, model.br_2_linear_actuation ] *= 1e-12
	actuation_weight_matrix[ model.br_2_angular_actuation, model.br_2_angular_actuation ] *= 1.
	actuation_weight_matrix[ model.br_3_linear_actuation, model.br_3_linear_actuation ] *= 1e-12
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
			# time_step = time_step * 2,
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
		dump( chain_mpc.__dict__, f, default = serialize_others )

	for frame in range( n_frames ):

		logger.log( f'frame {frame + 1}/{n_frames} starts at {perf_counter() - ti}' )

		chain_mpc.target_trajectory = trajectory[ frame + 1:frame + horizon + 1 ]
		chain_model.state[ model.br_3_pose ] = trajectory[ frame, 0, model.br_3_pose ]
		chain_model.state[ model.br_3_speed ] = trajectory_derivative[ frame, 0, model.br_3_pose ]

		chain_mpc.compute_actuation()
		chain_mpc.apply_result()
		chain_model.step()

		# try to recover if the optimization failed
		if (not chain_mpc.raw_result.success and chain_mpc.tolerance < 1):
			chain_mpc.tolerance *= 10
			logger.log( f'increasing tolerance: {chain_mpc.tolerance}' )
		elif (chain_mpc.raw_result.success and chain_mpc.tolerance > 2 * tolerance):  # *2
			# because of floating point error
			chain_mpc.tolerance /= 10
			logger.log( f'decreasing tolerance: {chain_mpc.tolerance}' )
		else:
			logger.log( f'keeping tolerance: {chain_mpc.tolerance}' )

		with open( f'{folder}/data/{frame}.json', 'w' ) as f:
			dump( chain_mpc.__dict__, f, default = serialize_others )

		logger.log( f'ends at {perf_counter() - ti}' )
		logger.log( f'{chain_mpc.raw_result.success}' )
		logger.log( f'{chain_model.state[ model.br_3_linear_speed ]}' )
		logger.log( f'{chain_model.state[ model.br_3_position ]}' )
		logger.lognl( '' )
		logger.save_at( folder )
