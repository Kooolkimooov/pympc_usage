from json import dump
from time import time

from numpy import array, cos, inf, linspace, pi, r_, sin
from numpy.linalg import norm

from bluerov import Bluerov, BluerovNoAngularActuation
from calc_catenary_from_ext_points import get_catenary_param
from mpc import *
from utils import check, generate_trajectory, Logger, serialize_others


class ChainOf43DoASurfaceEnd( Bluerov ):
	state_size = 48
	actuation_size = 9

	br0_pose = slice( 0, 6 )
	br0_position = slice( 0, 3 )
	br0_xy = slice( 0, 2 )
	br0_z = 2
	br0_orientation = slice( 3, 6 )

	br1_pose = slice( 6, 12 )
	br1_position = slice( 6, 9 )
	br1_xy = slice( 6, 8 )
	br1_z = 8
	br1_orientation = slice( 9, 12 )

	br2_pose = slice( 12, 18 )
	br2_position = slice( 12, 15 )
	br2_xy = slice( 12, 14 )
	br2_z = 14
	br2_orientation = slice( 15, 18 )

	brf_pose = slice( 18, 24 )
	brf_position = slice( 18, 21 )
	brf_xy = slice( 18, 20 )
	brf_z = 20
	brf_orientation = slice( 21, 24 )

	br0_speed = slice( 24, 30 )
	br0_linear_speed = slice( 24, 27 )
	br0_angular_speed = slice( 27, 30 )

	br1_speed = slice( 30, 36 )
	br1_linear_speed = slice( 30, 33 )
	br1_angular_speed = slice( 33, 36 )

	br2_speed = slice( 36, 42 )
	br2_linear_speed = slice( 36, 39 )
	br2_angular_speed = slice( 39, 42 )

	brf_speed = slice( 42, 48 )
	brf_linear_speed = slice( 42, 45 )
	brf_angular_speed = slice( 45, 48 )

	br0_actuation_start = 0
	br0_actuation = slice( 0, 3 )
	br0_linear_actuation = slice( 0, 3 )
	br0_angular_actuation = slice( 0, 0 )

	br1_actuation_start = 3
	br1_actuation = slice( 3, 6 )
	br1_linear_actuation = slice( 3, 6 )
	br1_angular_actuation = slice( 0, 0 )

	br2_actuation_start = 6
	br2_actuation = slice( 6, 9 )
	br2_linear_actuation = slice( 6, 9 )
	br2_angular_actuation = slice( 0, 0 )

	br0_state = r_[ br0_pose, br0_speed ]
	br1_state = r_[ br1_pose, br1_speed ]
	br2_state = r_[ br2_pose, br2_speed ]
	brf_state = r_[ brf_pose, brf_speed ]

	def __call__( self, state: ndarray, actuation: ndarray ) -> ndarray:
		"""
		evalutes the dynamics of each robot of the chain
		:param state: current state of the system
		:param actuation: current actuation of the system
		:return: state derivative of the system
		"""
		state_derivative = zeros( state.shape )

		state_derivative[ self.br0_state ] = BluerovNoAngularActuation.__call__(
				self, state[ self.br0_state ], actuation[ self.br0_actuation ]
				)
		state_derivative[ self.br1_state ] = BluerovNoAngularActuation.__call__(
				self, state[ self.br1_state ], actuation[ self.br1_actuation ]
				)
		state_derivative[ self.br2_state ] = BluerovNoAngularActuation.__call__(
				self, state[ self.br2_state ], actuation[ self.br2_actuation ]
				)
		state_derivative[ self.brf_state ] = BluerovNoAngularActuation.__call__(
				self, state[ self.brf_state ], zeros( (self.actuation_size // 3,) )
				)

		return state_derivative


def chain_of_three_moving_end_constraint( self: MPC, candidate ):

	system_class = self.model.model_dynamics

	actuation, actuation_derivatives = self.get_actuation( candidate )

	prediction = self.predict( actuation )
	prediction = prediction[ :, 0 ]

	# 3 constraints on cables (lowest points),
	# 6 on inter robot_distance (3 horizontal, 2 3d),
	# 3 on robot speed
	n_constraints = 3 + 6 + 3
	constraint = zeros( (chain_of_3_fixed_end_mpc.horizon, n_constraints) )

	# z position of robots 0 and 1; we add H afterward
	constraint[ :, 0 ] = prediction[ :, system_class.br0_z ]
	constraint[ :, 1 ] = prediction[ :, system_class.br1_z ]
	constraint[ :, 2 ] = prediction[ :, system_class.br2_z ]

	# horizontal distance between consecutive robots
	constraint[ :, 3 ] = norm(
			prediction[ :, system_class.br1_xy ] - prediction[ :, system_class.br0_xy ], axis = 1
			)
	constraint[ :, 4 ] = norm(
			prediction[ :, system_class.br2_xy ] - prediction[ :, system_class.br1_xy ], axis = 1
			)
	constraint[ :, 5 ] = norm(
			prediction[ :, system_class.brf_xy ] - prediction[ :, system_class.br2_xy ], axis = 1
			)

	# distance between consecutive robots
	constraint[ :, 6 ] = norm(
			prediction[ :, system_class.br1_position ] - prediction[ :, system_class.br0_position ], axis = 1
			)
	constraint[ :, 7 ] = norm(
			prediction[ :, system_class.br2_position ] - prediction[ :, system_class.br1_position ], axis = 1
			)
	constraint[ :, 8 ] = norm(
			prediction[ :, system_class.brf_position ] - prediction[ :, system_class.br2_position ], axis = 1
			)

	# speed
	constraint[ :, 9 ] = norm( prediction[ :, system_class.br0_speed ], axis = 1 )
	constraint[ :, 10 ] = norm( prediction[ :, system_class.br1_speed ], axis = 1 )
	constraint[ :, 11 ] = norm( prediction[ :, system_class.br2_speed ], axis = 1 )

	for i, state in enumerate( prediction ):

		try:
			_, _, H01 = get_catenary_param(
					state[ system_class.br0_z ] - state[ system_class.br1_z ], constraint[ i, 3 ], 3
					)
			_, _, H12 = get_catenary_param(
					state[ system_class.br1_z ] - state[ system_class.br2_z ], constraint[ i, 4 ], 3
					)
			_, _, H23 = get_catenary_param(
					state[ system_class.br2_z ] - state[ system_class.brf_z ], constraint[ i, 5 ], 3
					)
		except:
			H01 = 1.5
			H12 = 1.5
			H23 = 1.5

		constraint[ i, 0 ] += H01
		constraint[ i, 1 ] += H12
		constraint[ i, 2 ] += H23

	return constraint.flatten()


def chain_of_three_moving_end_objective( self: MPC, prediction: ndarray, actuation: ndarray ):

	system_class = self.model.model_dynamics

	objective = 0.

	objective += norm( prediction[ :, 0, system_class.br0_linear_speed ], axis = 1 ).sum()
	objective += norm( prediction[ :, 0, system_class.br1_linear_speed ], axis = 1 ).sum()
	objective += norm( prediction[ :, 0, system_class.br2_linear_speed ], axis = 1 ).sum()

	objective += abs(
			norm(
					prediction[ :, 0, system_class.br0_position ] - prediction[ :, 0, system_class.br1_position ], axis = 1
					) - 1.5
			).sum()
	objective += abs(
			norm(
					prediction[ :, 0, system_class.br1_position ] - prediction[ :, 0, system_class.br2_position ], axis = 1
					) - 1.5
			).sum()
	objective += abs(
			norm(
					prediction[ :, 0, system_class.br2_position ] - prediction[ :, 0, system_class.brf_position ], axis = 1
					) - 1.5
			).sum()

	return objective


if __name__ == "__main__":

	ti = perf_counter()

	n_frames = 200
	tolerance = 1e-6
	time_step = 0.1

	model = ChainOf43DoASurfaceEnd()

	initial_state = zeros( (model.state_size,) )
	initial_state[ model.br0_position ][ 0 ] = 2.
	initial_state[ model.br0_position ][ 2 ] = 1.
	initial_state[ model.br1_position ][ 0 ] = 2.5
	initial_state[ model.br1_position ][ 2 ] = 1.
	initial_state[ model.br2_position ][ 0 ] = 3.
	initial_state[ model.br2_position ][ 2 ] = 1.
	initial_state[ model.brf_position ][ 0 ] = 3.5

	initial_actuation = zeros( (model.actuation_size,) )

	horizon = 5
	time_steps_per_act = 5

	key_frames = [ (0., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 18), (.5, [ -3., 0., 0., 0., 0., 0. ] + [ 0. ] * 18),
								 (1., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 18), (2., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 18) ]

	trajectory = generate_trajectory( key_frames, 2 * n_frames )
	times = linspace( 0, trajectory.shape[ 0 ] * time_step, trajectory.shape[ 0 ] )
	trajectory[ :, 0, model.br0_z ] = 1.5 * cos( 1.25 * (trajectory[ :, 0, model.br0_position ][ :, 0 ] - 2) + pi ) + 2.5
	trajectory[ :, 0, model.brf_z ] = -2.5 * sin( times / 6 ) + .2 * sin( times ) + .1 * sin( 3.3 * times )
	trajectory[ :, 0, model.brf_position ][ :, 0 ] = 3.5

	trajectory_derivative = diff( trajectory, append = trajectory[ :1, :, : ], axis = 0 ) / time_step

	max_required_speed = (max( norm( diff( trajectory[ :, 0, :3 ], axis = 0 ), axis = 1 ) ) / time_step)

	# import matplotlib.pyplot as plt
	# plt.plot( trajectory[:, 0, model.brf_z] )
	# plt.plot( norm(diff(trajectory[:, 0, :3], axis=0), axis=1) / time_step )
	# plt.show()
	# exit()

	pose_weight_matrix = eye( initial_state.shape[ 0 ] // 2 )
	pose_weight_matrix[ model.br0_position, model.br0_position ] *= 10.
	pose_weight_matrix[ model.br0_orientation, model.br0_orientation ] *= 1.
	pose_weight_matrix[ model.br1_position, model.br1_position ] *= 0.
	pose_weight_matrix[ model.br1_orientation, model.br1_orientation ] *= 1.
	pose_weight_matrix[ model.br2_position, model.br2_position ] *= 0.
	pose_weight_matrix[ model.br2_orientation, model.br2_orientation ] *= 1.
	pose_weight_matrix[ model.brf_position, model.brf_position ] *= 0.
	pose_weight_matrix[ model.brf_orientation, model.brf_orientation ] *= 0.

	actuation_weight_matrix = eye( initial_actuation.shape[ 0 ] )
	actuation_weight_matrix[ model.br0_linear_actuation, model.br0_linear_actuation ] *= 0.
	actuation_weight_matrix[ model.br0_angular_actuation, model.br0_angular_actuation ] *= 1.
	actuation_weight_matrix[ model.br1_linear_actuation, model.br1_linear_actuation ] *= 0.
	actuation_weight_matrix[ model.br1_angular_actuation, model.br1_angular_actuation ] *= 1.
	actuation_weight_matrix[ model.br2_linear_actuation, model.br2_linear_actuation ] *= 0.
	actuation_weight_matrix[ model.br2_angular_actuation, model.br2_angular_actuation ] *= 1.

	final_cost_weight = 0.
	objective_weight = .1

	chain_of_3_fixed_end_model = Model(
			model, time_step, initial_state, initial_actuation, record = True
			)

	chain_of_3_fixed_end_mpc = MPC(
			chain_of_3_fixed_end_model,
			horizon,
			trajectory,
			optimize_on = 'actuation',
			objective_weight = objective_weight,
			# time_step = time_step * 2,
			tolerance = tolerance,
			max_iter = 100,
			time_steps_per_actuation = time_steps_per_act,
			pose_weight_matrix = pose_weight_matrix,
			actuation_derivative_weight_matrix = actuation_weight_matrix,
			final_weight = final_cost_weight,
			record = True,
			# verbose = True
			)

	chain_of_3_fixed_end_mpc.constraint_function = chain_of_three_moving_end_constraint.__get__(
			chain_of_3_fixed_end_mpc, MPC
			)

	chain_of_3_fixed_end_mpc.objective = chain_of_three_moving_end_objective.__get__(
			chain_of_3_fixed_end_mpc, MPC
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
			chain_of_3_fixed_end_mpc.constraint_function, array( lb ).flatten(), array( ub ).flatten()
			)
	constraint.labels = constraints_labels

	chain_of_3_fixed_end_mpc.constraints = (constraint,)

	previous_nfeval_record = [ 0 ]
	previous_H01_record = [ 0. ]
	previous_H12_record = [ 0. ]
	previous_H23_record = [ 0. ]

	folder = f'./export/three_robots_chain_with_moving_end_{int( time() )}'
	if check( folder ) + check( f'{folder}/data' ):
		exit()

	logger = Logger()

	with open( f'{folder}/config.json', 'w' ) as f:
		dump( chain_of_3_fixed_end_mpc.__dict__, f, default = serialize_others )

	for frame in range( n_frames ):

		logger.log( f'frame {frame + 1}/{n_frames} starts at {perf_counter() - ti}' )

		chain_of_3_fixed_end_mpc.target_trajectory = trajectory[ frame + 1:frame + horizon + 1 ]
		chain_of_3_fixed_end_model.state[ model.brf_pose ] = trajectory[ frame, 0, model.brf_pose ]
		chain_of_3_fixed_end_model.state[ model.brf_speed ] = trajectory_derivative[ frame, 0, model.brf_pose ]

		chain_of_3_fixed_end_mpc.compute_actuation()
		chain_of_3_fixed_end_mpc.apply_result()
		chain_of_3_fixed_end_model.step()

		# try to recover if the optimization failed
		if (not chain_of_3_fixed_end_mpc.raw_result.success and chain_of_3_fixed_end_mpc.tolerance < 1):
			chain_of_3_fixed_end_mpc.tolerance *= 10
			logger.log( f'increasing tolerance: {chain_of_3_fixed_end_mpc.tolerance}' )
		elif (chain_of_3_fixed_end_mpc.raw_result.success and chain_of_3_fixed_end_mpc.tolerance > 2 * tolerance):  # *2
			# because of floating point error
			chain_of_3_fixed_end_mpc.tolerance /= 10
			logger.log( f'decreasing tolerance: {chain_of_3_fixed_end_mpc.tolerance}' )
		else:
			logger.log( f'keeping tolerance: {chain_of_3_fixed_end_mpc.tolerance}' )

		with open( f'{folder}/data/{frame}.json', 'w' ) as f:
			dump( chain_of_3_fixed_end_mpc.__dict__, f, default = serialize_others )

		logger.log( f'ends at {perf_counter() - ti}' )
		logger.log( f'{chain_of_3_fixed_end_mpc.raw_result.success}' )
		logger.log( f'{chain_of_3_fixed_end_model.state[ model.brf_linear_speed ]}' )
		logger.log( f'{chain_of_3_fixed_end_model.state[ model.brf_position ]}' )
		logger.lognl( '' )
		logger.save_at( folder )
