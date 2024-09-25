from json import dump
from time import time

from numpy import array, cos, inf, pi, r_
from numpy.linalg import norm

from calc_catenary_from_ext_points import *
from mpc import *
from three_bluerov_chain import ChainOf3, ChainOf33DoA
from utils import check, generate_trajectory, Logger, serialize_others


class ChainOf3FixedEnd( ChainOf3 ):
	state_size = 48

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

	br0_state = r_[ ChainOf3.br0_pose, br0_speed ]
	br1_state = r_[ ChainOf3.br1_pose, br1_speed ]
	br2_state = r_[ ChainOf3.br2_pose, br2_speed ]
	brf_state = r_[ brf_pose, brf_speed ]

	def __call__( self, state: ndarray, actuation: ndarray ) -> ndarray:
		"""
		evalutes the dynamics of each robot of the chain
		:param state: current state of the system
		:param actuation: current actuation of the system
		:return: state derivative of the system
		"""
		return ChainOf3.__call__( self, state, actuation )


class ChainOf33DoAFixedEnd( ChainOf3FixedEnd, ChainOf33DoA ):
	actuation_size = 9

	br0_actuation_start = 0
	br0_actuation = slice( 0, 3 )
	br0_linear_actuation = slice( 0, 3 )

	br1_actuation_start = 3
	br1_actuation = slice( 3, 6 )
	br1_linear_actuation = slice( 3, 6 )

	br2_actuation_start = 6
	br2_actuation = slice( 6, 9 )
	br2_linear_actuation = slice( 6, 9 )

	def __call__( self, state: ndarray, actuation: ndarray ) -> ndarray:
		"""
		evalutes the dynamics of each robot of the chain
		:param state: current state of the system
		:param actuation: current actuation of the system
		:return: state derivative of the system
		"""
		return ChainOf33DoA.__call__( self, state, actuation )


def chain_of_three_fixed_end_constraint( candidate ):
	global three_bluerov_chain_with_fixed_end_mpc

	actuation, actuation_derivatives = three_bluerov_chain_with_fixed_end_mpc.get_actuation( candidate )

	prediction = three_bluerov_chain_with_fixed_end_mpc.predict( actuation )
	prediction = prediction[ :, 0 ]

	# 3 constraints on cables (lowest points),
	# 6 on inter robot_distance (3 horizontal, 2 3d),
	# 3 on robot speed
	n_constraints = 3 + 6 + 3
	constraint = zeros( (three_bluerov_chain_with_fixed_end_mpc.horizon, n_constraints) )

	# z position of robots 0 and 1; we add H afterward
	constraint[ :, 0 ] = prediction[ :, ChainOf3FixedEnd.br0_z ]
	constraint[ :, 1 ] = prediction[ :, ChainOf3FixedEnd.br1_z ]
	constraint[ :, 2 ] = prediction[ :, ChainOf3FixedEnd.br2_z ]

	# horizontal distance between consecutive robots
	constraint[ :, 3 ] = norm(
			prediction[ :, ChainOf3FixedEnd.br1_xy ] - prediction[ :, ChainOf3FixedEnd.br0_xy ], axis = 1
			)
	constraint[ :, 4 ] = norm(
			prediction[ :, ChainOf3FixedEnd.br2_xy ] - prediction[ :, ChainOf3FixedEnd.br1_xy ], axis = 1
			)
	constraint[ :, 5 ] = norm(
			prediction[ :, ChainOf3FixedEnd.brf_xy ] - prediction[ :, ChainOf3FixedEnd.br2_xy ], axis = 1
			)

	# distance between consecutive robots
	constraint[ :, 6 ] = norm(
			prediction[ :, ChainOf3FixedEnd.br1_position ] - prediction[ :, ChainOf3FixedEnd.br0_position ], axis = 1
			)
	constraint[ :, 7 ] = norm(
			prediction[ :, ChainOf3FixedEnd.br2_position ] - prediction[ :, ChainOf3FixedEnd.br1_position ], axis = 1
			)
	constraint[ :, 8 ] = norm(
			prediction[ :, ChainOf3FixedEnd.brf_position ] - prediction[ :, ChainOf3FixedEnd.br2_position ], axis = 1
			)

	# speed
	constraint[ :, 9 ] = norm( prediction[ :, ChainOf3FixedEnd.br0_speed ], axis = 1 )
	constraint[ :, 10 ] = norm( prediction[ :, ChainOf3FixedEnd.br1_speed ], axis = 1 )
	constraint[ :, 11 ] = norm( prediction[ :, ChainOf3FixedEnd.br2_speed ], axis = 1 )

	for i, state in enumerate( prediction ):

		try:
			_, _, H01 = get_catenary_param(
					state[ ChainOf3FixedEnd.br0_z ] - state[ ChainOf3FixedEnd.br1_z ], constraint[ i, 3 ], 3
					)
			_, _, H12 = get_catenary_param(
					state[ ChainOf3FixedEnd.br1_z ] - state[ ChainOf3FixedEnd.br2_z ], constraint[ i, 4 ], 3
					)
			_, _, H23 = get_catenary_param(
					state[ ChainOf3FixedEnd.br2_z ] - state[ ChainOf3FixedEnd.brf_z ], constraint[ i, 5 ], 3
					)
		except:
			H01 = 1.5
			H12 = 1.5
			H23 = 1.5

		constraint[ i, 0 ] += H01
		constraint[ i, 1 ] += H12
		constraint[ i, 2 ] += H23

	return constraint.flatten()


def chain_of_three_3_doa_fixed_end_objective( prediction: ndarray, actuation: ndarray ):
	objective = 0.

	objective += norm( prediction[ :, 0, ChainOf33DoAFixedEnd.br0_linear_speed ], axis = 1 ).sum()
	objective += norm( prediction[ :, 0, ChainOf33DoAFixedEnd.br1_linear_speed ], axis = 1 ).sum()
	objective += norm( prediction[ :, 0, ChainOf33DoAFixedEnd.br2_linear_speed ], axis = 1 ).sum()

	objective += abs(
			norm(
					prediction[ :, 0, ChainOf33DoAFixedEnd.br0_position ] - prediction[ :, 0,
																																	ChainOf33DoAFixedEnd.br1_position ],
					axis = 1
					) - 1.5
			).sum()
	objective += abs(
			norm(
					prediction[ :, 0, ChainOf33DoAFixedEnd.br1_position ] - prediction[ :, 0,
																																	ChainOf33DoAFixedEnd.br2_position ],
					axis = 1
					) - 1.5
			).sum()
	objective += abs(
			norm(
					prediction[ :, 0, ChainOf33DoAFixedEnd.br2_position ] - prediction[ :, 0,
																																	ChainOf33DoAFixedEnd.brf_position ],
					axis = 1
					) - 1.5
			).sum()

	return objective


if __name__ == "__main__":

	ti = perf_counter()

	n_frames = 200
	tolerance = 1e-6
	time_step = 0.1

	model = ChainOf33DoAFixedEnd()

	initial_state = zeros( (model.state_size,) )
	initial_state[ 0 ] = 2.
	initial_state[ 2 ] = 1.
	initial_state[ 6 ] = 2.5
	initial_state[ 8 ] = 1.
	initial_state[ 12 ] = 3.
	initial_state[ 14 ] = 1.
	initial_state[ 18 ] = 3.5

	initial_actuation = zeros( (model.actuation_size,) )

	horizon = 5
	time_steps_per_act = 5

	key_frames = [ (0., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 18), (.5, [ -3., 0., 0., 0., 0., 0. ] + [ 0. ] * 18),
								 (1., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 18), (2., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 18) ]

	trajectory = generate_trajectory( key_frames, 2 * n_frames )
	trajectory[ :, 0, 2 ] = 1.5 * cos( 1.25 * (trajectory[ :, 0, 0 ] - 2) + pi ) + 2.5

	max_required_speed = (max( norm( diff( trajectory[ :, 0, :3 ], axis = 0 ), axis = 1 ) ) / time_step)

	# plt.plot( trajectory[:, 0, 2] )
	# plt.plot( norm(diff(trajectory[:, 0, :3], axis=0), axis=1) / time_step )
	# plt.show()
	# exit()

	pose_weight_matrix = eye( initial_state.shape[ 0 ] // 2 )
	pose_weight_matrix[ model.br0_position, model.br0_position ] *= 500.
	pose_weight_matrix[ model.br0_orientation, model.br0_orientation ] *= 5.
	pose_weight_matrix[ model.br1_position, model.br1_position ] *= 0.
	pose_weight_matrix[ model.br1_orientation, model.br1_orientation ] *= 5.
	pose_weight_matrix[ model.br2_position, model.br2_position ] *= 0.
	pose_weight_matrix[ model.br2_orientation, model.br2_orientation ] *= 5.
	pose_weight_matrix[ model.brf_position, model.brf_position ] *= 0.
	pose_weight_matrix[ model.brf_orientation, model.brf_orientation ] *= 0.

	actuation_weight_matrix = eye( initial_actuation.shape[ 0 ] )
	actuation_weight_matrix[ model.br0_linear_actuation, model.br0_linear_actuation ] *= 0.
	actuation_weight_matrix[ model.br0_angular_actuation, model.br0_angular_actuation ] *= 1000.
	actuation_weight_matrix[ model.br1_linear_actuation, model.br1_linear_actuation ] *= 0.
	actuation_weight_matrix[ model.br1_angular_actuation, model.br1_angular_actuation ] *= 1000.
	actuation_weight_matrix[ model.br2_linear_actuation, model.br2_linear_actuation ] *= 0.
	actuation_weight_matrix[ model.br2_angular_actuation, model.br2_angular_actuation ] *= 1000.

	final_cost_weight = 0.
	objective_weight = 100.

	three_bluerov_chain_with_fixed_end_model = Model(
			model, time_step, initial_state, initial_actuation, record = True
			)

	three_bluerov_chain_with_fixed_end_mpc = MPC(
			three_bluerov_chain_with_fixed_end_model,
			horizon,
			trajectory,
			optimize_on = 'actuation',
			objective = chain_of_three_3_doa_fixed_end_objective,
			objective_weight = objective_weight,
			# time_step = time_step * 2,
			tolerance = tolerance,
			time_steps_per_actuation = time_steps_per_act,
			pose_weight_matrix = pose_weight_matrix,
			actuation_derivative_weight_matrix = actuation_weight_matrix,
			final_weight = final_cost_weight,
			record = True,
			# verbose = True
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

	constraint = NonlinearConstraint( chain_of_three_fixed_end_constraint, array( lb ).flatten(), array( ub ).flatten() )
	constraint.labels = constraints_labels

	three_bluerov_chain_with_fixed_end_mpc.constraints = (constraint,)

	previous_nfeval_record = [ 0 ]
	previous_H01_record = [ 0. ]
	previous_H12_record = [ 0. ]
	previous_H23_record = [ 0. ]

	folder = f'./export/three_robots_chain_with_fixed_end_{int( time() )}'
	if check( folder ) + check( f'{folder}/data' ):
		exit()

	logger = Logger()

	with open( f'{folder}/config.json', 'w' ) as f:
		dump( three_bluerov_chain_with_fixed_end_mpc.__dict__, f, default = serialize_others )

	for frame in range( n_frames ):

		logger.log( f'frame {frame + 1}/{n_frames} starts at {perf_counter() - ti}' )

		three_bluerov_chain_with_fixed_end_mpc.target_trajectory = trajectory[ frame + 1:frame + horizon + 1 ]

		three_bluerov_chain_with_fixed_end_mpc.compute_actuation()
		three_bluerov_chain_with_fixed_end_mpc.apply_result()
		three_bluerov_chain_with_fixed_end_model.step()

		# try to recover if the optimization failed
		if (
				not three_bluerov_chain_with_fixed_end_mpc.raw_result.success and
				three_bluerov_chain_with_fixed_end_mpc.tolerance < 1):
			three_bluerov_chain_with_fixed_end_mpc.tolerance *= 10
			logger.log( 'increasing tolerance' )
		elif (
				three_bluerov_chain_with_fixed_end_mpc.raw_result.success and three_bluerov_chain_with_fixed_end_mpc.tolerance
				> tolerance):
			three_bluerov_chain_with_fixed_end_mpc.tolerance /= 10
			logger.log( 'decreasing tolerance' )

		with open( f'{folder}/data/{frame}.json', 'w' ) as f:
			dump( three_bluerov_chain_with_fixed_end_mpc.__dict__, f, default = serialize_others )

		logger.log( f'ends at {perf_counter() - ti}' )
		logger.log( f'{three_bluerov_chain_with_fixed_end_mpc.raw_result.success}' )
		logger.log( f'{ three_bluerov_chain_with_fixed_end_model.actuation[ChainOf33DoAFixedEnd.br2_actuation] }' )
		logger.lognl( '' )
		logger.save_at( folder )
