from glob import glob
from json import dump
from os import mkdir, path, remove
from time import time

from numpy import array, cos, cross, diag, diff, inf, pi, sin, tan
from numpy.linalg import inv, norm

from calc_catenary_from_ext_points import *
from mpc import *
from utils import generate_trajectory, Logger, serialize_others


def compute_rotation_matrix( phi: float, theta: float, psi: float ) -> ndarray:
	cPhi, sPhi = cos( phi ), sin( phi )
	cTheta, sTheta, tTheta = cos( theta ), sin( theta ), tan( theta )
	cPsi, sPsi = cos( psi ), sin( psi )
	matrix = zeros( (6, 6) )
	matrix[ 0, :3 ] = array(
			[ cPsi * cTheta, -sPsi * cPhi + cPsi * sTheta * sPhi, sPsi * sPhi + cPsi * sTheta * cPhi ]
			)
	matrix[ 1, :3 ] = array(
			[ sPsi * cTheta, cPsi * cPhi + sPsi * sTheta * sPhi, -cPsi * sPhi + sPsi * sTheta * cPhi ]
			)
	matrix[ 2, :3 ] = array( [ -sTheta, cTheta * sPhi, cTheta * cPhi ] )
	matrix[ 3, 3: ] = array( [ 1, sPhi * tTheta, cPhi * tTheta ] )
	matrix[ 4, 3: ] = array( [ 0, cPhi, -sPhi ] )
	matrix[ 5, 3: ] = array( [ 0, sPhi / cTheta, cPhi / cTheta ] )
	return matrix


def build_inertial_matrix(
		mass: float, center_of_mass: ndarray, inertial_coefficients: list[ float ]
		):
	inertial_matrix = eye( 6 )
	for i in range( 3 ):
		inertial_matrix[ i, i ] = mass
		inertial_matrix[ i + 3, i + 3 ] = inertial_coefficients[ i ]
	inertial_matrix[ 0, 4 ] = mass * center_of_mass[ 2 ]
	inertial_matrix[ 0, 5 ] = - mass * center_of_mass[ 1 ]
	inertial_matrix[ 1, 3 ] = - mass * center_of_mass[ 2 ]
	inertial_matrix[ 1, 5 ] = mass * center_of_mass[ 0 ]
	inertial_matrix[ 2, 3 ] = mass * center_of_mass[ 1 ]
	inertial_matrix[ 2, 4 ] = - mass * center_of_mass[ 0 ]
	inertial_matrix[ 4, 0 ] = mass * center_of_mass[ 2 ]
	inertial_matrix[ 5, 0 ] = - mass * center_of_mass[ 1 ]
	inertial_matrix[ 3, 1 ] = - mass * center_of_mass[ 2 ]
	inertial_matrix[ 5, 1 ] = mass * center_of_mass[ 0 ]
	inertial_matrix[ 3, 2 ] = mass * center_of_mass[ 1 ]
	inertial_matrix[ 4, 2 ] = - mass * center_of_mass[ 0 ]
	inertial_matrix[ 3, 4 ] = - inertial_coefficients[ 3 ]
	inertial_matrix[ 3, 5 ] = - inertial_coefficients[ 4 ]
	inertial_matrix[ 4, 5 ] = - inertial_coefficients[ 5 ]
	inertial_matrix[ 4, 3 ] = - inertial_coefficients[ 3 ]
	inertial_matrix[ 5, 3 ] = - inertial_coefficients[ 4 ]
	inertial_matrix[ 5, 4 ] = - inertial_coefficients[ 5 ]

	return inertial_matrix


def compute_hydrostatic_force(
		weight: ndarray,
		buoyancy: ndarray,
		center_of_mass: ndarray,
		center_of_volume: ndarray,
		rotation_matrix
		) -> ndarray:
	force = zeros( 6 )
	force[ :3 ] = rotation_matrix.T @ (weight + buoyancy)
	force[ 3: ] = cross( center_of_mass, rotation_matrix.T @ weight ) + cross(
			center_of_volume, rotation_matrix.T @ buoyancy
			)

	return force


def three_robots_chain_with_fixed_end(
		state: ndarray,
		actuation: ndarray,
		weight: ndarray,
		buoyancy: ndarray,
		center_of_mass: ndarray,
		center_of_volume: ndarray,
		inverted_inertial_matrix: ndarray,
		hydrodynamic_matrix: ndarray
		) -> ndarray:
	"""
	:param state: state of the chain such that
	x = [	pose_robot_0_wf, pose_robot_1_wf, pose_robot_2_wf, pose_robot_3_wf, vel_robot_0_rf,
	vel_robot_1_rf, vel_robot_2_rf, vel_robot_3_rf]
		poses: [0:6-6:12-12:18-18:24]--speeds: [24:30-30:36-36:42-42:48]
	:param actuation: actuation of the chain such that
	u = [actuation_robot_0, actuation_robot_1, actuation_robot_2]
	:param hydrodynamic_matrix
	:param inverted_inertial_matrix
	:param center_of_volume
	:param center_of_mass
	:param buoyancy
	:param weight
	:return: ndarray: derivative of the state of the chain
	"""

	x0 = state[ :6 ]
	x0d = state[ 24:30 ]
	u0 = actuation[ :6 ]

	x1 = state[ 6:12 ]
	x1d = state[ 30:36 ]
	u1 = actuation[ 6:12 ]

	x2 = state[ 12:18 ]
	x2d = state[ 36:42 ]
	u2 = actuation[ 12:18 ]

	R0 = compute_rotation_matrix( *x0[ 3: ] )
	R1 = compute_rotation_matrix( *x1[ 3: ] )
	R2 = compute_rotation_matrix( *x2[ 3: ] )

	s0 = compute_hydrostatic_force(
			weight, buoyancy, center_of_mass, center_of_volume, R0[ :3, :3 ]
			)
	s1 = compute_hydrostatic_force(
			weight, buoyancy, center_of_mass, center_of_volume, R1[ :3, :3 ]
			)
	s2 = compute_hydrostatic_force(
			weight, buoyancy, center_of_mass, center_of_volume, R2[ :3, :3 ]
			)

	xdot = zeros( state.shape )

	xdot[ :6 ] = R0 @ x0d
	xdot[ 24:30 ] = inverted_inertial_matrix @ (hydrodynamic_matrix @ x0d + s0 + u0)

	xdot[ 6:12 ] = R1 @ x1d
	xdot[ 30:36 ] = inverted_inertial_matrix @ (hydrodynamic_matrix @ x1d + s1 + u1)

	xdot[ 12:18 ] = R2 @ x2d
	xdot[ 36:42 ] = inverted_inertial_matrix @ (hydrodynamic_matrix @ x2d + s2 + u2)

	return xdot


def three_robot_chain_objective( trajectory: ndarray, actuation: ndarray ):
	obj = 0.
	trajectory_derivative = diff( trajectory, axis = 0 )
	obj += norm( trajectory_derivative[ :, 0, 6:9 ], axis = 1 ).sum()
	obj += norm( trajectory_derivative[ :, 0, 12:15 ], axis = 1 ).sum()
	return obj


def constraint_f( candidate_actuation_derivative ):
	global three_bluerov_chain_mpc

	candidate_actuation = candidate_actuation_derivative.reshape(
			mpc_controller.result_shape
			).cumsum( axis = 0 ) + mpc_controller.model.actuation
	candidate_actuation = candidate_actuation.repeat(
			mpc_controller.time_steps_per_actuation, axis = 0
			)

	predicted_trajectory = mpc_controller.predict( candidate_actuation, with_speed = True )

	# 3 constraints on cables (lowest points),
	# 6 on inter robot_distance (3 horizontal, 2 3d),
	# 3 on robot speed
	n_constraints = 3 + 6 + 3
	constraint = zeros( (mpc_controller.horizon, n_constraints) )

	# z position of robots 0 and 1; we add H afterward
	constraint[ :, 0 ] = predicted_trajectory[ :, 0, 2 ]
	constraint[ :, 1 ] = predicted_trajectory[ :, 0, 8 ]
	constraint[ :, 2 ] = predicted_trajectory[ :, 0, 14 ]

	# horizontal distance between consecutive robots
	constraint[ :, 3 ] = norm(
			predicted_trajectory[ :, 0, 6:8 ] - predicted_trajectory[ :, 0, 0:2 ], axis = 1
			)
	constraint[ :, 4 ] = norm(
			predicted_trajectory[ :, 0, 12:14 ] - predicted_trajectory[ :, 0, 6:8 ], axis = 1
			)
	constraint[ :, 5 ] = norm(
			predicted_trajectory[ :, 0, 18:20 ] - predicted_trajectory[ :, 0, 12:14 ], axis = 1
			)

	# distance between consecutive robots
	constraint[ :, 6 ] = norm(
			predicted_trajectory[ :, 0, 6:9 ] - predicted_trajectory[ :, 0, 0:3 ], axis = 1
			)
	constraint[ :, 7 ] = norm(
			predicted_trajectory[ :, 0, 12:15 ] - predicted_trajectory[ :, 0, 6:9 ], axis = 1
			)
	constraint[ :, 8 ] = norm(
			predicted_trajectory[ :, 0, 18:21 ] - predicted_trajectory[ :, 0, 12:15 ], axis = 1
			)

	# speed
	constraint[ :, 9 ] = norm( predicted_trajectory[ :, 0, 24:27 ], axis = 1 )
	constraint[ :, 10 ] = norm( predicted_trajectory[ :, 0, 30:33 ], axis = 1 )
	constraint[ :, 11 ] = norm( predicted_trajectory[ :, 0, 36:39 ], axis = 1 )

	for i, state in enumerate( predicted_trajectory[ :, 0 ] ):

		try:
			_, _, H01 = get_catenary_param( state[ 2 ] - state[ 8 ], constraint[ i, 3 ], 3 )
			_, _, H12 = get_catenary_param( state[ 8 ] - state[ 14 ], constraint[ i, 4 ], 3 )
			_, _, H23 = get_catenary_param( state[ 14 ] - state[ 20 ], constraint[ i, 5 ], 3 )
		except:
			H01 = 1.5
			H12 = 1.5
			H23 = 1.5

		constraint[ i, 0 ] += H01
		constraint[ i, 1 ] += H12
		constraint[ i, 2 ] += H23

	return constraint.flatten()


if __name__ == "__main__":

	ti = perf_counter()

	n_frames = 2000
	tolerance = 1e-4
	time_step = 0.01
	n_robots = 4
	state = zeros( (12 * n_robots,) )
	state[ 0 ] = 2.
	state[ 6 ] = 2.5
	state[ 12 ] = 3.
	state[ 18 ] = 3.5
	state[ 20 ] = -1.
	actuation = zeros( (6 * (n_robots - 1),) )
	area = array( [ [ -3, 3 ], [ -3, 3 ], [ -2, 4 ] ] )
	max_actuation_x = 300.
	max_actuation_t = 1.
	floor_depth = 3.00001

	horizon = 25
	time_steps_per_act = 25

	key_frames = [ (0., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * (n_robots - 1) * 6),
								 (.5, [ -3., 0., 0., 0., 0., 0. ] + [ 0. ] * (n_robots - 1) * 6),
								 (1., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * (n_robots - 1) * 6),
								 (2., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * (n_robots - 1) * 6) ]

	trajectory = generate_trajectory( key_frames, 2 * n_frames )
	trajectory[ :, 0, 2 ] = 1.5 * cos( 1.25 * (trajectory[ :, 0, 0 ] - 2) + pi ) + 1.5

	max_required_speed = (
			max( norm( diff( trajectory[ :, 0, :3 ], axis = 0 ), axis = 1 ) ) / time_step)
	print( f'{max_required_speed=}' )

	# plt.plot( trajectory[:, 0, 2] )
	# plt.plot( norm(diff(trajectory[:, 0, :3], axis=0), axis=1) / time_step )
	# plt.show()
	# exit()

	model_kwargs = {
			"weight"                  : 11.5 * array( [ 0., 0., 9.81 ] ),
			"buoyancy"                : 120. * array( [ 0., 0., -1. ] ),
			"center_of_mass"          : array( [ 0., 0., 0. ] ),
			"center_of_volume"        : array( [ 0., 0., - 0.02 ] ),
			"inverted_inertial_matrix": inv(
					build_inertial_matrix( 11.5, array( [ 0., 0., 0. ] ), [ .16, .16, .16, 0.0, 0.0, 0.0 ] )
					),
			"hydrodynamic_matrix"     : diag( array( [ 4.03, 6.22, 5.18, 0.07, 0.07, 0.07 ] ) )
			}

	pose_weight_matrix = eye( state.shape[ 0 ] // 2 )
	pose_weight_matrix[ 0:3, 0:3 ] *= 10.
	pose_weight_matrix[ 3:6, 3:6 ] *= 5.
	pose_weight_matrix[ 6:9, 6:9 ] *= 0.
	pose_weight_matrix[ 9:12, 9:12 ] *= 5.
	pose_weight_matrix[ 12:15, 12:15 ] *= 0.
	pose_weight_matrix[ 15:18, 15:18 ] *= 5.
	pose_weight_matrix[ 18:21, 18:21 ] *= 0.
	pose_weight_matrix[ 21:24, 21:24 ] *= 0.

	actuation_weight_matrix = eye( actuation.shape[ 0 ] )
	actuation_weight_matrix[ 0:3, 0:3 ] *= 0.
	actuation_weight_matrix[ 3:6, 3:6 ] *= 1000.
	actuation_weight_matrix[ 6:9, 6:9 ] *= 0.
	actuation_weight_matrix[ 9:12, 9:12 ] *= 1000.
	actuation_weight_matrix[ 12:15, 12:15 ] *= 0.
	actuation_weight_matrix[ 15:18, 15:18 ] *= 1000.

	final_cost_weight = 10.

	bluerov_chain = Model(
			three_robots_chain_with_fixed_end,
			time_step,
			state,
			actuation,
			kwargs = model_kwargs,
			record = True
			)

	three_bluerov_chain_mpc = MPC(
			bluerov_chain,
			horizon,
			trajectory,
			objective = three_robot_chain_objective,
			objective_weight = 10.,
			tolerance = tolerance,
			time_steps_per_actuation = time_steps_per_act,
			pose_weight_matrix = pose_weight_matrix,
			actuation_derivative_weight_matrix = actuation_weight_matrix,
			final_weight = final_cost_weight,
			record = True
			)

	# mpc_controller.verbose = True

	three_bluerov_chain_mpc.bounds = Bounds(
			array( [ [ -20, -20, -20, -.1, -.1, -.1 ] ] ).repeat( n_robots-1, axis = 0 ).flatten(),
			array( [ [ 20, 20, 20, .1, .1, .1 ] ] ).repeat( n_robots-1, axis = 0 ).flatten()
			)

	dp_lb = 0.4
	dp_ub = 2.8
	dr_lb = -inf
	dr_ub = 2.8
	v_lb = -inf
	v_ub = 3.

	# -----0---1---2---3----4----5----6----7----8----9--10-11
	# -----H01-H12-H23-dp01-dp12-dp23-dr01-dr12-dr23-v0-v1-v2

	lb_base = [ -inf, -inf, -inf, dp_lb, dp_lb, dp_lb, dr_lb, dr_lb, dr_lb, v_lb, v_lb, v_lb ]
	ub_base = [ floor_depth, floor_depth, floor_depth, dp_ub, dp_ub, dp_ub, dr_ub, dr_ub, dr_ub,
							v_ub,
							v_ub, v_ub ]

	lb = [ lb_base ] * horizon
	ub = [ ub_base ] * horizon

	three_bluerov_chain_mpc.constraints = (
			NonlinearConstraint( constraint_f, array( lb ).flatten(), array( ub ).flatten() ),)

	previous_nfeval_record = [ 0 ]
	previous_H01_record = [ 0. ]
	previous_H12_record = [ 0. ]
	previous_H23_record = [ 0. ]

	logger = Logger( False )

	folder = (f'./plots/{three_robots_chain_with_fixed_end.__name__}_'
						f'{int( time() )}')

	print( folder )

	if path.exists( folder ):
		files_in_dir = glob( f'{folder}/*' )
		if len( files_in_dir ) > 0:
			if input( f"{folder} contains data. Remove? (y/n) " ) == 'y':
				for fig in files_in_dir:
					remove( fig )
			else:
				exit()
	else:
		mkdir( folder )

	with open( f'{folder}/config.json', 'w' ) as f:
		dump( bluerov_chain.__dict__ | three_bluerov_chain_mpc.__dict__, f, default = serialize_others )

	logger.log( "index" )
	logger.log( "sim_time" )
	logger.log( "step_time" )
	logger.log( "success" )
	logger.log( "C01" )
	logger.log( "C12" )
	logger.log( "C23" )
	logger.log( "D01" )
	logger.log( "D12" )
	logger.log( "D23" )
	logger.log( "H01" )
	logger.log( "H12" )
	logger.log( "H23" )
	logger.log( "state_r0" )
	logger.log( "state_r1" )
	logger.log( "state_r2" )
	logger.log( "state_r3" )
	logger.log( "speed_r0" )
	logger.log( "speed_r1" )
	logger.log( "speed_r2" )
	logger.log( "speed_r3" )
	logger.log( "actuation_r0" )
	logger.log( "actuation_r1" )
	logger.log( "actuation_r2" )
	logger.log( "objective" )
	logger.lognl( "" )

	for frame in range( n_frames ):

		three_bluerov_chain_mpc.target_trajectory = trajectory[ frame + 1:frame + horizon + 1 ]

		three_bluerov_chain_mpc.compute_actuation()
		three_bluerov_chain_mpc.apply_result()
		bluerov_chain.step()

		if not three_bluerov_chain_mpc.raw_result.success and three_bluerov_chain_mpc.tolerance < 1:
			three_bluerov_chain_mpc.tolerance *= 10
		elif three_bluerov_chain_mpc.raw_result.success and three_bluerov_chain_mpc.tolerance > tolerance:
			three_bluerov_chain_mpc.tolerance /= 10

		try:
			C01, D01, H01 = get_catenary_param(
					bluerov_chain.state[ 2 ] - bluerov_chain.state[ 8 ],
					norm( bluerov_chain.state[ 0:2 ] - bluerov_chain.state[ 6:8 ] ),
					3
					)
			C12, D12, H12 = get_catenary_param(
					bluerov_chain.state[ 8 ] - bluerov_chain.state[ 14 ],
					norm( bluerov_chain.state[ 6:8 ] - bluerov_chain.state[ 12:14 ] ),
					3
					)
			C23, D23, H23 = get_catenary_param(
					bluerov_chain.state[ 14 ] - bluerov_chain.state[ 20 ],
					norm( bluerov_chain.state[ 12:14 ] - bluerov_chain.state[ 18:20 ] ),
					3
					)
		except:
			C01 = None
			C12 = None
			C23 = None
			D01 = None
			D12 = None
			D23 = None
			H01 = None
			H12 = None
			H23 = None

		logger.log( f"{frame}" )
		logger.log( f"{perf_counter() - ti:.6f}" )
		logger.log( f"{three_bluerov_chain_mpc.times[ -1 ]:.6f}" )
		logger.log( f"{three_bluerov_chain_mpc.raw_result.success}" )
		logger.log( f"{C01}" )
		logger.log( f"{C12}" )
		logger.log( f"{C23}" )
		logger.log( f"{D01}" )
		logger.log( f"{D12}" )
		logger.log( f"{D23}" )
		logger.log( f"{H01}" )
		logger.log( f"{H12}" )
		logger.log( f"{H23}" )
		logger.log( f"{[ float( v ) for v in bluerov_chain.state[ 0:6 ] ]}" )
		logger.log( f"{[ float( v ) for v in bluerov_chain.state[ 6:12 ] ]}" )
		logger.log( f"{[ float( v ) for v in bluerov_chain.state[ 12:18 ] ]}" )
		logger.log( f"{[ float( v ) for v in bluerov_chain.state[ 18:24 ] ]}" )
		logger.log( f"{[ float( v ) for v in bluerov_chain.state[ 24:30 ] ]}" )
		logger.log( f"{[ float( v ) for v in bluerov_chain.state[ 30:36 ] ]}" )
		logger.log( f"{[ float( v ) for v in bluerov_chain.state[ 36:42 ] ]}" )
		logger.log( f"{[ float( v ) for v in bluerov_chain.state[ 42:48 ] ]}" )
		logger.log( f"{[ float( v ) for v in bluerov_chain.actuation[ 0:6 ] ]}" )
		logger.log( f"{[ float( v ) for v in bluerov_chain.actuation[ 6:12 ] ]}" )
		logger.log( f"{[ float( v ) for v in bluerov_chain.actuation[ 12:18 ] ]}" )
		logger.log(
				f"{three_bluerov_chain_mpc.objective_weight * three_robot_chain_objective(
						three_bluerov_chain_mpc.predict(
								(three_bluerov_chain_mpc.result.cumsum( axis = 0 ) + three_bluerov_chain_mpc.model.actuation).repeat( three_bluerov_chain_mpc.time_steps_per_actuation, axis = 0 )
								), three_bluerov_chain_mpc.result, )}"
				)
		logger.lognl( "" )
		logger.save_at( folder )

		print(
				f"{frame}/{n_frames} - {perf_counter() - ti:.6f} - {three_bluerov_chain_mpc.times[ -1 ]:.6f} - "
				f"{three_bluerov_chain_mpc.tolerance} - {three_bluerov_chain_mpc.raw_result.success}"
				)
