from glob import glob
from json import dump
from os import mkdir, path, remove
from time import perf_counter, time

from cycler import cycler
from numpy import array, concatenate, cos, cross, diag, eye, meshgrid, ones, sin, tan
from numpy.linalg import inv
from PIL.Image import Image
from scipy.spatial.transform import Rotation

from calc_catenary_from_ext_points import get_catenary_param, get_coor_marker_points_ideal_catenary
from mpc import *


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


def three_robots_chain(
		x: ndarray,
		u: ndarray,
		weight: ndarray,
		buoyancy: ndarray,
		center_of_mass: ndarray,
		center_of_volume: ndarray,
		inverted_inertial_matrix: ndarray,
		hydrodynamic_matrix: ndarray
		) -> ndarray:
	"""
	:param x: state of the chain such that x = [pose_robot_0_wf, pose_robot_1_wf, vel_robot_0_rf,
	vel_robot_1_rf]
	:param u: actuation of the chain such that u = [actuation_robot_0, actuation_robot_1]
	:param hydrodynamic_matrix:
	:param inverted_inertial_matrix:
	:param center_of_volume:
	:param center_of_mass:
	:param buoyancy:
	:param weight:
	:return: xdot: derivative of the state of the chain
	"""

	x0 = x[ :6 ]
	x0d = x[ 18:24 ]
	u0 = u[ :6 ]

	x1 = x[ 6:12 ]
	x1d = x[ 24:30 ]
	u1 = u[ 6:12 ]

	x2 = x[ 12:18 ]
	x2d = x[ 30:36 ]
	u2 = u[ 12:18 ]

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

	xdot = zeros( x.shape )

	xdot[ :6 ] = R0 @ x0d
	xdot[ 18:24 ] = inverted_inertial_matrix @ (hydrodynamic_matrix @ x0d + s0 + u0)

	xdot[ 6:12 ] = R1 @ x1d
	xdot[ 24:30 ] = inverted_inertial_matrix @ (hydrodynamic_matrix @ x1d + s1 + u1)

	xdot[ 12:18 ] = R2 @ x2d
	xdot[ 30:36 ] = inverted_inertial_matrix @ (hydrodynamic_matrix @ x2d + s2 + u2)

	return xdot


def three_robot_chain_objective(
		x: ndarray,
		u: ndarray,
		weight: ndarray,
		buoyancy: ndarray,
		center_of_mass: ndarray,
		center_of_volume: ndarray,
		inverted_inertial_matrix: ndarray,
		hydrodynamic_matrix: ndarray
		) -> ndarray:
	"""
	:param x: state of the chain such that
	x = [pose_robot_0_wf, pose_robot_1_wf, pose_robot2_wf, vel_robot_0_rf, vel_robot_1_rf,
	vel_robot_2_rf]
	:param u: actuation of the chain such that u = [actuation_robot_0, actuation_robot_1]
	:param hydrodynamic_matrix:
	:param inverted_inertial_matrix:
	:param center_of_volume:
	:param center_of_mass:
	:param buoyancy:
	:param weight:
	:return: objective to minimize
	"""
	# # 													ideal distance is .5m horizontally
	# dist01 = x[ :3 ] - x[ 6:9 ] - .5 * array( [ 1., 1., 0. ] )
	# dist12 = x[ 6:9 ] - x[ 12:15 ] - .5 * array( [ 1., 1., 0. ] )
	# return dist01 @ eye( 3 ) @ dist01.T + dist12 @ eye( 3 ) @ dist12.T

	floor_depth = 1.5

	dp01 = norm( x[ 6:8 ] - x[ 0:2 ] )
	d01 = norm( x[ 6:9 ] - x[ 0:3 ])
	dz01 = x[ 8 ] - x[ 2 ]
	dp12 = norm( x[ 12:14 ] - x[ 6:8 ] )
	d12 = norm( x[ 12:15 ] - x[ 6:9 ] )
	dz12 = x[ 14 ] - x[ 8 ]

	_, _, H01 = get_catenary_param( dz01, dp01, 3 )
	_, _, H12 = get_catenary_param( dz12, dp12, 3 )

	objective01 = 1. / abs(H01 + x[ 8 ] - floor_depth)
	objective12 = 1. / abs(H12 + x[ 14 ] - floor_depth)

	objective1 = 1 / abs(d01) + 1 / abs(d01 - 3.)
	objective2 = 1 / abs(d12) + 1 / abs(d12 - 3.)

	return objective01 + objective12 + objective1 + objective2


if __name__ == "__main__":

	n_frames = 300
	max_iter = 1000
	tolerance = 1e-3
	time_step = 0.025
	sequence_time = n_frames * time_step
	n_robots = 3
	state = zeros( (12 * n_robots,) )
	state[ 6 ] = 1
	state[ 12 ] = 2
	actuation = zeros( (6 * n_robots,) )

	base_optimization_horizon = 25
	optimization_horizon = base_optimization_horizon
	time_steps_per_actuation = 25
	result_shape = (optimization_horizon // time_steps_per_actuation + (
			1 if optimization_horizon % time_steps_per_actuation != 0 else 0), actuation.shape[ 0 ])
	result = zeros( result_shape )

	key_frames = [

			(time_step * 0.0 * n_frames, [ 0., 0., 0., 0., 0., 0. ] + [ 0. ] * 12),
			(time_step * .2 * n_frames, [ 0., 0., 1., 0., 0., 0. ] + [ 0. ] * 12),
			(time_step * .4 * n_frames, [ 0., 0., 1., 0., 0., -pi ] + [ 0. ] * 12),
			(time_step * .6 * n_frames, [ 1., -1., 1., 0., 0., -pi ] + [ 0. ] * 12),
			(time_step * .8 * n_frames, [ 1., -1., 0., 0., 0., -pi ] + [ 0. ] * 12),
			(time_step * 1.0 * n_frames, [ 0., 0., 0., 0., 0., 0. ] + [ 0. ] * 12),
			(time_step * (n_frames + optimization_horizon + 1), [ 0., 0., 0., 0., 0., 0. ] + [ 0. ] * 12)

			]

	trajectory = generate_trajectory( key_frames, n_frames )

	model_kwargs = {
			"weight"                  : 11.5 * array( [ 0., 0., - 9.81 ] ),
			"buoyancy"                : 120. * array( [ 0., 0., 1. ] ),
			"center_of_mass"          : array( [ 0., 0., 0. ] ),
			"center_of_volume"        : array( [ 0., 0., - 0.02 ] ),
			"inverted_inertial_matrix": inv(
					build_inertial_matrix( 11.5, array( [ 0., 0., 0. ] ), [ .16, .16, .16, 0.0, 0.0, 0.0 ] )
					),
			"hydrodynamic_matrix"     : diag( array( [ 4.03, 6.22, 5.18, 0.07, 0.07, 0.07 ] ) )
			}

	pose_weight_matrix = eye( actuation.shape[ 0 ] )
	pose_weight_matrix[ :3, :3 ] *= 2.
	# ignore error for robot1 and robot2, only consider robot0, the objective will determine robot1
	pose_weight_matrix[ 6:, 6: ] *= 0
	actuation_weight_matrix = eye( actuation.shape[ 0 ] )
	actuation_weight_matrix[ :3, :3 ] *= 0.01
	actuation_weight_matrix[ 6:9, 6:9 ] *= 0.01
	actuation_weight_matrix[ 12:15, 12:15 ] *= 0.01
	final_cost_weight_matrix = eye( actuation.shape[ 0 ] )

	command_upper_bound = 50
	command_lower_bound = -50

	mpc_config = {
			'candidate_shape'         : result_shape,
			'model'                   : three_robots_chain,
			'initial_actuation'       : actuation,
			'initial_state'           : state,
			'model_kwargs'            : model_kwargs,
			'target_trajectory'       : trajectory,
			'optimization_horizon'    : optimization_horizon,
			'prediction_horizon'      : 0,
			'time_step'               : time_step,
			'time_steps_per_actuation': time_steps_per_actuation,
			'objective_function'      : three_robot_chain_objective,
			'pose_weight_matrix'      : pose_weight_matrix,
			'actuation_weight_matrix' : actuation_weight_matrix,
			'final_cost_weight_matrix': final_cost_weight_matrix,
			'objective_weight'        : 1.,
			'state_record'            : [ ],
			'actuation_record'        : [ ],
			'objective_record'        : [ ],
			'verbose'                 : False
			}

	other_config = {
			'max_iter'           : max_iter,
			'tolerance'          : tolerance,
			'n_frames'           : n_frames,
			'command_upper_bound': command_upper_bound,
			'command_lower_bound': command_lower_bound,
			}

	previous_states_record = [ deepcopy( state ) ]
	previous_actuation_record = [ deepcopy( actuation ) ]
	previous_objective_record = [ three_robot_chain_objective( state, actuation, **model_kwargs ) ]
	previous_compute_time_record = [ 0. ]
	previous_nfeval_record = [ 0. ]
	previous_H01_record = [ 0. ]
	previous_H12_record = [ 0. ]

	folder = (f'./plots/{three_robots_chain.__name__}_'
						f'{int( time() )}')

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
		dump( mpc_config | other_config, f, default = serialize_others )

	logger = Logger()

	for frame in range( n_frames ):

		mpc_config[ 'target_trajectory' ] = [ (p[ 0 ] - frame * time_step, p[ 1 ]) for p in trajectory
																					if p[ 0 ] - frame * time_step >= 0 ]

		mpc_config[ 'state_record' ] = [ ]
		mpc_config[ 'actuation_record' ] = [ ]
		mpc_config[ 'objective_record' ] = [ ]

		logger.log( f"frame {frame + 1}/{n_frames}" )

		result = concatenate( (result[ 1: ], zeros( (1, 6 * n_robots) )) )

		ti = perf_counter()

		result = optimize(
				cost_function = model_predictive_control_cost_function,
				cost_kwargs = mpc_config,
				initial_guess = result,
				tolerance = tolerance,
				max_iter = max_iter,
				bounds = None,
				constraints = None,
				verbose = False
				)

		actuation += result[ 0 ]
		state += three_robots_chain( state, actuation, **model_kwargs ) * time_step

		tf = perf_counter()
		compute_time = tf - ti
		previous_compute_time_record += [ compute_time ]
		previous_objective_record += [ three_robot_chain_objective(
				state, actuation, **model_kwargs
				) ]

		mpc_config[ 'initial_state' ] = state
		mpc_config[ 'initial_actuation' ] = actuation

		previous_states_record.append( deepcopy( state ) )
		previous_actuation_record.append( deepcopy( actuation ) )

		n_f_eval = len( mpc_config[ 'state_record' ] )
		previous_nfeval_record += [ n_f_eval ]

		logger.log( f"actuation={actuation[ :3 ]}-{actuation[ 6:9 ]}-{actuation[ 12:15 ]}" )
		logger.log( f"state={state[ : 3 ]}-{state[ 6:9 ]}-{state[ 12:15 ]}" )
		logger.log( f"{compute_time=:.6f}s" )
		logger.log( f"{n_f_eval=}" )

		ti = perf_counter()

		time_previous = [ i * time_step - (frame + 1) * time_step for i in range( frame + 2 ) ]
		time_prediction = [ i * time_step for i in range(
				mpc_config[ 'optimization_horizon' ] + mpc_config[ 'prediction_horizon' ]
				) ]

		fig_grid_shape = (15, 22)
		vsize = 13
		psizex = 3
		psizey = 3
		r0x = vsize
		r1x = vsize + psizex
		r2x = vsize + 2 * psizex
		fig = plt.figure( figsize = [ 21, 9 ] )

		view = plt.subplot2grid( fig_grid_shape, (0, 0), vsize, vsize, fig, projection = '3d' )
		view.set_xlabel( "x" )
		view.set_ylabel( "y" )
		view.set_zlabel( "z" )
		view.set_xlim( -1.5, 1.5 )
		view.set_ylim( -1.5, 1.5 )
		view.set_zlim( 0, 3 )
		view.invert_yaxis()
		view.invert_zaxis()

		ax_pos0 = plt.subplot2grid( fig_grid_shape, (0, r0x), psizey, psizex, fig )
		ax_pos0.set_title( 'robot0' )
		ax_pos0.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_pos0.set_ylim( -3, 3 )
		ax_pos0.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )
		ax_pos0.yaxis.set_label_position( "right" )
		ax_pos0.yaxis.tick_right()

		ax_ang0 = plt.subplot2grid( fig_grid_shape, (psizey, r0x), psizey, psizex, fig )
		ax_ang0.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_ang0.set_ylim( -2 * pi, 2 * pi )
		ax_ang0.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )
		ax_ang0.yaxis.set_label_position( "right" )
		ax_ang0.yaxis.tick_right()

		ax_act_pos0 = plt.subplot2grid( fig_grid_shape, (2 * psizey, r0x), psizey, psizex, fig )
		ax_act_pos0.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_act_pos0.set_ylim( 1.1 * command_lower_bound, 1.1 * command_upper_bound )
		ax_act_pos0.set_prop_cycle(
				cycler( 'color', [ 'blue', 'red', 'green' ] )
				)
		ax_act_pos0.yaxis.set_label_position( "right" )
		ax_act_pos0.yaxis.tick_right()

		ax_act_ang0 = plt.subplot2grid( fig_grid_shape, (3 * psizey, r0x), psizey, psizex, fig )
		ax_act_ang0.set_xlabel( 'time' )
		ax_act_ang0.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_act_ang0.set_ylim( 1.1 * command_lower_bound, 1.1 * command_upper_bound )
		ax_act_ang0.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )
		ax_act_ang0.yaxis.set_label_position( "right" )
		ax_act_ang0.yaxis.tick_right()

		ax_cat_H = plt.subplot2grid(
				fig_grid_shape, (4 * psizey + 1, r1x - psizex // 2), psizey - 1, psizex, fig
				)
		ax_cat_H.set_xlabel( 'time' )
		ax_cat_H.set_ylabel( 'lowest point' )
		ax_cat_H.set_xlim( time_previous[ 0 ], time_previous[ -1 ] )
		ax_cat_H.set_prop_cycle( cycler( 'color', [ 'blue', 'red' ] ) )

		ax_pos1 = plt.subplot2grid( fig_grid_shape, (0, r1x), psizey, psizex, fig )
		ax_pos1.set_title( 'robot1' )
		ax_pos1.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_pos1.set_ylim( -3, 3 )
		ax_pos1.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )
		ax_pos1.yaxis.set_label_position( "right" )
		ax_pos1.yaxis.tick_right()

		ax_ang1 = plt.subplot2grid( fig_grid_shape, (psizey, r1x), psizey, psizex, fig )
		ax_ang1.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_ang1.set_ylim( -2 * pi, 2 * pi )
		ax_ang1.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )
		ax_ang1.yaxis.set_label_position( "right" )
		ax_ang1.yaxis.tick_right()

		ax_act_pos1 = plt.subplot2grid( fig_grid_shape, (2 * psizey, r1x), psizey, psizex, fig )
		ax_act_pos1.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_act_pos1.set_ylim( 1.1 * command_lower_bound, 1.1 * command_upper_bound )
		ax_act_pos1.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )
		ax_act_pos1.yaxis.set_label_position( "right" )
		ax_act_pos1.yaxis.tick_right()

		ax_act_ang1 = plt.subplot2grid( fig_grid_shape, (3 * psizey, r1x), psizey, psizex, fig )
		ax_act_ang1.set_xlabel( 'time' )
		ax_act_ang1.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_act_ang1.set_ylim( 1.1 * command_lower_bound, 1.1 * command_upper_bound )
		ax_act_ang1.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )
		ax_act_ang1.yaxis.set_label_position( "right" )
		ax_act_ang1.yaxis.tick_right()

		ax_pos2 = plt.subplot2grid( fig_grid_shape, (0, r2x), psizey, psizex, fig )
		ax_pos2.set_title( 'robot2' )
		ax_pos2.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_pos2.set_ylim( -3, 3 )
		ax_pos2.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )
		ax_pos2.set_ylabel( 'position' )
		ax_pos2.yaxis.set_label_position( "right" )
		ax_pos2.yaxis.tick_right()

		ax_ang2 = plt.subplot2grid( fig_grid_shape, (psizey, r2x), psizey, psizex, fig )
		ax_ang2.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_ang2.set_ylim( -2 * pi, 2 * pi )
		ax_ang2.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )
		ax_ang2.set_ylabel( 'orientation' )
		ax_ang2.yaxis.set_label_position( "right" )
		ax_ang2.yaxis.tick_right()

		ax_act_pos2 = plt.subplot2grid( fig_grid_shape, (2 * psizey, r2x), psizey, psizex, fig )
		ax_act_pos2.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_act_pos2.set_ylim( 1.1 * command_lower_bound, 1.1 * command_upper_bound )
		ax_act_pos2.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )
		ax_act_pos2.set_ylabel( 'pos. act.' )
		ax_act_pos2.yaxis.set_label_position( "right" )
		ax_act_pos2.yaxis.tick_right()

		ax_act_ang2 = plt.subplot2grid( fig_grid_shape, (3 * psizey, r2x), psizey, psizex, fig )
		ax_act_ang2.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_act_ang2.set_ylim( 1.1 * command_lower_bound, 1.1 * command_upper_bound )
		ax_act_ang2.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )
		ax_act_ang2.set_ylabel( 'ang. act.' )
		ax_act_ang2.yaxis.set_label_position( "right" )
		ax_act_ang2.yaxis.tick_right()

		ax_obj = plt.subplot2grid( fig_grid_shape, (4 * psizey, r2x), psizey, psizex, fig )
		ax_obj.yaxis.tick_right()
		ax_obj.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_obj.set_prop_cycle( cycler( 'color', [ 'blue' ] ) )
		ax_obj.set_ylabel( 'objective' )
		ax_obj.set_xlabel( 'time' )
		ax_obj.yaxis.set_label_position( "right" )

		ax_comp_time = plt.subplot2grid( fig_grid_shape, (vsize, 1), 2, 5, fig )
		ax_comp_time.set_prop_cycle( cycler( 'color', [ 'blue' ] ) )
		ax_comp_time.set_ylabel( 'comp. time' )
		ax_comp_time.set_xlabel( 'time' )
		ax_comp_time.set_xlim( time_previous[ 0 ], time_previous[ -1 ] )

		ax_nfeval = plt.subplot2grid( fig_grid_shape, (vsize, 7), 2, 5, fig )
		ax_nfeval.set_prop_cycle( cycler( 'color', [ 'blue' ] ) )
		ax_nfeval.set_ylabel( 'n. eval' )
		ax_nfeval.set_xlabel( 'time' )
		ax_nfeval.set_xlim( time_previous[ 0 ], time_previous[ -1 ] )

		plt.subplots_adjust( hspace = 0., wspace = 0. )
		fig.suptitle( f"{frame + 1}/{n_frames} - {compute_time=:.6f}s - {n_f_eval=}" )

		target_pose = mpc_config[ 'target_trajectory' ][ 0 ][ 1 ]

		state_r0 = Rotation.from_euler( 'xyz', state[ 3:6 ] ).as_matrix()
		state_r1 = Rotation.from_euler( 'xyz', state[ 9:12 ] ).as_matrix()
		state_r2 = Rotation.from_euler( 'xyz', state[ 15:18 ] ).as_matrix()
		target_r0 = Rotation.from_euler( 'xyz', target_pose[ 3:6 ] ).as_matrix()

		surf_x = array( [ -1.5, 1.5 ] )
		surf_y = array( [ -1.5, 1.5 ] )
		surf_x, surf_y = meshgrid( surf_x, surf_y )
		surf_z = ones( surf_x.shape ) * 1.5
		view.plot_surface( surf_x, surf_y, surf_z, alpha = 0.1 )

		quiver_scale = .25
		view.quiver(
				*state[ :3 ], *(state_r0 @ (quiver_scale * array( [ 1., 0., 0. ] ))), color = 'blue'
				)
		view.quiver(
				*state[ 6:9 ], *(state_r1 @ (quiver_scale * array( [ 1., 0., 0. ] ))), color = 'red'
				)
		view.quiver(
				*state[ 12:15 ], *(state_r2 @ (quiver_scale * array( [ 1., 0., 0. ] ))), color = 'green'
				)
		view.quiver(
				*target_pose[ :3 ], *(target_r0 @ (quiver_scale * array( [ 1., 0., 0. ] ))),
				color = 'black'
				)

		view.plot(
				array( previous_states_record )[ :, 0 ],
				array( previous_states_record )[ :, 1 ],
				array( previous_states_record )[ :, 2 ],
				color = 'blue'
				)
		view.plot(
				array( previous_states_record )[ :, 6 ],
				array( previous_states_record )[ :, 7 ],
				array( previous_states_record )[ :, 8 ],
				color = 'red'
				)
		view.plot(
				array( previous_states_record )[ :, 12 ],
				array( previous_states_record )[ :, 13 ],
				array( previous_states_record )[ :, 14 ],
				color = 'green'
				)

		trajectory_time = [ p[ 0 ] - time_step * frame for p in trajectory ]
		trajectory_pos = array( [ p[ 1 ][ :3 ] for p in trajectory ] )
		trajectory_ang = array( [ p[ 1 ][ 3:6 ] for p in trajectory ] )

		view.plot( trajectory_pos[ :, 0 ], trajectory_pos[ :, 1 ], trajectory_pos[ :, 2 ], ':' )

		cat01, _, _, H01 = get_coor_marker_points_ideal_catenary(
				state[ 0 ], -state[ 1 ], -state[ 2 ], state[ 6 ], -state[ 7 ], -state[ 8 ], 3., .2
				)
		cat12, _, _, H12 = get_coor_marker_points_ideal_catenary(
				state[ 6 ], -state[ 7 ], -state[ 8 ], state[ 12 ], -state[ 13 ], -state[ 14 ], 3., .2
				)
		previous_H01_record.append( H01 + max( state[ 2 ], state[ 8 ] ) )
		previous_H12_record.append( H12 + max( state[ 8 ], state[ 14 ] ) )

		view.plot( cat01[ :, 0 ], -cat01[ :, 1 ], -cat01[ :, 2 ], 'blue' )
		view.plot( cat12[ :, 0 ], -cat12[ :, 1 ], -cat12[ :, 2 ], 'red' )

		ax_cat_H.plot( time_previous, previous_H01_record )
		ax_cat_H.plot( time_previous, previous_H12_record )

		ax_pos0.plot( trajectory_time, trajectory_pos, ':' )
		ax_ang0.plot( trajectory_time, trajectory_ang, ':' )

		previous_pos_record_array0 = array( previous_states_record )[ :, :3 ]
		previous_ang_record_array0 = array( previous_states_record )[ :, 3:6 ]
		previous_act_pos_record_array0 = array( previous_actuation_record )[ :, :3 ]
		previous_act_ang_record_array0 = array( previous_actuation_record )[ :, 3:6 ]
		previous_pos_record_array1 = array( previous_states_record )[ :, 6:9 ]
		previous_ang_record_array1 = array( previous_states_record )[ :, 9:12 ]
		previous_act_pos_record_array1 = array( previous_actuation_record )[ :, 6:9 ]
		previous_act_ang_record_array1 = array( previous_actuation_record )[ :, 9:12 ]
		previous_pos_record_array2 = array( previous_states_record )[ :, 12:15 ]
		previous_ang_record_array2 = array( previous_states_record )[ :, 15:18 ]
		previous_act_pos_record_array2 = array( previous_actuation_record )[ :, 12:15 ]
		previous_act_ang_record_array2 = array( previous_actuation_record )[ :, 15:18 ]

		ax_pos0.plot( time_previous, previous_pos_record_array0 )
		ax_ang0.plot( time_previous, previous_ang_record_array0 )
		ax_act_pos0.plot( time_previous, previous_act_pos_record_array0 )
		ax_act_ang0.plot( time_previous, previous_act_ang_record_array0 )
		ax_pos1.plot( time_previous, previous_pos_record_array1 )
		ax_ang1.plot( time_previous, previous_ang_record_array1 )
		ax_act_pos1.plot( time_previous, previous_act_pos_record_array1 )
		ax_act_ang1.plot( time_previous, previous_act_ang_record_array1 )
		ax_pos2.plot( time_previous, previous_pos_record_array2 )
		ax_ang2.plot( time_previous, previous_ang_record_array2 )
		ax_act_pos2.plot( time_previous, previous_act_pos_record_array2 )
		ax_act_ang2.plot( time_previous, previous_act_ang_record_array2 )
		ax_obj.plot( time_previous, previous_objective_record )

		for f_eval in range( n_f_eval ):
			pos_record_array0 = array( mpc_config[ 'state_record' ][ f_eval ] )[ :, :3 ]
			ang_record_array0 = array( mpc_config[ 'state_record' ][ f_eval ] )[ :, 3:6 ]
			pos_record_array1 = array( mpc_config[ 'state_record' ][ f_eval ] )[ :, 6:9 ]
			ang_record_array1 = array( mpc_config[ 'state_record' ][ f_eval ] )[ :, 9:12 ]
			pos_record_array2 = array( mpc_config[ 'state_record' ][ f_eval ] )[ :, 12:15 ]
			ang_record_array2 = array( mpc_config[ 'state_record' ][ f_eval ] )[ :, 15:18 ]

			view.plot(
					pos_record_array0[ :, 0 ],
					pos_record_array0[ :, 1 ],
					pos_record_array0[ :, 2 ],
					'blue',
					linewidth = .1
					)
			ax_pos0.plot( time_prediction, pos_record_array0, linewidth = .1 )
			ax_ang0.plot( time_prediction, ang_record_array0, linewidth = .1 )
			ax_act_pos0.plot(
					time_prediction, mpc_config[ 'actuation_record' ][ f_eval ][ :, :3 ], linewidth = .1
					)
			ax_act_ang0.plot(
					time_prediction, mpc_config[ 'actuation_record' ][ f_eval ][ :, 3:6 ], linewidth = .1
					)

			view.plot(
					pos_record_array1[ :, 0 ],
					pos_record_array1[ :, 1 ],
					pos_record_array1[ :, 2 ],
					'red',
					linewidth = .1
					)
			ax_pos1.plot( time_prediction, pos_record_array1, linewidth = .1 )
			ax_ang1.plot( time_prediction, ang_record_array1, linewidth = .1 )
			ax_act_pos1.plot(
					time_prediction, mpc_config[ 'actuation_record' ][ f_eval ][ :, 6:9 ], linewidth = .1
					)
			ax_act_ang1.plot(
					time_prediction, mpc_config[ 'actuation_record' ][ f_eval ][ :, 9:12 ], linewidth = .1
					)

			view.plot(
					pos_record_array2[ :, 0 ],
					pos_record_array2[ :, 1 ],
					pos_record_array2[ :, 2 ],
					'green',
					linewidth = .1
					)
			ax_pos2.plot( time_prediction, pos_record_array2, linewidth = .1 )
			ax_ang2.plot( time_prediction, ang_record_array2, linewidth = .1 )
			ax_act_pos2.plot(
					time_prediction, mpc_config[ 'actuation_record' ][ f_eval ][ :, 12:15 ], linewidth = .1
					)
			ax_act_ang2.plot(
					time_prediction, mpc_config[ 'actuation_record' ][ f_eval ][ :, 15:18 ], linewidth = .1
					)

			ax_obj.plot( time_prediction, mpc_config[ 'objective_record' ][ f_eval ], linewidth = .1 )

		# plot vertical line from y min to y max
		ax_pos0.axvline( color = 'g' )
		ax_ang0.axvline( color = 'g' )
		ax_act_pos0.axvline( color = 'g' )
		ax_act_ang0.axvline( color = 'g' )
		ax_pos1.axvline( color = 'g' )
		ax_ang1.axvline( color = 'g' )
		ax_act_pos1.axvline( color = 'g' )
		ax_act_ang1.axvline( color = 'g' )
		ax_pos2.axvline( color = 'g' )
		ax_ang2.axvline( color = 'g' )
		ax_act_pos2.axvline( color = 'g' )
		ax_act_ang2.axvline( color = 'g' )
		ax_obj.axvline( color = 'g' )
		ax_cat_H.axhline( 1.5, color = 'g' )

		ax_comp_time.plot( time_previous, previous_compute_time_record )
		ax_nfeval.plot( time_previous, previous_nfeval_record )

		plt.savefig( f'{folder}/{frame}.png' )
		plt.close( 'all' )
		del fig

		tf = perf_counter()
		save_time = tf - ti

		logger.lognl( f'saved figure {frame}.png in {save_time:.6f}s' )

		logger.save_at( folder )

	# create gif from frames
	logger.log( 'creating gif ...' )
	names = [ image for image in glob( f"{folder}/*.png" ) ]
	names.sort( key = lambda x: path.getmtime( x ) )
	frames = [ Image.open( name ) for name in names ]
	frame_one = frames[ 0 ]
	frame_one.save(
			f"{folder}/animation.gif", append_images = frames, loop = True, save_all = True
			)
	logger.log( f'saved at {folder}/animation.gif' )
