from glob import glob
from json import dump
from os import mkdir, path, remove
from time import perf_counter, time

from cycler import cycler
from numpy import array, concatenate, cos, cross, cumsum, diag, eye, multiply, sin, tan, ones
from numpy.linalg import inv
from PIL import Image
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt

from mpc import *


def two_robots_chain(
		x: ndarray, u: ndarray, robot_configuration
		) -> ndarray:
	"""
	:param x: state of the chain such that x = [pose_robot_0_wf, pose_robot_1_wf, vel_robot_0_rf,
	vel_robot_1_rf]
	:param u: actuation of the chain such that u = [actuation_robot_0, actuation_robot_1]
	:param robot_configuration: configuration of the robots considered identical
	:return: xdot: derivative of the state of the chain
	"""

	x0 = x[ :6 ]
	x0d = x[ 12:18 ]
	x1 = x[ 6:12 ]
	x1d = x[ 18: ]
	u0 = u[ :6 ]
	u1 = u[ 6: ]
	Phi0, Theta0, Psi0 = x0[ 3 ], x0[ 4 ], x0[ 5 ]
	Phi1, Theta1, Psi1 = x1[ 3 ], x1[ 4 ], x1[ 5 ]
	cPhi0, sPhi0 = cos( Phi0 ), sin( Phi0 )
	cPhi1, sPhi1 = cos( Phi1 ), sin( Phi1 )
	cTheta0, sTheta0, tTheta0 = cos( Theta0 ), sin( Theta0 ), tan( Theta0 )
	cTheta1, sTheta1, tTheta1 = cos( Theta1 ), sin( Theta1 ), tan( Theta1 )
	cPsi0, sPsi0 = cos( Psi0 ), sin( Psi0 )
	cPsi1, sPsi1 = cos( Psi1 ), sin( Psi1 )
	J0 = zeros( (6, 6) )
	J1 = zeros( (6, 6) )
	J0[ 0, :3 ] = array(
			[ cPsi0 * cTheta0, -sPsi0 * cPhi0 + cPsi0 * sTheta0 * sPhi0,
				sPsi0 * sPhi0 + cPsi0 * sTheta0 * cPhi0 ]
			)
	J0[ 1, :3 ] = array(
			[ sPsi0 * cTheta0, cPsi0 * cPhi0 + sPsi0 * sTheta0 * sPhi0,
				-cPsi0 * sPhi0 + sPsi0 * sTheta0 * cPhi0 ]
			)
	J0[ 2, :3 ] = array( [ -sTheta0, cTheta0 * sPhi0, cTheta0 * cPhi0 ] )
	J0[ 3, 3: ] = array( [ 1, sPhi0 * tTheta0, cPhi0 * tTheta0 ] )
	J0[ 4, 3: ] = array( [ 0, cPhi0, -sPhi0 ] )
	J0[ 5, 3: ] = array( [ 0, sPhi0 / cTheta0, cPhi0 / cTheta0 ] )
	J1[ 0, :3 ] = array(
			[ cPsi1 * cTheta1, -sPsi1 * cPhi1 + cPsi1 * sTheta1 * sPhi1,
				sPsi1 * sPhi1 + cPsi1 * sTheta1 * cPhi1 ]
			)
	J1[ 1, :3 ] = array(
			[ sPsi1 * cTheta1, cPsi1 * cPhi1 + sPsi1 * sTheta1 * sPhi1,
				-cPsi1 * sPhi1 + sPsi1 * sTheta1 * cPhi1 ]
			)
	J1[ 2, :3 ] = array( [ -sTheta1, cTheta1 * sPhi1, cTheta1 * cPhi1 ] )
	J1[ 3, 3: ] = array( [ 1, sPhi1 * tTheta1, cPhi1 * tTheta1 ] )
	J1[ 4, 3: ] = array( [ 1, cPhi1, -sPhi1 ] )
	J1[ 5, 3: ] = array( [ 1, sPhi1 / cTheta1, cPhi1 / cTheta1 ] )

	hydrodynamic_coefficients = robot_configuration[ "hydrodynamic_coefficients" ]
	D0 = diag( hydrodynamic_coefficients[ :6 ] ) + diag(
			multiply( hydrodynamic_coefficients[ 6: ], abs( x0d ) )
			)
	D1 = diag( hydrodynamic_coefficients[ :6 ] ) + diag(
			multiply( hydrodynamic_coefficients[ 6: ], abs( x1d ) )
			)
	buoyancy = robot_configuration[ "buoyancy" ]
	center_of_volume = robot_configuration[ "center_of_volume" ]
	Fw = mass * array( [ 0, 0, 9.80665 ] )
	Fb = buoyancy * array( [ 0, 0, -1 ] )
	s0 = zeros( 6 )
	s1 = zeros( 6 )
	s0[ :3 ] = J0[ :3, :3 ].T @ (Fw + Fb)
	s1[ :3 ] = J1[ :3, :3 ].T @ (Fw + Fb)
	s0[ 3: ] = cross( center_of_mass, J0[ :3, :3 ].T @ Fw ) + cross(
			center_of_volume, J0[ :3, :3 ].T @ Fb
			)
	s1[ 3: ] = cross( center_of_mass, J1[ :3, :3 ].T @ Fw ) + cross(
			center_of_volume, J1[ :3, :3 ].T @ Fb
			)

	I_inv = robot_configuration[ 'inertial_matrix_inv' ]
	xdot = zeros( x.shape )
	xdot[ :6 ] = J0 @ x0d
	xdot[ 6:12 ] = J1 @ x1d
	xdot[ 12:18 ] = I_inv @ (D0 @ x0 + s0 + u0)
	xdot[ 18: ] = I_inv @ (D1 @ x1 + s1 + u1)

	return xdot


def two_robot_chain_objective(
		x: ndarray, u: ndarray, robot_configuration
		) -> ndarray:
	"""
	:param x: state of the chain such that x = [pose_robot_0_wf, pose_robot_1_wf, vel_robot_0_rf,
	vel_robot_1_rf]
	:param u: actuation of the chain such that u = [actuation_robot_0, actuation_robot_1]
	:param robot_configuration: configuration of the robots considered identical
	:return: objective to minimize
	"""
	# 													ideal distance is .5m horizontally
	dist = x[ :3 ] - x[ 6:9 ] - .5 * array( [ 1., 1., 0. ] )
	return dist @ eye( 3 ) @ dist.T


def build_inertial_matrix( mass: float, inertial_coefficients: list[ float ] ):
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


if __name__ == "__main__":

	n_frames = 500
	max_iter = 1000
	tolerance = 1e-3
	time_step = 0.025

	sequence_time = n_frames * time_step
	state = zeros( (24,) )
	actuation = zeros( (12,) )

	base_optimization_horizon = 25
	optimization_horizon = base_optimization_horizon
	time_steps_per_actuation = 20
	result_shape = (optimization_horizon // time_steps_per_actuation + (
			1 if optimization_horizon % time_steps_per_actuation != 0 else 0), actuation.shape[ 0 ])
	result = zeros( result_shape )

	trajectory = [

			(time_step * 0.0 * n_frames, [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ]),
			(time_step * .2 * n_frames, [ 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0. ]),
			(time_step * .4 * n_frames, [ 0., 0., 1., 0., 0., -pi, 0., 0., 0., 0., 0., 0. ]),
			(time_step * .6 * n_frames, [ 1., -1., 1., 0., 0., -pi, 0., 0., 0., 0., 0., 0. ]),
			(time_step * .8 * n_frames, [ 1., -1., 0., 0., 0., -pi, 0., 0., 0., 0., 0., 0. ]),
			(time_step * 1.0 * n_frames, [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ]), (
					time_step * (n_frames + optimization_horizon + 1),
					[ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ])

			]

	trajectory = generate_trajectory(
			trajectory, n_frames  # // 20 + (1 if n_frames % 20 != 0 else 0)
			)

	mass = 11.5
	inertial_coefficients = [ .16, .16, .16, 0.0, 0.0, 0.0 ]
	center_of_mass = array( [ 0.0, 0.0, 0.0 ] )
	inertial_matrix = build_inertial_matrix( mass, inertial_coefficients )

	bluerov_configuration = {
			"mass"                     : mass,
			"center_of_mass"           : center_of_mass,
			"buoyancy"                 : 120.0,
			"center_of_volume"         : array( [ 0.0, 0.0, - 0.02 ] ),
			"inertial_matrix_inv"      : inv( inertial_matrix ),
			"hydrodynamic_coefficients": array(
					[ 4.03, 6.22, 5.18, 0.07, 0.07, 0.07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
					),
			"robot_max_actuation"      : array( [ 40, 40, 40, 40, 40, 40 ] ),
			"robot_max_actuation_ramp" : array( [ 80, 80, 80, 80, 80, 80 ] )
			}

	model_kwargs = { 'robot_configuration': bluerov_configuration }

	pose_weight_matrix = eye( actuation.shape[ 0 ] )
	pose_weight_matrix[ :3, :3 ] *= 2.
	# ignore error for robot1, only consider robot0, the objective will determine robot1
	pose_weight_matrix[ 6:12, 6:12 ] *= 0
	actuation_weight_matrix = eye( actuation.shape[ 0 ] )
	actuation_weight_matrix[ :3, :3 ] *= 0.01
	actuation_weight_matrix[ 6:9, 6:9 ] *= 0.01
	final_cost_weight_matrix = eye( actuation.shape[ 0 ] )

	command_upper_bound = 50
	command_lower_bound = -50

	mpc_config = {
			'candidate_shape'         : result_shape,
			'model'                   : two_robots_chain,
			'initial_actuation'       : actuation,
			'initial_state'           : state,
			'model_kwargs'            : model_kwargs,
			'target_trajectory'       : trajectory,
			'optimization_horizon'    : optimization_horizon,
			'prediction_horizon'      : 0,
			'time_step'               : time_step,
			'time_steps_per_actuation': time_steps_per_actuation,
			'objective_function'      : two_robot_chain_objective,
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
	previous_objective_record = [ two_robot_chain_objective( state, actuation, **model_kwargs ) ]
	previous_compute_time = [ 0. ]
	previous_nfeval = [ 0. ]

	folder = (f'./plots/{two_robots_chain.__name__}_'
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

		result = concatenate(
				(result[ 1: ], array( [ [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ] ] ))
				)

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
		state += two_robots_chain( state, actuation, **model_kwargs ) * time_step

		tf = perf_counter()
		compute_time = tf - ti
		previous_compute_time += [ compute_time ]
		previous_objective_record += [ two_robot_chain_objective( state, actuation, **model_kwargs ) ]

		mpc_config[ 'initial_state' ] = state
		mpc_config[ 'initial_actuation' ] = actuation

		previous_states_record.append( deepcopy( state ) )
		previous_actuation_record.append( deepcopy( actuation ) )

		n_f_eval = len( mpc_config[ 'state_record' ] )
		previous_nfeval += [ n_f_eval ]

		logger.log( f"actuation={actuation[ :3 ]}+{actuation[ 6:9 ]}" )
		logger.log( f"state={state[ : 3 ]}+{state[ 6:9 ]}" )
		logger.log( f"{compute_time=:.6f}s" )
		logger.log( f"{n_f_eval=}" )

		ti = perf_counter()

		time_previous = [ i * time_step - (frame + 1) * time_step for i in range( frame + 2 ) ]
		time_prediction = [ i * time_step for i in range(
				mpc_config[ 'optimization_horizon' ] + mpc_config[ 'prediction_horizon' ]
				) ]

		fig_grid_shape = (5, 12)
		r0x = 0
		r1x = 9
		fig = plt.figure( figsize = [ 16, 9 ] )

		view = plt.subplot2grid( fig_grid_shape, (0, r0x + 3), 4, 6, fig, projection = '3d' )
		view.set_xlabel( "x" )
		view.set_ylabel( "y" )
		view.set_xlim( -1.5, 1.5 )
		view.set_ylim( -1.5, 1.5 )
		view.set_zlim( 0, 3 )
		view.invert_yaxis()
		view.invert_zaxis()

		ax_pos0 = plt.subplot2grid( fig_grid_shape, (0, r0x), 1, 3, fig )
		ax_pos0.set_title( 'robot0' )
		ax_pos0.set_ylabel( 'position' )
		ax_pos0.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_pos0.set_ylim( -3, 3 )
		ax_pos0.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )
		ax_pos1 = plt.subplot2grid( fig_grid_shape, (0, r1x), 1, 3, fig )
		ax_pos1.set_title( 'robot1' )
		ax_pos1.set_ylabel( 'position' )
		ax_pos1.yaxis.set_label_position( "right" )
		ax_pos1.yaxis.tick_right()
		ax_pos1.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_pos1.set_ylim( -3, 3 )
		ax_pos1.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )

		ax_ang0 = plt.subplot2grid( fig_grid_shape, (1, r0x), 1, 3, fig )
		ax_ang0.set_ylabel( 'angle' )
		ax_ang0.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_ang0.set_ylim( -2 * pi, 2 * pi )
		ax_ang0.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )
		ax_ang1 = plt.subplot2grid( fig_grid_shape, (1, r1x), 1, 3, fig )
		ax_ang1.set_ylabel( 'angle' )
		ax_ang1.yaxis.set_label_position( "right" )
		ax_ang1.yaxis.tick_right()
		ax_ang1.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_ang1.set_ylim( -2 * pi, 2 * pi )
		ax_ang1.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )

		ax_act_pos0 = plt.subplot2grid( fig_grid_shape, (2, r0x), 1, 3, fig )
		ax_act_pos0.set_ylabel( 'pos. act.' )
		ax_act_pos0.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_act_pos0.set_ylim( 1.1 * command_lower_bound, 1.1 * command_upper_bound )
		ax_act_pos0.set_prop_cycle(
				cycler( 'color', [ 'blue', 'red', 'green' ] )
				)
		ax_act_pos1 = plt.subplot2grid( fig_grid_shape, (2, r1x), 1, 3, fig )
		ax_act_pos1.set_ylabel( 'pos. act.' )
		ax_act_pos1.yaxis.set_label_position( "right" )
		ax_act_pos1.yaxis.tick_right()
		ax_act_pos1.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_act_pos1.set_ylim( 1.1 * command_lower_bound, 1.1 * command_upper_bound )
		ax_act_pos1.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )

		ax_act_ang0 = plt.subplot2grid( fig_grid_shape, (3, r0x), 1, 3, fig )
		ax_act_ang0.set_ylabel( 'ang. act.' )
		ax_act_ang0.set_xlabel( 'time' )
		ax_act_ang0.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_act_ang0.set_ylim( 1.1 * command_lower_bound, 1.1 * command_upper_bound )
		ax_act_ang0.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )
		ax_act_ang1 = plt.subplot2grid( fig_grid_shape, (3, r1x), 1, 3, fig )
		ax_act_ang1.set_ylabel( 'ang. act.' )
		ax_act_ang1.yaxis.set_label_position( "right" )
		ax_act_ang1.yaxis.tick_right()
		ax_act_ang1.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_act_ang1.set_ylim( 1.1 * command_lower_bound, 1.1 * command_upper_bound )
		ax_act_ang1.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )

		ax_obj = plt.subplot2grid( fig_grid_shape, (4, r1x), 1, 3, fig )
		ax_obj.set_ylabel( 'objective' )
		ax_obj.set_xlabel( 'time' )
		ax_obj.yaxis.set_label_position( "right" )
		ax_obj.yaxis.tick_right()
		ax_obj.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_obj.set_prop_cycle( cycler( 'color', [ 'blue' ] ) )

		ax_comp_time = plt.subplot2grid( fig_grid_shape, (4, r0x + 4), 1, 2, fig )
		ax_comp_time.set_ylabel( 'comp. time' )
		ax_comp_time.set_xlabel( 'time' )
		ax_comp_time.set_xlim( time_previous[ 0 ], time_previous[ -1 ] )

		ax_nfeval = plt.subplot2grid( fig_grid_shape, (4, r0x + 6), 1, 2, fig )
		ax_nfeval.set_ylabel( 'n. eval' )
		ax_nfeval.set_xlabel( 'time' )
		ax_nfeval.yaxis.set_label_position( "right" )
		ax_nfeval.yaxis.tick_right()
		ax_nfeval.set_xlim( time_previous[ 0 ], time_previous[ -1 ] )

		plt.subplots_adjust( hspace = 0. )
		fig.suptitle( f"{frame + 1}/{n_frames} - {compute_time=:.6f}s - {n_f_eval=}" )

		target_pose = mpc_config[ 'target_trajectory' ][ 0 ][ 1 ]

		state_r0 = Rotation.from_euler( 'xyz', state[ 3:6 ] ).as_matrix()
		state_r1 = Rotation.from_euler( 'xyz', state[ 9:12 ] ).as_matrix()
		target_r0 = Rotation.from_euler( 'xyz', target_pose[ 3:6 ] ).as_matrix()

		quiver_scale = .5
		view.quiver(
				*state[ :3 ], *(state_r0 @ (quiver_scale * array( [ 1., 0., 0. ] ))), color = 'blue'
				)
		view.quiver(
				*state[ 6:9 ], *(state_r1 @ (quiver_scale * array( [ 1., 0., 0. ] ))), color = 'red'
				)
		view.quiver(
				*target_pose[ :3 ], *(target_r0 @ (quiver_scale * array( [ 1., 0., 0. ] ))), color = 'cyan'
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

		trajectory_time = [ p[ 0 ] - time_step * frame for p in trajectory ]
		trajectory_pos = array( [ p[ 1 ][ :3 ] for p in trajectory ] )
		trajectory_ang = array( [ p[ 1 ][ 3:6 ] for p in trajectory ] )

		view.plot( trajectory_pos[ :, 0 ], trajectory_pos[ :, 1 ], trajectory_pos[ :, 2 ], ':' )

		ax_pos0.plot( trajectory_time, trajectory_pos, ':' )
		ax_ang0.plot( trajectory_time, trajectory_ang, ':' )

		previous_pos_record_array0 = array( previous_states_record )[ :, :3 ]
		previous_ang_record_array0 = array( previous_states_record )[ :, 3:6 ]
		previous_pos_record_array1 = array( previous_states_record )[ :, 6:9 ]
		previous_ang_record_array1 = array( previous_states_record )[ :, 9:12 ]
		previous_act_pos_record_array0 = array( previous_actuation_record )[ :, :3 ]
		previous_act_ang_record_array0 = array( previous_actuation_record )[ :, 3:6 ]
		previous_act_pos_record_array1 = array( previous_actuation_record )[ :, 6:9 ]
		previous_act_ang_record_array1 = array( previous_actuation_record )[ :, 9:12 ]

		ax_pos0.plot( time_previous, previous_pos_record_array0 )
		ax_ang0.plot( time_previous, previous_ang_record_array0 )
		ax_act_pos0.plot( time_previous, previous_act_pos_record_array0 )
		ax_act_ang0.plot( time_previous, previous_act_ang_record_array0 )
		ax_pos1.plot( time_previous, previous_pos_record_array1 )
		ax_ang1.plot( time_previous, previous_ang_record_array1 )
		ax_act_pos1.plot( time_previous, previous_act_pos_record_array1 )
		ax_act_ang1.plot( time_previous, previous_act_ang_record_array1 )
		ax_obj.plot( time_previous, previous_objective_record )

		for f_eval in range( n_f_eval ):
			pos_record_array0 = array( mpc_config[ 'state_record' ][ f_eval ] )[ :, :3 ]
			ang_record_array0 = array( mpc_config[ 'state_record' ][ f_eval ] )[ :, 3:6 ]
			pos_record_array1 = array( mpc_config[ 'state_record' ][ f_eval ] )[ :, 6:9 ]
			ang_record_array1 = array( mpc_config[ 'state_record' ][ f_eval ] )[ :, 9:12 ]

			view.plot(
					pos_record_array0[ :, 0 ],
					pos_record_array0[ :, 1 ],
					pos_record_array0[ :, 2 ],
					'b',
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
					'b',
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
		ax_obj.axvline( color = 'g' )

		ax_comp_time.plot( time_previous, previous_compute_time )
		ax_nfeval.plot( time_previous, previous_nfeval )

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
