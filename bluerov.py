from glob import glob
from json import dump
from os import mkdir, path, remove
from time import perf_counter, time

from cycler import cycler
from numpy import array, concatenate, cos, cross, cumsum, diag, eye, multiply, sin, tan
from numpy.linalg import inv
from PIL import Image
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt

from mpc import *


def robot(
		x: ndarray, u: ndarray, robot_configuration
		) -> ndarray:
	Phi, Theta, Psi = x[ 3 ], x[ 4 ], x[ 5 ]
	cPhi, sPhi = cos( Phi ), sin( Phi )
	cTheta, sTheta, tTheta = cos( Theta ), sin( Theta ), tan( Theta )
	cPsi, sPsi = cos( Psi ), sin( Psi )
	J = zeros( (6, 6) )
	J[ 0, :3 ] = array(
			[ cPsi * cTheta, -sPsi * cPhi + cPsi * sTheta * sPhi, sPsi * sPhi + cPsi * sTheta * cPhi ]
			)
	J[ 1, :3 ] = array(
			[ sPsi * cTheta, cPsi * cPhi + sPsi * sTheta * sPhi, -cPsi * sPhi + sPsi * sTheta * cPhi ]
			)
	J[ 2, :3 ] = array( [ -sTheta, cTheta * sPhi, cTheta * cPhi ] )
	J[ 3, 3: ] = array( [ 1, sPhi * tTheta, cPhi * tTheta ] )
	J[ 4, 3: ] = array( [ 0, cPhi, -sPhi ] )
	J[ 5, 3: ] = array( [ 0, sPhi / cTheta, cPhi / cTheta ] )

	hydrodynamic_coefficients = robot_configuration[ "hydrodynamic_coefficients" ]
	D = diag( hydrodynamic_coefficients[ :6 ] ) + diag(
			multiply( hydrodynamic_coefficients[ 6: ], abs( x[ 6: ] ) )
			)
	buoyancy = robot_configuration[ "buoyancy" ]
	center_of_volume = robot_configuration[ "center_of_volume" ]
	Fw = mass * array( [ 0, 0, 9.80665 ] )
	Fb = buoyancy * array( [ 0, 0, -1 ] )
	s = zeros( 6 )
	s[ :3 ] = J[ :3, :3 ].T @ (Fw + Fb)
	s[ 3: ] = (
			cross( center_of_mass, J[ :3, :3 ].T @ Fw ) + cross( center_of_volume, J[ :3, :3 ].T @ Fb ))

	I_inv = robot_configuration[ 'inertial_matrix_inv' ]
	xdot = zeros( x.shape )
	xdot[ :6 ] = J @ x[ 6: ]
	xdot[ 6: ] = I_inv @ (D @ x[ 6: ] + s + u)

	return xdot


def robot_objective(
		x: ndarray, u: ndarray, robot_configuration
		):
	return - norm( x[ 6: ] )


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

	n_frames = 400
	max_iter = 1000
	tolerance = 1e-6
	time_step = 0.025

	sequence_time = n_frames * time_step
	state = array( [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ] )
	actuation = array( [ 0., 0., 0., 0., 0., 0. ] )

	base_optimization_horizon = 25
	optimization_horizon = base_optimization_horizon
	time_steps_per_actuation = 5
	result_shape = (optimization_horizon // time_steps_per_actuation + (
			1 if optimization_horizon % time_steps_per_actuation != 0 else 0), actuation.shape[ 0 ])
	result = zeros( result_shape )

	trajectory = [ (time_step * 0.0 * n_frames, [ 0., 0., 0., 0., 0., 0. ]),
								 (time_step * .2 * n_frames, [ 0., 0., 1., 0., 0., 0. ]),
								 (time_step * .4 * n_frames, [ 0., 0., 1., 0., 0., -pi ]),
								 (time_step * .6 * n_frames, [ -1., -1., 1., 0., 0., -pi ]),
								 (time_step * .8 * n_frames, [ -1., -1., 0., 0., 0., -pi ]),
								 (time_step * 1.0 * n_frames, [ 0., 0., 0., 0., 0., 0. ]),
								 (time_step * (n_frames + optimization_horizon), [ 0., 0., 0., 0., 0., 0. ]) ]

	trajectory = generate_trajectory(
			trajectory, n_frames // 20 + (1 if n_frames % 20 != 0 else 0)
			)
	target_pose = trajectory[ 0 ][ 1 ]

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
	actuation_weight_matrix = eye( actuation.shape[ 0 ] )
	actuation_weight_matrix[ :3, :3 ] *= 0.01
	final_cost_weight_matrix = eye( actuation.shape[ 0 ] )

	command_upper_bound = 50
	command_lower_bound = -50

	mpc_config = {
			'candidate_shape'         : result_shape,
			'model'                   : robot,
			'initial_actuation'       : actuation,
			'initial_state'           : state,
			'model_kwargs'            : model_kwargs,
			'target_trajectory'       : trajectory,
			'optimization_horizon'    : optimization_horizon,
			'prediction_horizon'      : 0,
			'time_step'               : time_step,
			'time_steps_per_actuation': time_steps_per_actuation,
			'objective_function'      : robot_objective,
			'pose_weight_matrix'      : pose_weight_matrix,
			'actuation_weight_matrix' : actuation_weight_matrix,
			'final_cost_weight_matrix': final_cost_weight_matrix,
			'objective_weight'        : 0.,
			'state_record'            : [ ],
			'actuation_record'        : [ ],
			'objective_record'        : None,
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

	folder = (f'./plots/{robot.__name__}_'
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

		mpc_config[ 'target_trajectory' ] = [ (p[ 0 ] - frame * time_step, p[ 1 ]) for p in
																					trajectory ]

		mpc_config[ 'state_record' ] = [ ]
		mpc_config[ 'actuation_record' ] = [ ]

		logger.log( f"frame {frame + 1}/{n_frames}" )

		result = concatenate( (result[ 1: ], array( [ [ 0., 0., 0., 0., 0., 0. ] ] )) )

		ti = perf_counter()

		result = optimize(
				cost_function = model_predictive_control_cost_function,
				cost_kwargs = mpc_config,
				initial_guess = result,
				tolerance = tolerance,
				max_iter = max_iter,
				bounds = None,
				constraints = None,
				# NonlinearConstraint(
				# lambda x: (actuation + cumsum(
				# 		x.reshape( result_shape ), axis = 0
				# 		)).flatten(), command_lower_bound, command_upper_bound
				# ),
				verbose = False
				)

		actuation += result[ 0 ]
		state += robot( state, actuation, **model_kwargs ) * time_step

		tf = perf_counter()
		compute_time = tf - ti

		mpc_config[ 'initial_state' ] = state
		mpc_config[ 'initial_actuation' ] = actuation

		previous_states_record.append( deepcopy( state ) )
		previous_actuation_record.append( deepcopy( actuation ) )

		n_f_eval = len( mpc_config[ 'state_record' ] )

		logger.log( f"actuation={actuation}" )
		logger.log( f"state={state[ : state.shape[ 0 ] // 2 ]}" )
		logger.log( f"{compute_time=:.6f}s" )
		logger.log( f"{n_f_eval=}" )

		ti = perf_counter()

		time_previous = [ i * time_step - (frame + 1) * time_step for i in range( frame + 2 ) ]
		time_prediction = [ i * time_step for i in range(
				mpc_config[ 'optimization_horizon' ] + mpc_config[ 'prediction_horizon' ]
				) ]

		fig = plt.figure()
		view = plt.subplot2grid( (3, 5), (0, 0), 4, 3, fig, projection = '3d' )
		view.set_xlabel( "x" )
		view.set_ylabel( "y" )
		view.set_xlim( -2, 2 )
		view.set_ylim( -2, 2 )
		view.set_zlim( 0, 4 )
		view.invert_yaxis()
		view.invert_zaxis()

		ax_pos = plt.subplot2grid( (3, 5), (0, 3), 1, 2, fig )
		ax_pos.set_ylabel( 'position' )
		ax_pos.yaxis.set_label_position( "right" )
		ax_pos.yaxis.tick_right()
		ax_pos.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_pos.set_ylim( -3, 3 )
		ax_pos.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )

		ax_ang = plt.subplot2grid( (3, 5), (1, 3), 1, 2, fig )
		ax_ang.set_ylabel( 'angle' )
		ax_ang.yaxis.set_label_position( "right" )
		ax_ang.yaxis.tick_right()
		ax_ang.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_ang.set_ylim( -2 * pi, 2 * pi )
		ax_ang.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )

		ax_act = plt.subplot2grid( (3, 5), (2, 3), 1, 2, fig )
		ax_act.set_ylabel( 'actuation' )
		ax_act.set_xlabel( 'time' )
		ax_act.yaxis.set_label_position( "right" )
		ax_act.yaxis.tick_right()
		ax_act.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_act.set_ylim( 1.1 * command_lower_bound, 1.1 * command_upper_bound )
		ax_act.set_prop_cycle(
				cycler( 'color', [ 'blue', 'red', 'green', 'cyan', 'orange', 'olive' ] )
				)

		plt.subplots_adjust( hspace = 0., wspace = .5 )
		fig.suptitle( f"{frame + 1}/{n_frames} - {compute_time=:.6f}s - {n_f_eval=}" )

		filtered_traj = list(
				filter( lambda x: 0.0 <= x[ 0 ] <= time_step, mpc_config[ 'target_trajectory' ] )
				)
		if len( filtered_traj ) > 0:
			target_pose = list( filtered_traj )[ 0 ][ 1 ]

		state_r = Rotation.from_euler( 'xyz', state[ 3:6 ] ).as_matrix()
		target_r = Rotation.from_euler( 'xyz', target_pose[ 3:6 ] ).as_matrix()

		quiver_scale = .5
		view.quiver( *state[ :3 ], *(state_r @ (quiver_scale * array( [ 1., 0., 0. ] ))) )
		view.quiver( *target_pose[ :3 ], *(target_r @ (quiver_scale * array( [ 1., 0., 0. ] ))) )

		view.plot(
				array( previous_states_record )[ :, 0 ],
				array( previous_states_record )[ :, 1 ],
				array( previous_states_record )[ :, 2 ],
				'b'
				)

		trajectory_time = [ p[ 0 ] for p in mpc_config[ 'target_trajectory' ] ]
		trajectory_pos = array( [ p[ 1 ][ :3 ] for p in mpc_config[ 'target_trajectory' ] ] )
		trajectory_ang = array( [ p[ 1 ][ 3: ] for p in mpc_config[ 'target_trajectory' ] ] )

		view.plot( trajectory_pos[ :, 0 ], trajectory_pos[ :, 1 ], trajectory_pos[ :, 2 ], ':' )

		ax_pos.plot( trajectory_time, trajectory_pos, ':' )
		ax_ang.plot( trajectory_time, trajectory_ang, ':' )

		previous_pos_record_array = array( previous_states_record )[ :, :3 ]
		previous_ang_record_array = array( previous_states_record )[ :, 3:6 ]

		ax_pos.plot( time_previous, previous_pos_record_array )
		ax_ang.plot( time_previous, previous_ang_record_array )
		ax_act.plot( time_previous, previous_actuation_record )

		for f_eval in range( n_f_eval ):
			pos_record_array = array( mpc_config[ 'state_record' ][ f_eval ] )[ :, :3 ]
			ang_record_array = array( mpc_config[ 'state_record' ][ f_eval ] )[ :, 3:6 ]

			view.plot(
					pos_record_array[ :, 0 ],
					pos_record_array[ :, 1 ],
					pos_record_array[ :, 2 ],
					'b',
					linewidth = .1
					)

			ax_pos.plot( time_prediction, pos_record_array, linewidth = .1 )
			ax_ang.plot( time_prediction, ang_record_array, linewidth = .1 )
			ax_act.plot( time_prediction, mpc_config[ 'actuation_record' ][ f_eval ], linewidth = .1 )

		# plot vertical line from y min to y max
		ax_pos.axvline( color = 'g' )
		ax_ang.axvline( color = 'g' )
		ax_act.axvline( color = 'g' )

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
