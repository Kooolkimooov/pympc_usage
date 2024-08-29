from glob import glob
from json import dump
from os import mkdir, path, remove
from time import time

import matplotlib.pyplot as plt
from cycler import cycler
from numpy import array, cos, cross, diag, multiply, pi, sin, tan
from numpy.linalg import inv
from PIL import Image
from scipy.spatial.transform import Rotation

from mpc import *


def robot(
		state: ndarray, actuation: ndarray, robot_configuration
		) -> ndarray:
	Phi, Theta, Psi = state[ 3 ], state[ 4 ], state[ 5 ]
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
			multiply( hydrodynamic_coefficients[ 6: ], abs( state[ 6: ] ) )
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
	xdot = zeros( state.shape )
	xdot[ :6 ] = J @ state[ 6: ]
	xdot[ 6: ] = I_inv @ (D @ state[ 6: ] + s + actuation)

	return xdot


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


def plot( mpc, full_trajectory ):
	# we record the initial value + the new value after the integration in `step()`
	time_previous = [ i * time_step - (frame + 1) * time_step for i in
										range( len( mpc.model.previous_states ) ) ]
	time_prediction = [ i * time_step for i in
											range( mpc.predicted_trajectories[ 0 ].shape[ 0 ] - 1 ) ]

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
	ax_act.set_prop_cycle(
			cycler( 'color', [ 'blue', 'red', 'green', 'cyan', 'orange', 'olive' ] )
			)

	plt.subplots_adjust( hspace = 0., wspace = .5 )
	fig.suptitle( f"{frame + 1}/{n_frames} - {mpc.times[ -1 ]:.6f}s" )

	state_r = Rotation.from_euler( 'xyz', mpc.model.state[ 3:6 ] ).as_matrix()
	target_r = Rotation.from_euler( 'xyz', mpc.target_trajectory[ 0, 0, 3:6 ] ).as_matrix()

	quiver_scale = .5
	view.quiver( *mpc.model.state[ :3 ], *(state_r @ (quiver_scale * array( [ 1., 0., 0. ] ))) )
	view.quiver(
			*mpc.target_trajectory[ 0, 0, :3 ], *(target_r @ (quiver_scale * array( [ 1., 0., 0. ] )))
			)

	view.plot(
			array( mpc.model.previous_states )[ :, 0 ],
			array( mpc.model.previous_states )[ :, 1 ],
			array( mpc.model.previous_states )[ :, 2 ],
			'b'
			)

	ax_pos.plot(
			time_previous + time_prediction[ 1: ],
			full_trajectory[ :len( time_previous ) + len( time_prediction ) - 1, 0, 0 ],
			':b'
			)

	ax_pos.plot(
			time_previous + time_prediction[ 1: ],
			full_trajectory[ :len( time_previous ) + len( time_prediction ) - 1, 0, 1 ],
			':b'
			)

	ax_pos.plot(
			time_previous + time_prediction[ 1: ],
			full_trajectory[ :len( time_previous ) + len( time_prediction ) - 1, 0, 2 ],
			':b'
			)

	view.plot(
			full_trajectory[ :, 0, 0 ], full_trajectory[ :, 0, 1 ], full_trajectory[ :, 0, 2 ], ':'
			)

	previous_pos_record_array = array( mpc.model.previous_states )[ :, :3 ]
	previous_ang_record_array = array( mpc.model.previous_states )[ :, 3:6 ]

	ax_pos.plot( time_previous, previous_pos_record_array )
	ax_ang.plot( time_previous, previous_ang_record_array )
	ax_act.plot( time_previous, mpc.model.previous_actuations )

	step = 1
	if len( mpc.predicted_trajectories ) > 1000:
		step = len( mpc.predicted_trajectories ) // 1000

	for f_eval in range( 0, len( mpc.predicted_trajectories ), step ):
		pos_record_array = mpc.predicted_trajectories[ f_eval ][ 1:, 0, :3 ]
		ang_record_array = mpc.predicted_trajectories[ f_eval ][ 1:, 0, 3:6 ]

		view.plot(
				pos_record_array[ :, 0 ],
				pos_record_array[ :, 1 ],
				pos_record_array[ :, 2 ],
				'b',
				linewidth = .1
				)

		ax_pos.plot( time_prediction, pos_record_array, linewidth = .1 )
		ax_ang.plot( time_prediction, ang_record_array, linewidth = .1 )
		ax_act.plot( time_prediction, mpc.candidate_actuations[ f_eval ][ 1:, 0, : ], linewidth = .1 )

	# plot vertical line from y min to y max
	ax_pos.axvline( color = 'g' )
	ax_ang.axvline( color = 'g' )
	ax_act.axvline( color = 'g' )

	return fig


if __name__ == "__main__":

	n_frames = 400
	time_step = 0.025

	state = array( [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ] )
	actuation = array( [ 0., 0., 0., 0., 0., 0. ] )

	horizon = 25

	key_frames = [ (0.0, [ 0., 0., 0., 0., 0., 0. ]), (.2, [ 0., 0., 1., 0., 0., 0. ]),
								 (.4, [ 0., 0., 1., 0., 0., -pi ]), (.6, [ -1., -1., 1., 0., 0., -pi ]),
								 (.8, [ -1., -1., 0., 0., 0., -pi ]), (1.0, [ 0., 0., 0., 0., 0., 0. ]),
								 (2., [ 0., 0., 0., 0., 0., 0. ]) ]

	trajectory = generate_trajectory(
			key_frames, 2 * n_frames
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
					)
			}

	model_kwargs = { 'robot_configuration': bluerov_configuration }

	pose_weight_matrix = eye( actuation.shape[ 0 ] )
	pose_weight_matrix[ :3, :3 ] *= 2.
	actuation_weight_matrix = eye( actuation.shape[ 0 ] )
	actuation_weight_matrix[ :3, :3 ] *= 0.01
	final_cost_weight = 1

	bluerov = Model( robot, time_step, state, actuation, kwargs = model_kwargs, record = True )
	mpc = MPC(
			bluerov,
			horizon,
			trajectory[ 1:horizon + 1 ],
			time_steps_per_actuation = 25,
			tolerance = 1e-3,
			pose_weight_matrix = pose_weight_matrix,
			actuation_derivative_weight_matrix = actuation_weight_matrix,
			final_weight = final_cost_weight,
			record = True
			)

	logger = Logger()

	folder = (f'./plots/{robot.__name__}_{int( time() )}')

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
		dump( bluerov.__dict__ | mpc.__dict__, f, default = serialize_others )

	for frame in range( n_frames ):

		logger.log( f"frame {frame + 1}/{n_frames}" )

		mpc.target_trajectory = trajectory[ frame + 1:frame + mpc.horizon + 1 ]

		mpc.optimize()
		mpc.apply_result()
		bluerov.step()

		logger.log( f"ux={bluerov.actuation[ :3 ]}" )
		logger.log( f"ut={bluerov.actuation[ 3:6 ]}" )
		logger.log( f"{mpc.times[-1]=:.6f}s" )

		fig = plot( mpc, trajectory )
		plt.savefig( f'{folder}/{frame}.png' )
		plt.close( 'all' )
		del fig

		logger.lognl( f'saved figure {frame}.png' )
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
