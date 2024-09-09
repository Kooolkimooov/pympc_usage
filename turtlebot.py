from glob import glob
from json import dump
from os import mkdir, path, remove
from time import time

from cycler import cycler
from matplotlib import pyplot as plt
from numpy import array, cos, linspace, pi, sin
from PIL import Image

from mpc import *
from utils import Logger, serialize_others


# pendulum with cart
def turtle(
		state: ndarray, actuation: ndarray
		) -> ndarray:

	state_derivative = zeros( state.shape )
	_, _, theta, _, _, _ = state
	v, w = actuation
	state_derivative[ 0 ] = v * cos( theta )
	state_derivative[ 1 ] = v * sin( theta )
	state_derivative[ 2 ] = w

	return state_derivative


def plot( mpc, full_trajectory ):

	# we record the initial value + the new value after the integration in `step()`
	time_previous = [ i * time_step - (frame + 1) * time_step for i in
										range( len( mpc.model.previous_states ) ) ]
	time_prediction = [ i * time_step for i in
											range( mpc.predicted_trajectories[ 0 ].shape[ 0 ] - 1 ) ]

	fig = plt.figure()
	view = plt.subplot2grid( (3, 5), (0, 0), 4, 3, fig )
	view.grid( True )
	view.set_xlabel( "x" )
	view.set_ylabel( "y" )
	how = 11. / 8.9
	view.set_xlim( -2, 2 )
	view.set_ylim( -2 * how, 2 * how )

	ax_pos = plt.subplot2grid( (3, 5), (0, 3), 1, 2, fig )
	ax_pos.set_ylabel( 'position' )
	ax_pos.yaxis.set_label_position( "right" )
	ax_pos.yaxis.tick_right()
	ax_pos.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
	ax_pos.set_ylim( -3, 3 )
	ax_pos.set_prop_cycle( cycler( 'color', [ 'blue', 'red' ] ) )

	ax_ang = plt.subplot2grid( (3, 5), (1, 3), 1, 2, fig )
	ax_ang.set_ylabel( 'angle' )
	ax_ang.yaxis.set_label_position( "right" )
	ax_ang.yaxis.tick_right()
	ax_ang.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
	ax_ang.set_ylim( - pi, 2 * pi )
	ax_ang.set_prop_cycle( cycler( 'color', [ 'blue' ] ) )

	ax_act = plt.subplot2grid( (3, 5), (2, 3), 1, 2, fig )
	ax_act.set_ylabel( 'actuation' )
	ax_act.set_xlabel( 'time' )
	ax_act.yaxis.set_label_position( "right" )
	ax_act.yaxis.tick_right()
	ax_act.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
	ax_act.set_prop_cycle( cycler( 'color', [ 'blue', 'red' ] ) )

	plt.subplots_adjust( hspace = 0., wspace = 0. )
	fig.suptitle(
			f"{frame + 1}/{n_frames} - {mpc.times[ -1 ]:.6f}s - {len( mpc.candidate_actuations )}"
			)

	view.scatter( mpc.model.state[ 0 ], mpc.model.state[ 1 ], c = 'r', s = 100 )
	view.quiver(
			mpc.model.state[ 0 ],
			mpc.model.state[ 1 ],
			.1 * cos( mpc.model.state[ 2 ] ),
			.1 * sin( mpc.model.state[ 2 ] ),
			color = 'r'
			)

	view.quiver(
			mpc.target_trajectory[ 0, 0, 0 ],
			mpc.target_trajectory[ 0, 0, 1 ],
			.1 * cos( mpc.target_trajectory[ 0, 0, 2 ] ),
			.1 * sin( mpc.target_trajectory[ 0, 0, 2 ] ),
			color = 'b'
			)
	view.plot( full_trajectory[ :, 0, 0 ], full_trajectory[ :, 0, 1 ], ':b' )

	previous_pos_record_array = array( mpc.model.previous_states )[ :, :2 ]
	previous_ang_record_array = array( mpc.model.previous_states )[ :, 2 ]

	view.plot( previous_pos_record_array[ :, 0 ], previous_pos_record_array[ :, 1 ], 'r' )
	ax_pos.plot( time_previous, previous_pos_record_array )
	ax_ang.plot( time_previous, previous_ang_record_array )
	ax_act.plot( time_previous, mpc.model.previous_actuations )

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
	ax_ang.plot(
			time_previous + time_prediction[ 1: ],
			full_trajectory[ :len( time_previous ) + len( time_prediction ) - 1, 0, 2 ],
			':b'
			)

	step = 1
	if len( mpc.predicted_trajectories ) > 1000:
		step = len( mpc.predicted_trajectories ) // 1000

	for f_eval in range( 0, len( mpc.predicted_trajectories ), step ):
		pos_record_array = mpc.predicted_trajectories[ f_eval ][ 1:, 0, :2 ]
		ang_record_array = mpc.predicted_trajectories[ f_eval ][ 1:, 0, 2 ]

		view.plot( pos_record_array[ :, 0 ], pos_record_array[ :, 1 ], 'r', linewidth = .1 )

		ax_pos.plot( time_prediction, pos_record_array, linewidth = .1 )
		ax_ang.plot( time_prediction, ang_record_array, linewidth = .1 )
		ax_act.plot( time_prediction, mpc.candidate_actuations[ f_eval ][ 1:, 0, : ], linewidth = .1 )

	# plot vertical line from y min to y max
	ax_pos.axvline( color = 'k' )
	ax_ang.axvline( color = 'k' )
	ax_act.axvline( color = 'k' )

	return fig


if __name__ == "__main__":

	n_frames = 200
	time_step = 0.05

	initial_state = array( [ 0., 0., 0., 0., 0., 0. ] )
	initial_actuation = array( [ 0., 0. ] )

	actuation_weight_matrix = .001 * eye( 2 )

	angle = linspace( 0, 4 * pi, 2 * n_frames )
	trajectory = array( [ cos( angle ), sin( angle ), angle + pi / 2 ] ).T.reshape(
			2 * n_frames, 1, 3
			)

	turtle_model = Model( turtle, time_step, initial_state, initial_actuation, record = True )
	mpc = MPC(
			turtle_model,
			30,
			trajectory[ 1:31 ],
			time_steps_per_actuation = 5,
			actuation_derivative_weight_matrix = actuation_weight_matrix,
			final_weight = 2,
			tolerance = 1e-3,
			record = True
			)

	logger = Logger()

	folder = (f'./plots/{turtle.__name__}_{int( time() )}')

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
		dump( mpc.__dict__ | turtle_model.__dict__, f, default = serialize_others )

	for frame in range( n_frames ):

		logger.log( f"frame {frame + 1}/{n_frames}" )

		mpc.target_trajectory = array( trajectory[ frame + 1:frame + mpc.horizon + 1 ] )

		mpc.compute_actuation()
		mpc.apply_result()
		turtle_model.step()

		logger.log( f"{turtle_model.actuation=}" )
		logger.log( f"{turtle_model.state[:3]=}" )
		logger.log( f"{mpc.times[-1]=}" )

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
