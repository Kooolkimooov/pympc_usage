from glob import glob
from json import dump
from os import mkdir, path, remove
from time import perf_counter, time

from matplotlib import pyplot as plt
from numpy import array, cos, cumsum, eye, pi, sin
from numpy.linalg import solve
from PIL import Image

from mpc import *


# pendulum with cart
def pendulum(
		x: ndarray, u: ndarray, cart_mass: float = 1, arm_length: float = 1, mass: float = 1
		) -> ndarray:

	g = 9.80665
	x, theta, dx, dtheta = x

	A = array(
			[ [ 1, 0, 0, 0 ], [ 0, 1, 0, 0 ], [ 0, 0, cos( theta ), arm_length ],
				[ 0, 0, 1 + cart_mass / mass, arm_length * cos( theta ) ] ]
			)

	b = array(
			[ dx, dtheta, -g * sin( theta ), arm_length * dtheta ** 2 * sin( theta ) + u[ 0 ] / mass ]
			)

	xdot = solve( A, b )

	return xdot


def pendulum_objective(
		x: ndarray, u: ndarray, cart_mass: float = 1, arm_length: float = 1, mass: float = 1
		) -> float:

	g = 9.80665

	_, theta, dx, dtheta = x
	Ep = - mass * g * arm_length * cos( theta )
	Ek = 0.5 * cart_mass * dx ** 2 + 0.5 * mass * (
			(arm_length * dtheta * sin( theta )) ** 2 + (dx + arm_length * dtheta * cos( theta )) ** 2)

	# objective is to maximize potiential energy and minimize kinetic energy
	objective = Ek - Ep

	return objective


if __name__ == "__main__":

	n_frames = 300
	time_step = 0.05
	max_iter = 1000
	tolerance = 1e-6

	state = array( [ 0., 0., 0., 0. ] )
	trajectory = [ (time_step * .34 * n_frames, [ 1., pi ]),
								 (time_step * 1.0 * n_frames, [ -1., pi ]) ]
	actuation = array( [ 0. ] )

	model_kwargs = { "cart_mass": 1, "arm_length": 1, "mass": 1 }

	optimization_horizon = 25
	time_steps_per_actuation = 3

	pose_weight_matrix = eye( state.shape[ 0 ] // 2 )
	# weight_matrix[0, 0] = 2.

	actuation_weight_matrix = .1 * eye( actuation.shape[ 0 ] )

	result_shape = (optimization_horizon // time_steps_per_actuation + 1, actuation.shape[ 0 ])
	result = zeros( result_shape )

	command_upper_bound = 50
	command_lower_bound = -50
	command_derivative_upper_bound = int( 25 / time_step )
	command_derivative_lower_bound = int( -25 / time_step )

	mpc_config = {
			'candidate_shape'         : result_shape,
			'model'                   : pendulum,
			'initial_actuation'       : actuation,
			'initial_state'           : state,
			'model_kwargs'            : model_kwargs,
			'target_pose'             : trajectory[ 0 ][ 1 ],
			'optimization_horizon'    : optimization_horizon,
			'prediction_horizon'      : 0,
			'time_step'               : time_step,
			'time_steps_per_actuation': time_steps_per_actuation,
			'objective_function'      : pendulum_objective,
			'pose_weight_matrix'      : pose_weight_matrix,
			'actuation_weight_matrix' : actuation_weight_matrix,
			'final_cost_weight'       : 0.,
			'objective_weight'        : 1.,
			'state_record'            : [ ],
			'actuation_record'        : [ ],
			'objective_record'        : [ ],
			'verbose'                 : False
			}

	other_config = {
			'max_iter'                      : max_iter,
			'tolerance'                     : tolerance,
			'n_frames'                      : n_frames,
			'trajectory'                    : trajectory,
			'command_upper_bound'           : command_upper_bound,
			'command_lower_bound'           : command_lower_bound,
			'command_derivative_upper_bound': command_derivative_upper_bound,
			'command_derivative_lower_bound': command_derivative_lower_bound,
			}

	previous_states_record = [ deepcopy( state ) ]
	previous_actuation_record = [ deepcopy( actuation ) ]
	previous_objective_record = [ pendulum_objective( state, actuation, **model_kwargs ) ]
	previous_target_record = [ ]

	folder = (f'./plots/{pendulum.__name__}_'
						f'{int( time() )}')

	logger = Logger()

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

	for frame in range( n_frames ):

		for index in range( len( previous_target_record ) + 1, len( trajectory ) ):
			if trajectory[ index - 1 ][ 0 ] < frame * time_step:
				previous_target_record.append( trajectory[ index - 1 ] )
				mpc_config[ 'target_pose' ] = trajectory[ index ][ 1 ]
				break

		mpc_config[ 'state_record' ] = [ ]
		mpc_config[ 'actuation_record' ] = [ ]
		mpc_config[ 'objective_record' ] = [ ]

		print( f"frame {frame + 1}/{n_frames}\t", end = ' ' )

		ti = perf_counter()

		result = optimize(
				cost_function = model_predictive_control_cost_function,
				cost_kwargs = mpc_config,
				initial_guess = result,
				tolerance = tolerance,
				max_iter = max_iter,
				constraints = NonlinearConstraint(
						lambda x: actuation + cumsum( x ), command_lower_bound, command_upper_bound
						)
				)

		actuation += result[ 0 ]
		state += pendulum( state, actuation, **model_kwargs ) * mpc_config[ 'time_step' ]

		tf = perf_counter()
		compute_time = tf - ti

		mpc_config[ 'initial_state' ] = state
		mpc_config[ 'initial_actuation' ] = actuation

		previous_states_record.append( deepcopy( state ) )
		previous_actuation_record.append( deepcopy( actuation ) )
		previous_objective_record.append( pendulum_objective( state, actuation, **model_kwargs ) )

		n_f_eval = len( mpc_config[ 'state_record' ] )

		logger.log( f"{actuation=}" )
		logger.log( f"state={state[ : state.shape[ 0 ] // 2 ]}" )
		logger.log( f"objective={pendulum_objective( state, actuation, **model_kwargs )}" )
		logger.log( f"{compute_time=:.6f}s" )
		logger.log( f"{n_f_eval=}" )

		ti = perf_counter()

		fig = plt.figure()
		view = plt.subplot2grid( (4, 5), (0, 0), 4, 3, fig )
		view.grid( True )
		view.set_xlabel( "x" )
		view.set_ylabel( "y" )

		ax_pos = plt.subplot2grid( (4, 5), (0, 3), 1, 2, fig )
		ax_pos.set_ylabel( 'position' )
		ax_pos.yaxis.set_label_position( "right" )
		ax_pos.yaxis.tick_right()

		ax_ang = plt.subplot2grid( (4, 5), (1, 3), 1, 2, fig )
		ax_ang.set_ylabel( 'angle' )
		ax_ang.yaxis.set_label_position( "right" )
		ax_ang.yaxis.tick_right()

		ax_obj = plt.subplot2grid( (4, 5), (2, 3), 1, 2, fig )
		ax_obj.set_ylabel( 'Ek - Ep' )
		ax_obj.yaxis.set_label_position( "right" )
		ax_obj.yaxis.tick_right()

		ax_act = plt.subplot2grid( (4, 5), (3, 3), 1, 2, fig )
		ax_act.set_ylabel( 'actuation' )
		ax_act.set_xlabel( 'time' )
		ax_act.yaxis.set_label_position( "right" )
		ax_act.yaxis.tick_right()

		plt.subplots_adjust( hspace = 0., wspace = 0. )
		fig.suptitle( f"{frame + 1}/{n_frames} - {compute_time=:.6f}s - {n_f_eval=}" )

		ax_obj.set_ylim( -20, 40 )
		ax_pos.set_ylim( -3, 3 )
		ax_ang.set_ylim( - pi, 2 * pi )
		ax_act.set_ylim( command_lower_bound, command_upper_bound )

		x, theta, _, _ = state
		l = mpc_config[ 'model_kwargs' ][ "arm_length" ]
		X = [ x, x + l * sin( theta ), ]
		Y = [ 0, - l * cos( theta ) ]
		view.plot( X[ :2 ], Y[ :2 ], 'b', linewidth = 5 )
		view.plot( X[ 1: ], Y[ 1: ], 'g', linewidth = 5 )
		view.scatter( X, Y, c = 'r', s = 100 )

		how = 11. / 8.9
		view.set_xlim( x - 1.1 * l, x + 1.1 * l )
		view.set_ylim( -1.1 * l * how, 1.1 * l * how )

		time_previous = [ i * mpc_config[ 'time_step' ] - (frame + 1) * mpc_config[ 'time_step' ] for i
											in range( frame + 2 ) ]
		time_prediction = [ i * mpc_config[ 'time_step' ] for i in range(
				mpc_config[ 'optimization_horizon' ] + mpc_config[ 'prediction_horizon' ]
				) ]

		t1 = 0.
		timespan = time_prediction[ -1 ] - time_previous[ 0 ]
		for index in range( len( previous_target_record ) ):
			t2 = (previous_target_record[ index ][ 0 ] - 2 * mpc_config[ 'time_step' ]) / timespan
			ax_pos.axhline(
					previous_target_record[ index ][ 1 ][ 0 ], t1, t2, color = 'r', linewidth = 5
					)
			ax_ang.axhline(
					previous_target_record[ index ][ 1 ][ 1 ], t1, t2, color = 'g', linewidth = 5
					)
			t1 = t2 + mpc_config[ 'time_step' ]

		ax_pos.axhline( mpc_config[ 'target_pose' ][ 0 ], t1, 1, color = 'r', linewidth = 5 )
		ax_ang.axhline( mpc_config[ 'target_pose' ][ 1 ], t1, 1, color = 'g', linewidth = 5 )

		ax_pos.plot( time_previous, array( previous_states_record )[ :, 0 ], 'b' )
		ax_ang.plot( time_previous, array( previous_states_record )[ :, 1 ], 'b' )
		ax_obj.plot( time_previous, previous_objective_record, 'b' )
		ax_act.plot( time_previous, previous_actuation_record, 'b' )

		for f_eval in range( n_f_eval ):
			state_record_array = array( mpc_config[ 'state_record' ][ f_eval ] )
			ax_pos.plot( time_prediction, state_record_array[ :, 0 ], linewidth = .1 )
			ax_ang.plot( time_prediction, state_record_array[ :, 1 ], linewidth = .1 )
			ax_obj.plot( time_prediction, mpc_config[ 'objective_record' ][ f_eval ], linewidth = .1 )
			ax_act.plot( time_prediction, mpc_config[ 'actuation_record' ][ f_eval ], linewidth = .1 )

		# plot vertical line from y min to y max
		ax_obj.axvline( color = 'g' )
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
