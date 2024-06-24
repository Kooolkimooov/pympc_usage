from glob import glob
from json import dump
from os import mkdir, path, remove
from time import perf_counter, time

from matplotlib import pyplot as plt
from numpy import array, concatenate, cos, cumsum, eye, pi, sin
from numpy.linalg import solve
from PIL import Image

from mpc import *


# double pendulum with cart
def double_pendulum(
		x: ndarray,
		u: ndarray,
		cart_mass: float = 1,
		first_arm_length: float = 1,
		first_arm_mass: float = 1,
		second_arm_length: float = 1,
		second_arm_mass: float = 1
		) -> ndarray:

	g = 9.80665
	x, theta_1, theta_2, dx, dtheta_1, dtheta_2 = x

	l1 = first_arm_length / 2
	l2 = second_arm_length / 2
	J1 = (first_arm_mass * l1 ** 2) / 3
	J2 = (second_arm_mass * l2 ** 2) / 3

	h1 = cart_mass + first_arm_mass + second_arm_mass
	h2 = first_arm_mass * l1 + second_arm_mass * second_arm_length
	h3 = second_arm_mass * l2
	h4 = first_arm_mass * l1 ** 2 + second_arm_mass * second_arm_length ** 2 + J1
	h5 = second_arm_mass * l2 * first_arm_length
	h6 = second_arm_mass * l2 ** 2 + J2
	h7 = (first_arm_mass * l1 + second_arm_mass * second_arm_length) * g
	h8 = second_arm_mass * l2 * g

	A = array(
			[ [ 1, 0, 0, 0, 0, 0 ], [ 0, 1, 0, 0, 0, 0 ], [ 0, 0, 1, 0, 0, 0 ],
				[ 0, 0, 0, h1, h2 * cos( theta_1 ), h3 * cos( theta_2 ) ],
				[ 0, 0, 0, h2 * cos( theta_1 ), h4, h5 * cos( theta_1 - theta_2 ) ],
				[ 0, 0, 0, h3 * cos( theta_2 ), h5 * cos( theta_1 - theta_2 ), h6 ] ]
			)

	b = array(
			[ dx, dtheta_1, dtheta_2,
				h2 * dtheta_1 ** 2 * sin( theta_1 ) + h3 * dtheta_2 ** 2 * sin( theta_2 ) + u[ 0 ],
				h7 * sin( theta_1 ) - h5 * dtheta_2 ** 2 * sin( theta_1 - theta_2 ),
				h5 * dtheta_1 ** 2 * sin( theta_1 - theta_2 ) + h8 * sin( theta_2 ) ]
			)

	xdot = solve( A, b )

	return xdot


def double_pendulum_objective(
		x: ndarray,
		u: ndarray,
		cart_mass: float = 1,
		first_arm_length: float = 1,
		first_arm_mass: float = 1,
		second_arm_length: float = 1,
		second_arm_mass: float = 1
		) -> float:

	x, theta_1, theta_2, dx, dtheta_1, dtheta_2 = x
	g = 9.80665

	l1 = first_arm_length / 2
	l2 = second_arm_length / 2
	J1 = (first_arm_mass * l1 ** 2) / 3
	J2 = (second_arm_mass * l2 ** 2) / 3

	Ep = first_arm_mass * l1 * g * cos( theta_1 ) + second_arm_mass * g * (
			first_arm_length * cos( theta_1 ) + l2 * cos( theta_2 ))

	Ek_c = .5 * cart_mass * dx ** 2
	Ek_1 = .5 * first_arm_mass * ((dx + l1 * dtheta_1 * cos( theta_1 )) ** 2 + (
			l1 * dtheta_1 * sin( theta_1 )) ** 2) + .5 * J1 * dtheta_1 ** 2
	Ek_2 = .5 * second_arm_mass * ((dx + first_arm_length * dtheta_1 * cos(
			theta_1
			) + l2 * dtheta_2 * cos( theta_2 )) ** 2 + (first_arm_length * dtheta_1 * sin(
			theta_1
			) + l2 * dtheta_2 * sin( theta_2 )) ** 2) + .5 * J2 * dtheta_2 ** 2
	Ek = Ek_c + Ek_1 + Ek_2

	# objective is to minimize kinetic energy and maximize potiential energy
	return (Ek - Ep) ** 3


if __name__ == "__main__":

	state = array( [ 0., pi, pi, 0., 0., 0. ] )
	actuation = array( [ 0. ] )

	model_kwargs = {
			"cart_mass"        : .6,
			"first_arm_mass"   : .2,
			"second_arm_mass"  : .2,
			"first_arm_length" : .5,
			"second_arm_length": .5
			}

	base_optimization_horizon = 50
	optimization_horizon = base_optimization_horizon
	time_steps_per_actuation = 2

	optimization_horizon_lower_bound = 50

	pose_weight_matrix = eye( state.shape[ 0 ] // 2 )
	# pose_weight_matrix[ 0, 0 ] = 0.
	# pose_weight_matrix[ 1, 1 ] = 0.
	# pose_weight_matrix[ 2, 2 ] = 0.

	actuation_weight_matrix = .01 * eye( actuation.shape[ 0 ] )

	mpc_config = {
			'candidate_shape'         : (
					optimization_horizon // time_steps_per_actuation + 1, actuation.shape[ 0 ]),
			'model'                   : double_pendulum,
			'initial_actuation'       : actuation,
			'initial_state'           : state,
			'model_kwargs'            : model_kwargs,
			'target_pose'             : array(
					[ 0., 0., 0. ]
					),
			'optimization_horizon'    : optimization_horizon,
			'prediction_horizon'      : 0,
			'time_step'               : 0.025,
			'time_steps_per_actuation': time_steps_per_actuation,
			'objective_function'      : double_pendulum_objective,
			'pose_weight_matrix'      : pose_weight_matrix,
			'actuation_weight_matrix' : actuation_weight_matrix,
			'objective_weight'        : 100.,
			'final_cost_weight'       : 1.,
			'state_record'            : [ ],
			'actuation_record'        : [ ],
			'objective_record'        : [ ],
			'verbose'                 : False
			}

	result = zeros( mpc_config[ 'candidate_shape' ] )

	command_upper_bound = 50
	command_lower_bound = -50
	command_derivative_upper_bound = int( 25 / mpc_config[ 'time_step' ] )
	command_derivative_lower_bound = int( -25 / mpc_config[ 'time_step' ] )

	n_frames = 300

	max_iter = 1000
	tolerance = 1e-6

	previous_states_record = [ deepcopy( state ) ]
	previous_actuation_record = [ deepcopy( actuation ) ]
	previous_objective_record = [ double_pendulum_objective( state, actuation, **model_kwargs ) ]
	previous_target_record = [ ]

	note = ''

	folder = (f'./plots/{double_pendulum.__name__}_'
						f'{note}_'
						f'dt={mpc_config[ "time_step" ]}_'
						f'opth={optimization_horizon}_'
						f'preh=_{mpc_config[ "prediction_horizon" ]}'
						f'dtpu={time_steps_per_actuation}_'
						f'{max_iter=}_'
						f'{tolerance=}_'
						f'{n_frames=}_'
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
		dump( mpc_config, f, default = serialize_others )

	for frame in range( n_frames ):

		if optimization_horizon > optimization_horizon_lower_bound:
			optimization_horizon -= 1

		if frame == 100 and False:
			previous_target_record.append(
					(frame * mpc_config[ 'time_step' ], deepcopy( mpc_config[ 'target_pose' ] ))
					)
			mpc_config[ 'target_pose' ] = array( [ -1., 0., 0. ] )
			optimization_horizon = base_optimization_horizon

		mpc_config[ 'optimization_horizon' ] = optimization_horizon
		mpc_config[ 'candidate_shape' ] = (
				optimization_horizon // time_steps_per_actuation + 1, actuation.shape[ 0 ])

		mpc_config[ 'state_record' ] = [ ]
		mpc_config[ 'actuation_record' ] = [ ]
		mpc_config[ 'objective_record' ] = [ ]

		print( f"frame {frame + 1}/{n_frames}\t", end = ' ' )

		result = result[ 1:mpc_config[ 'candidate_shape' ][ 0 ] ]
		difference = result.shape[ 0 ] - mpc_config[ 'candidate_shape' ][ 0 ]
		if difference < 0:
			result = concatenate( (result, array( [ [ 0. ] ] * abs( difference ) )) )

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
		state += double_pendulum( state, actuation, **model_kwargs ) * mpc_config[ 'time_step' ]

		tf = perf_counter()
		compute_time = tf - ti

		mpc_config[ 'initial_state' ] = state
		mpc_config[ 'initial_actuation' ] = actuation

		previous_states_record.append( deepcopy( state ) )
		previous_actuation_record.append( deepcopy( actuation ) )
		previous_objective_record.append(
				double_pendulum_objective( state, actuation, **model_kwargs )
				)

		n_f_eval = len( mpc_config[ 'state_record' ] )

		print(
				f"actuation={actuation}\t"
				f"state={state[ : state.shape[ 0 ] // 2 ]}\t"
				f"objective={double_pendulum_objective( state, actuation, **model_kwargs )}\t"
				f"{compute_time=:.6f}s - {n_f_eval=}\t", end = ' '
				)

		ti = perf_counter()

		x, theta, phi = state[ :3 ]
		l1 = model_kwargs[ "first_arm_length" ]
		l2 = model_kwargs[ "second_arm_length" ]

		time_previous = [ i * mpc_config[ 'time_step' ] - (frame + 1) * mpc_config[ 'time_step' ] for i
											in range( frame + 2 ) ]
		time_prediction = [ i * mpc_config[ 'time_step' ] for i in range(
				mpc_config[ 'optimization_horizon' ] + mpc_config[ 'prediction_horizon' ]
				) ]

		fig = plt.figure()
		view = plt.subplot2grid( (4, 5), (0, 0), 4, 3, fig )
		view.grid( True )
		view.set_xlabel( "x" )
		view.set_ylabel( "y" )
		how = 11. / 8.9
		view.set_xlim( x - (l1 + l2), x + (l1 + l2) )
		view.set_ylim( -(l1 + l2) * how, (l1 + l2) * how )

		ax_pos = plt.subplot2grid( (4, 5), (0, 3), 1, 2, fig )
		ax_pos.set_ylabel( 'position' )
		ax_pos.yaxis.set_label_position( "right" )
		ax_pos.yaxis.tick_right()
		ax_pos.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_pos.set_ylim( -3, 3 )

		ax_ang = plt.subplot2grid( (4, 5), (1, 3), 1, 2, fig )
		ax_ang.set_ylabel( 'angle' )
		ax_ang.yaxis.set_label_position( "right" )
		ax_ang.yaxis.tick_right()
		ax_ang.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_ang.set_ylim( - pi, 2 * pi )

		ax_obj = plt.subplot2grid( (4, 5), (2, 3), 1, 2, fig )
		ax_obj.set_ylabel( 'Ek - Ep' )
		ax_obj.yaxis.set_label_position( "right" )
		ax_obj.yaxis.tick_right()
		ax_obj.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_obj.set_ylim( -20, 40 )

		ax_act = plt.subplot2grid( (4, 5), (3, 3), 1, 2, fig )
		ax_act.set_ylabel( 'actuation' )
		ax_act.set_xlabel( 'time' )
		ax_act.yaxis.set_label_position( "right" )
		ax_act.yaxis.tick_right()
		ax_act.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_act.set_ylim( command_lower_bound, command_upper_bound )

		plt.subplots_adjust( hspace = 0., wspace = 0. )
		fig.suptitle( f"{frame + 1}/{n_frames} - {compute_time=:.6f}s - {n_f_eval=}" )

		X = [ x, x + l1 * sin( theta ), x + l1 * sin( theta ) + l2 * sin( phi ) ]
		Y = [ 0, l1 * cos( theta ), l1 * cos( theta ) + l2 * cos( phi ) ]
		view.plot( X, Y, 'b', linewidth = 5 )
		view.scatter( X, Y, c = 'r', s = 100 )

		t1 = 0.
		timespan = time_prediction[ -1 ] - time_previous[ 0 ]
		for index in range( len( previous_target_record ) ):
			t2 = (previous_target_record[ index ][ 0 ] - 2 * mpc_config[ 'time_step' ]) / timespan
			ax_pos.axhline(
					previous_target_record[ index ][ 1 ][ 0 ],
					t1,
					t2,
					color = 'b',
					linestyle = ':',
					linewidth = pose_weight_matrix[ 0, 0 ] + .0001
					)
			ax_ang.axhline(
					previous_target_record[ index ][ 1 ][ 1 ],
					t1,
					t2,
					color = 'b',
					linestyle = ':',
					linewidth = pose_weight_matrix[ 1, 1 ] + .0001
					)
			ax_ang.axhline(
					previous_target_record[ index ][ 1 ][ 2 ],
					t1,
					t2,
					color = 'r',
					linestyle = ':',
					linewidth = pose_weight_matrix[ 2, 2 ] + .0001
					)
			t1 = t2 + mpc_config[ 'time_step' ]

		ax_pos.axhline(
				mpc_config[ 'target_pose' ][ 0 ],
				t1,
				1,
				color = 'b',
				linestyle = ':',
				linewidth = pose_weight_matrix[ 0, 0 ] + .0001
				)
		ax_ang.axhline(
				mpc_config[ 'target_pose' ][ 1 ],
				t1,
				1,
				color = 'b',
				linestyle = ':',
				linewidth = pose_weight_matrix[ 1, 1 ] + .0001
				)
		ax_ang.axhline(
				mpc_config[ 'target_pose' ][ 2 ],
				t1,
				1,
				color = 'r',
				linestyle = ':',
				linewidth = pose_weight_matrix[ 2, 2 ] + .0001
				)

		ax_pos.plot( time_previous, array( previous_states_record )[ :, 0 ], 'b' )
		ax_ang.plot( time_previous, array( previous_states_record )[ :, 1 ], 'b' )
		ax_ang.plot( time_previous, array( previous_states_record )[ :, 2 ], 'r' )
		ax_obj.plot( time_previous, previous_objective_record, 'b' )
		ax_act.plot( time_previous, previous_actuation_record, 'b' )

		for f_eval in range( n_f_eval ):
			state_record_array = array( mpc_config[ 'state_record' ][ f_eval ] )
			ax_pos.plot( time_prediction, state_record_array[ :, 0 ], 'b', linewidth = .1 )
			ax_ang.plot( time_prediction, state_record_array[ :, 1 ], 'b', linewidth = .1 )
			ax_ang.plot( time_prediction, state_record_array[ :, 2 ], 'r', linewidth = .1 )
			ax_obj.plot(
					time_prediction, mpc_config[ 'objective_record' ][ f_eval ], 'b', linewidth = .1
					)
			ax_act.plot(
					time_prediction, mpc_config[ 'actuation_record' ][ f_eval ], 'b', linewidth = .1
					)

		# plot vertical line from y min to y max
		ax_pos.axvline( color = 'g' )
		ax_ang.axvline( color = 'g' )
		ax_obj.axvline( color = 'g' )
		ax_act.axvline( color = 'g' )

		plt.savefig( f'{folder}/{frame}.png' )
		plt.close( 'all' )
		del fig

		tf = perf_counter()
		save_time = tf - ti

		print( f'saved figure in {save_time:.6f}s\t', end = '' )
		print()

	# create gif from frames
	print( 'creating gif ...', end = ' ' )
	names = [ image for image in glob( f"{folder}/*.png" ) ]
	names.sort( key = lambda x: path.getmtime( x ) )
	frames = [ Image.open( name ) for name in names ]
	frame_one = frames[ 0 ]
	frame_one.save(
			f"{folder}/animation.gif", append_images = frames, loop = True, save_all = True
			)
	print( f'saved at {folder}/animation.gif' )
