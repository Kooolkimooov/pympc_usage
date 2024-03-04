from numpy import cos, pi, sin

from mpc import *


# double pendulum with cart
def double_pendulum(
		x: np.ndarray,
		u: float,
		cart_mass: float = 1,
		first_arm_length: float = 1,
		first_arm_mass: float = 1,
		second_arm_length: float = 1,
		second_arm_mass: float = 1
		) -> np.ndarray:

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

	A = np.array(
			[ [ 1, 0, 0, 0, 0, 0 ],
				[ 0, 1, 0, 0, 0, 0 ],
				[ 0, 0, 1, 0, 0, 0 ],
				[ 0, 0, 0, h1, h2 * cos( theta_1 ), h3 * cos( theta_2 ) ],
				[ 0, 0, 0, h2 * cos( theta_1 ), h4, h5 * cos( theta_1 - theta_2 ) ],
				[ 0, 0, 0, h3 * cos( theta_2 ), h5 * cos( theta_1 - theta_2 ), h6 ] ]
			)

	b = np.array(
			[ dx,
				dtheta_1,
				dtheta_2,
				h2 * dtheta_1 ** 2 * sin( theta_1 ) + h3 * dtheta_2 ** 2 * sin( theta_2 ) + u,
				h7 * sin( theta_1 ) - h5 * dtheta_2 ** 2 * sin( theta_1 - theta_2 ),
				h5 * dtheta_1 ** 2 * sin( theta_1 - theta_2 ) + h8 * sin( theta_2 ) ]
			)

	xdot = np.linalg.solve( A, b )

	return xdot


def double_pendulum_objective(
		x: np.ndarray,
		u: float,
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
	return Ek - Ep


if __name__ == "__main__":
	# model, initial state and all parameters
	model = double_pendulum
	state = np.array( [ 0., pi, pi, 0, 0., 0. ] )
	actuation = 0.
	time_step = 0.025
	n_frames = 200
	horizon = 75
	euclidean_cost = True
	final_cost = True
	result = np.zeros( (horizon,) )
	max_iter = 1001
	tolerance = 1e-6
	target = np.array( [ -1, 0, 0 ] )
	model_args = {
			"cart_mass"        : .6,
			"first_arm_mass"   : .2,
			"second_arm_mass"  : .2,
			"first_arm_length" : .5,
			"second_arm_length": .5
			}
	command_upper_bound = 50
	command_lower_bound = -50
	command_derivative_upper_bound = 5
	command_derivative_lower_bound = -5
	note = f'receding_horizon-25_change_of_target'

	actual_states = [ state ]
	actual_actuations = [ actuation ]

	# create folder for plots
	folder = (
			f'./plots/{model.__name__}_{time_step}_{horizon}_{model_args[ "cart_mass" ]}_'
			f'{model_args[ "first_arm_mass" ]}_{model_args[ "second_arm_mass" ]}_'
			f'{model_args[ "first_arm_length" ]}_{model_args[ "second_arm_length" ]}_'
			f'{max_iter}_{tolerance}_{n_frames}_{euclidean_cost}_{final_cost}_{command_lower_bound}_'
			f'{command_upper_bound}_{command_derivative_lower_bound}_'
			f'{command_derivative_upper_bound}_{note}')

	if os.path.exists( folder ):
		files_in_dir = glob.glob( f'{folder}/*.png' )
		if len( files_in_dir ) > 0:
			if input( f"{folder} contains data. Remove? (y/n) " ) == 'y':
				for fig in files_in_dir:
					os.remove( fig )
			else:
				exit()
	else:
		os.mkdir( folder )

	print( f"executing {folder}" )

	for frame in range( n_frames ):
		horizon = horizon - 1 if horizon > 25 else 25

		# if frame == n_frames // 2:
		# 	target = np.array( [ 1, 0, 0 ] )
		# 	horizon = 75
		# 	result = np.concatenate((result, np.zeros( (horizon - len(result),) )))

		print( f"frame {frame + 1}/{n_frames}", end = ' ', flush = True )

		all_states = [ ]
		all_actuations = [ ]

		ti = time.perf_counter()
		# model predictive control
		result = model_predictive_control(
				model = model,
				cost = cost,
				target = target,
				last_result = result[ :horizon ],
				current_actuation = actuation,
				robust_horizon = horizon,
				state = state,
				time_step = time_step,
				tolerance = tolerance,
				max_iter = max_iter,
				model_args = model_args,
				bounds = Bounds( command_derivative_lower_bound, command_derivative_upper_bound ),
				constraints = (NonlinearConstraint(
						lambda u: actuation + np.cumsum( u ), command_lower_bound, command_upper_bound
						),
                   ),
				state_history = all_states,
				actuation_history = all_actuations,
				objective = double_pendulum_objective,
				activate_euclidean_cost = euclidean_cost,
				activate_final_cost = final_cost
				)

		actuation += result[ 0 ]

		actual_states.append( state )
		actual_actuations.append( actuation )

		# update state (Euler integration, maybe RK in future?)
		state += model( state, actuation, **model_args ) * time_step
		tf = time.perf_counter()
		print(
				f"- actuation: {actuation} - state: {state[ :3 ]} - objective: "
				f"{double_pendulum_objective( state, actuation, **model_args )}", end = ' ', flush = True
				)

		print( f"- compute time {tf - ti:.6f}s", end = ' ', flush = True )

		# plot results in subplots
		fig = plt.figure( figsize = (16, 9) )  # subplot shape is (y, x)
		pendulum = plt.subplot2grid( (3, 5), (0, 0), 3, 3, fig )
		ax0 = plt.subplot2grid( (3, 5), (0, 3), 1, 2, fig )
		ax1 = plt.subplot2grid( (3, 5), (1, 3), 1, 2, fig )
		ax2 = plt.subplot2grid( (3, 5), (2, 3), 1, 2, fig )
		plt.subplots_adjust( hspace = 0., wspace = .5 )
		fig.suptitle( f"{frame + 1}/{n_frames} - compute time: {tf - ti:.6f}s" )
		ax0.set_ylabel( 'position' )
		ax1.set_ylabel( 'angle' )
		ax2.set_ylabel( 'actuation' )

		min_x = np.array( actual_states )[ :, 0 ].min() if len( actual_states ) > 1 and np.array(
				actual_states
				)[ :, 0 ].min() < -2 else -2
		max_x = np.array( actual_states )[ :, 0 ].max() if len( actual_states ) > 1 and np.array(
				actual_states
				)[ :, 0 ].max() > 2 else 2

		ax0.set_ylim( min_x, max_x )
		ax1.set_ylim( -2 * pi, 2 * pi )
		ax2.set_ylim( command_lower_bound, command_upper_bound )

		x, theta_1, theta_2, _, _, _ = state
		l1 = model_args[ "first_arm_length" ]
		l2 = model_args[ "second_arm_length" ]
		X = [ x, x + l1 * sin( theta_1 ), x + l1 * sin( theta_1 ) + l2 * sin( theta_2 ) ]
		Y = [ 0, l1 * cos( theta_1 ), l1 * cos( theta_1 ) + l2 * cos( theta_2 ) ]
		pendulum.plot( X[ :2 ], Y[ :2 ], 'b', linewidth = 5 )
		pendulum.plot( X[ 1: ], Y[ 1: ], 'g', linewidth = 5 )
		pendulum.scatter( X, Y, c = 'r', s = 100 )
		pendulum.set_xlim( -2, 2 )
		pendulum.set_ylim( -2, 2 )
		pendulum.axhline(color="k")
		# pendulum.set_xlim( x - l1 - l2, x + l1 + l2 )
		# pendulum.set_ylim( -l1 - l2, l1 + l2 )

		time_axis_states = [ -(len( actual_states ) - 1) * time_step + i * time_step for i in
												 range( len( actual_states ) + len( all_states[ 0 ] ) - 1 ) ]
		time_axis_actuations = [ -(len( actual_actuations ) - 1) * time_step + i * time_step for i in
														 range( len( actual_actuations ) + len( all_actuations[ 0 ] ) - 1 ) ]

		ax0.axhline(
				target[ 0 ], color = 'r', linewidth = 5
				)
		ax1.axhline(
				target[ 1 ], color = 'r', linewidth = 5
				)
		ax1.axhline(
				target[ 2 ], color = 'b', linewidth = 5
				)

		actual_states = np.array( actual_states )

		for i in range( len( all_states ) ):
			ax0.plot(
					time_axis_states,
					actual_states[ :, 0 ].tolist() + all_states[ i ][ 1:, 0 ].tolist(),
					'b',
					linewidth = .1
					)
			ax1.plot(
					time_axis_states,
					actual_states[ :, 1 ].tolist() + all_states[ i ][ 1:, 1 ].tolist(),
					'r',
					linewidth = .1
					)
			ax1.plot(
					time_axis_states,
					actual_states[ :, 2 ].tolist() + all_states[ i ][ 1:, 2 ].tolist(),
					'b',
					linewidth = .1
					)
			ax2.plot(
					time_axis_actuations,
					actual_actuations + [ actuation + a for a in
																all_actuations[ i ].flatten()[ 1: ].cumsum().tolist() ],
					'b',
					linewidth = .1
					)

		actual_states = actual_states.tolist()

		# plot vertical line from y min to y max
		ax0.axvline( color = 'g' )
		ax1.axvline( color = 'g' )
		ax2.axvline( color = 'g' )

		plt.savefig( f'{folder}/{frame}.png' )
		plt.close( 'all' )
		del fig
		print()

	# create gif from frames
	print( 'creating gif ...', end = ' ' )
	names = [ image for image in glob.glob( f"{folder}/*.png" ) ]
	names.sort( key = lambda x: os.path.getmtime( x ) )
	frames = [ Image.open( name ) for name in names ]
	frame_one = frames[ 0 ]
	frame_one.save(
			f"{folder}/gif.gif",
			append_images = frames,
			loop = True,
			save_all = True
			)
	print( f'saved at {folder}/gif.gif' )

	# with scipy.optimize.minimize
	# Days: 0
	# Hours: 0
	# Minutes: 4
	# Seconds: 56
	# Milliseconds: 874
	# Ticks: 2968743334
	# TotalDays: 0, 00343604552546296
	# TotalHours: 0, 0824650926111111
	# TotalMinutes: 4, 94790555666667
	# TotalSeconds: 296, 8743334
	# TotalMilliseconds: 296874, 3334

	# with ipopt
	# Days: 0
	# Hours: 4
	# Minutes: 18
	# Seconds: 41
	# Milliseconds: 571
	# Ticks: 155215710386
	# TotalDays: 0, 179647812946759
	# TotalHours: 4, 31154751072222
	# TotalMinutes: 258, 692850643333
	# TotalSeconds: 15521, 5710386
	# TotalMilliseconds: 15521571, 03 86


