from numpy import cos, pi, sin

from mpc import *


# pendulum with cart
def pendulum(
		x: np.ndarray,
		u: float,
		cart_mass: float = 1,
		arm_length: float = 1,
		mass: float = 1, ) -> np.ndarray:

	g = 9.80665
	x, theta, dx, dtheta = x

	A = np.array(
			[ [ 1, 0, 0, 0 ], [ 0, 1, 0, 0 ], [ 0, 0, cos( theta ), arm_length ],
				[ 0, 0, 1 + cart_mass / mass, arm_length * cos( theta ) ] ]
			)

	b = np.array(
			[ dx, dtheta, -g * sin( theta ), arm_length * dtheta ** 2 * sin( theta ) + u / mass ]
			)

	xdot = np.linalg.solve( A, b )

	return xdot


def pendulum_objective(
		x: np.ndarray, u: float, cart_mass: float = 1, arm_length: float = 1, mass: float = 1
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
	# model, initial state and all parameters
	model = pendulum
	state = np.array( [ 0., 0, 0., 0. ] )
	actuation = 0.
	time_step = 0.025
	n_frames = 100
	horizon = 100
	result = np.zeros( (horizon,) )
	max_iter = 1000
	tolerance = 1e-6
	target = np.array( [ -1, pi ] )
	model_args = {
			"cart_mass": 1, "arm_length": 1, "mass": 1
			}
	command_upper_bound = 50
	command_lower_bound = -50
	command_derivative_upper_bound = 5
	command_derivative_lower_bound = -5

	actual_states = [ state ]
	actual_actuations = [ actuation ]

	# create folder for plots
	folder = (
		f'./plots/{model.__name__}_{time_step}_{horizon}_{model_args[ "cart_mass" ]}_'
		f'{model_args[ "arm_length" ]}_{model_args[ "mass" ]}_'
		f'{max_iter}_{tolerance}_{n_frames}_{command_lower_bound}_'
		f'{command_upper_bound}_{command_derivative_lower_bound}_'
		f'{command_derivative_upper_bound}')

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

	for frame in range( n_frames ):

		# if frame == n_frames // 2:
		# 	target = np.array( [ 1, pi ] )

		print( f"frame {frame + 1}/{n_frames}", end = ' ' )

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
				constraints = NonlinearConstraint(
						lambda x: actuation + np.cumsum( x ), command_lower_bound, command_upper_bound
						),
				state_history = all_states,
				actuation_history = all_actuations,
				objective = pendulum_objective,
				# activate_euclidean_cost = False,
				# activate_final_cost = False
				)

		actuation += result[ 0 ]

		actual_states.append( state )
		actual_actuations.append( actuation )

		# update state (Euler integration, maybe RK in future?)
		state += model( state, actuation, **model_args ) * time_step
		tf = time.perf_counter()
		print(
				f"actuation: {actuation} - state: {state[ :2 ]} - Ep: "
				f"{pendulum_objective( state, actuation, **model_args )}", end = ' '
				)

		# ramp
		# target += 1 / n_frames
		# sine
		# target = 1 + np.sin( frame / n_frames * 2 * np.pi )

		print( f"- {tf - ti:.6f}s", end = ' ' )

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

		x, theta, _, _ = state
		l = model_args[ "arm_length" ]
		X = [ x, x + l * sin( theta ), ]
		Y = [ 0, - l * cos( theta ) ]
		pendulum.plot( X[ :2 ], Y[ :2 ], 'b', linewidth = 5 )
		pendulum.plot( X[ 1: ], Y[ 1: ], 'g', linewidth = 5 )
		pendulum.scatter( X, Y, c = 'r', s = 100 )
		pendulum.set_xlim( x - l, x + l )
		pendulum.set_ylim( -l, l )

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
			f"{folder}/{folder.split( '/' )[ -1 ]}.gif",
			append_images = frames,
			loop = True,
			save_all = True
			)
	print( f'saved at {folder}/{folder.split( "/" )[ -1 ]}.gif' )
