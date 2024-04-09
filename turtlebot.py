from numpy import cos, pi, sin

from mpc import *


# pendulum with cart
def turtle(
		x: np.ndarray, u: np.array
		) -> np.ndarray:

	xdot = np.zeros( x.shape )
	_, _, theta, _, _, _ = x
	v, w = u
	xdot[ 0 ] = v * cos( theta )
	xdot[ 1 ] = v * sin( theta )
	xdot[ 2 ] = w

	return xdot


if __name__ == "__main__":
	# model, initial state and all parameters
	model = turtle
	state = np.array( [ 0., 0., 0., 0., 0., 0. ] )
	actuation = np.array( [ 0., 0. ] )
	time_step = 0.01
	n_frames = 100
	horizon = 7
	result_shape  = (horizon, 2)
	result = np.zeros( result_shape )
	max_iter = 1000
	tolerance = 1e-3
	target = np.array( [ 1., 1., np.pi ] )
	model_args = { }
	command_upper_bound = 50
	command_lower_bound = -50
	command_derivative_upper_bound = 5
	command_derivative_lower_bound = -5

	actual_states = [ state ]
	actual_actuations = [ actuation ]

	# create folder for plots
	folder = (f'./plots/{model.__name__}_{time_step}_{horizon}_'
						f'{max_iter}_{tolerance}_{n_frames}_{command_lower_bound}_'
						f'{command_upper_bound}_{command_derivative_lower_bound}_'
						f'{command_derivative_upper_bound}')

	if os.path.exists( folder ):
		files_in_dir = glob.glob( f'{folder}/*' )
		if len( files_in_dir ) > 0:
			if input( f"{folder} contains data. Remove? (y/n) " ) == 'y':
				for fig in files_in_dir:
					os.remove( fig )
			else:
				exit()
	else:
		os.mkdir( folder )

	for frame in range( n_frames ):

		if frame == n_frames // 2:
			target = np.array( [ -1., -1., 0 ] )

		print( f"frame {frame + 1}/{n_frames}", end = ' ' )

		all_states = [ ]
		all_actuations = [ ]

		def constraint(x: np.ndarray):
			c = (actuation + np.cumsum( x.reshape(result.shape), axis=0 )).flatten()
			return c

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
				# constraints = NonlinearConstraint(constraint, command_upper_bound, command_lower_bound),
				state_history = all_states,
				actuation_history = all_actuations,
				# activate_euclidean_cost = False,
				# activate_final_cost = False,
				# final_cost_weight = .10
				)

		actuation += result[ 0 ]

		actual_states.append( state )
		actual_actuations.append( actuation )

		# update state (Euler integration, maybe RK in future?)
		state += model( state, actuation, **model_args ) * time_step
		tf = time.perf_counter()
		print(
				f"actuation: {actuation} - state: {state[ :3 ]}", end = ' '
				)

		# ramp
		# target += 1 / n_frames
		# sine
		# target = 1 + np.sin( frame / n_frames * 2 * np.pi )

		print( f"- {tf - ti:.6f}s", end = ' ' )

		# plot results in subplots
		fig = plt.figure( figsize = (16, 9) )  # subplot shape is (y, x)
		bot = plt.subplot2grid( (3, 5), (0, 0), 3, 3, fig )
		ax0 = plt.subplot2grid( (3, 5), (0, 3), 1, 2, fig )
		ax1 = plt.subplot2grid( (3, 5), (1, 3), 1, 2, fig )
		ax2 = plt.subplot2grid( (3, 5), (2, 3), 1, 2, fig )
		plt.subplots_adjust( hspace = 0., wspace = .5 )
		fig.suptitle( f"{frame + 1}/{n_frames} - compute time: {tf - ti:.6f}s" )
		ax0.set_ylabel( 'position' )
		ax1.set_ylabel( 'angle' )
		ax2.set_ylabel( 'actuation' )

		x, y, theta, _, _, _ = state
		bot.quiver( x, y, .1 * cos(theta), .1 * sin(theta) )
		bot.scatter(target[0], target[1], c = 'b', s = 10)
		bot.set_xlim( [-2, 2] )
		bot.set_ylim( [-2, 2] )

		time_axis_states = [ -(len( actual_states ) - 1) * time_step + i * time_step for i in
												 range( len( actual_states ) + len( all_states[ 0 ] ) - 1 ) ]
		time_axis_actuations = [ -(len( actual_actuations ) - 1) * time_step + i * time_step for i in
														 range( len( actual_actuations ) + len( all_actuations[ 0 ] ) - 1 ) ]

		ax0.axhline(
				target[ 0 ], color = 'r', linewidth = 5
				)
		ax0.axhline(
				target[ 1 ], color = 'g', linewidth = 5
				)
		ax1.axhline(
				target[ 2 ], color = 'r', linewidth = 5
				)

		actual_states = np.array( actual_states )
		actual_actuations = np.array( actual_actuations )

		for i in range( len( all_states ) ):
			ax0.plot(
					time_axis_states,
					actual_states[ :, 0 ].tolist() + all_states[ i ][ 1:, 0 ].tolist(),
					'b',
					linewidth = .1
					)
			ax0.plot(
					time_axis_states,
					actual_states[ :, 1 ].tolist() + all_states[ i ][ 1:, 1 ].tolist(),
					'b',
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
					actual_states[ :, 0 ].tolist() + all_actuations[ i ][ 1:, 0 ].tolist(),
					'b',
					linewidth = .1
					)
			ax2.plot(
					time_axis_actuations,
					actual_actuations[ :, 1 ].tolist() + all_actuations[ i ][ 1:, 1 ].tolist(),
					'b',
					linewidth = .1
					)

		actual_states = actual_states.tolist()
		actual_actuations = actual_actuations.tolist()

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
