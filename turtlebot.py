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
	n_frames = 90
	horizon = 15
	result_shape = (horizon, 2)
	result = np.zeros( result_shape )
	max_iter = 1000
	tolerance = 1e-3
	target_index = 0
	targets = np.array(
			[ [ 1., 1., np.pi ], [ -1., -1., -np.pi ], [ 1., 0, 0 ], [ 0, 0, np.pi ], [ 0, 1., 0 ] ]
			).T
	targets_times = []
	pose_weight_matrix = np.eye( 3 )
	pose_weight_matrix[ 2, 2 ] = 0
	model_args = { }
	command_upper_bound = 50
	command_lower_bound = -50
	command_derivative_upper_bound = 5
	command_derivative_lower_bound = -5

	actual_states = [ deepcopy( state ) ]
	actual_actuations = [ deepcopy( actuation ) ]

	note = ""
	note = "angle_weight=0"

	# create folder for plots
	folder = (f'./plots/{model.__name__}_{note}_{time_step=}_{horizon=}_'
						f'{max_iter=}_{tolerance=}_{n_frames=}_{command_lower_bound=}_'
						f'{command_upper_bound=}_{command_derivative_lower_bound=}_'
						f'{command_derivative_upper_bound=}')

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
		target = targets[ target_index ]
		if np.linalg.norm( (state[ :3 ] - target) @ pose_weight_matrix ) < 0.1:
			target_index = (target_index + 1) % len( targets )
			targets_times.append( frame * time_step)

		if frame == n_frames // 2:
			target = np.array( [ -1., -1., 0 ] )

		print( f"frame {frame + 1}/{n_frames}", end = ' ' )

		all_states = [ ]
		all_actuations = [ ]


		def constraint( x: np.ndarray ):
			c = (actuation + np.cumsum( x.reshape( result.shape ), axis = 0 )).flatten()
			return c


		ti = time.perf_counter()
		# model predictive control

		result = optimize(
				model = model,
				cost_function = model_predictive_control_cost_function,
				target = target,
				initial_guess = result[ :horizon ],
				current_actuation = actuation,
				optimization_horizon = horizon,
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
				final_cost_weight = 5,
				error_weight_matrix = pose_weight_matrix
				)

		actuation += result[ 0 ]

		state += model( state, actuation, **model_args ) * time_step
		tf = time.perf_counter()
		actual_states.append( deepcopy( state ) )
		actual_actuations.append( deepcopy( actuation ) )

		# update state (Euler integration, maybe RK in future?)
		print(
				f"actuation: {actuation} - state: {state[ :3 ]}", end = ' '
				)

		# ramp
		# target += 1 / n_frames
		# sine
		# target = 1 + np.sin( frame / n_frames * 2 * np.pi )

		print( f"- {tf - ti:.6f}s", end = ' ' )

		# plot results in subplots
		fig = plt.figure()#( figsize = (16, 9) )  # subplot shape is (y, x)
		bot = plt.subplot2grid( (3, 5), (0, 0), 3, 3, fig )
		bot.grid(True)
		bot.set_xlabel("x")
		bot.set_ylabel("y")
  
		ax_pos = plt.subplot2grid( (3, 5), (0, 3), 1, 2, fig )
		ax_pos.set_ylabel( 'position' )
		ax_pos.yaxis.set_label_position("right")
		ax_pos.yaxis.tick_right()
  
		ax_ang = plt.subplot2grid( (3, 5), (1, 3), 1, 2, fig )
		ax_ang.set_ylabel( 'angle' )
		ax_ang.yaxis.set_label_position("right")
		ax_ang.yaxis.tick_right()
  
		ax_act = plt.subplot2grid( (3, 5), (2, 3), 1, 2, fig )
		ax_act.set_ylabel( 'actuation' )
		ax_act.yaxis.set_label_position("right")
		ax_act.yaxis.tick_right()
  
		plt.subplots_adjust( hspace = 0., wspace = 0. )
		fig.suptitle( f"{frame + 1}/{n_frames} - compute time: {tf - ti:.6f}s" )

		x, y, theta, _, _, _ = state
		bot.quiver( x, y, .1 * cos( theta ), .1 * sin( theta ) )
		bot.quiver( target[ 0 ], target[ 1 ], .1 * cos( target[ 2 ] ), .1 * sin( target[ 2 ] ), color = 'r')
  
		how = 8.4 / 6.7
		bot.set_xlim( [ -2, 2 ] )
		bot.set_ylim( [ -how*2, how*2 ] )

		time_axis_states = [ -(len( actual_states ) - 1) * time_step + i * time_step for i in
												 range( len( actual_states ) + len( all_states[ 0 ] ) - 1 ) ]
		time_axis_actuations = [ -(len( actual_actuations ) - 1) * time_step + i * time_step for i in
														 range( len( actual_actuations ) + len( all_actuations[ 0 ] ) - 1 ) ]

		t1 = 0.
		timespan = time_axis_states[-1] - time_axis_states[0]
		for time_index in range( len( targets_times ) ):
			t2 = (targets_times[time_index] + time_step) / timespan
			ax_pos.axhline( targets[time_index % len( targets )][0], t1, t2, color = 'r', linewidth = 5 )
			ax_pos.axhline( targets[time_index % len( targets )][1], t1, t2, color = 'g', linewidth = 5 )
			ax_ang.axhline( targets[time_index % len( targets )][2], t1, t2, color = 'r', linewidth = 5 )
			t1 = t2 + 2 * time_step
   
		ax_pos.axhline(
				target[ 0 ], t1, (time_axis_states[-1] + frame * time_step) / timespan, color = 'r', linewidth = 5
				)
		ax_pos.axhline(
				target[ 1 ], t1, (time_axis_states[-1] + frame * time_step) / timespan, color = 'g', linewidth = 5
				)
		ax_ang.axhline(
				target[ 2 ], t1, (time_axis_states[-1] + frame * time_step) / timespan, color = 'r', linewidth = 5
				)

		state_start = 0 if len( time_axis_states ) < 2 * horizon else len(
				time_axis_states
				) - 2 * horizon
		actuation_start = 0 if len( time_axis_actuations ) < 2 * horizon else len(
				time_axis_actuations
				) - 2 * horizon
  
		state_start = 0
		actuation_start = 0

		ax_pos.plot(
				time_axis_states[ state_start:len( actual_states ) ],
				np.array( actual_states )[ state_start:, 0 ],
				'b',
				linewidth = 2
				)
		ax_pos.plot(
				time_axis_states[ state_start:len( actual_states ) ],
				np.array( actual_states )[ state_start:, 1 ],
				'b',
				linewidth = 2
				)
		ax_ang.plot(
				time_axis_states[ state_start:len( actual_states ) ],
				np.array( actual_states )[ state_start:, 2 ],
				'b',
				linewidth = 2
				)
		ax_act.plot(
				time_axis_actuations[ actuation_start:len( actual_actuations ) ],
				np.array( actual_actuations )[ actuation_start:, 0 ],
				'b',
				linewidth = 2
				)
		ax_act.plot(
				time_axis_actuations[ actuation_start:len( actual_actuations ) ],
				np.array( actual_actuations )[ actuation_start:, 1 ],
				'b',
				linewidth = 2
				)

		for i in range( len( all_states ) ):
			width = .05
			style = ':'
			if np.all( all_actuations[ i ] == result[ 0 ] ):
				width = 2
				style = '-'
			ax_pos.plot(
					time_axis_states[ len( actual_states ) - 1:-1 ],
					all_states[ i ][ 1:, 0 ],
					'b',
					linestyle = style,
					linewidth = width
					)
			ax_pos.plot(
					time_axis_states[ len( actual_states ) - 1:-1 ],
					all_states[ i ][ 1:, 1 ],
					'b',
					linestyle = style,
					linewidth = width
					)
			ax_ang.plot(
					time_axis_states[ len( actual_states ) - 1:-1 ],
					all_states[ i ][ 1:, 2 ],
					'b',
					linestyle = style,
					linewidth = width
					)
			ax_act.plot(
					time_axis_actuations[ len( actual_actuations ) - 1:-1 ],
					actual_actuations[ -1 ][ 0 ] - all_actuations[ i ][ 1, 0 ] + all_actuations[ i ][ 1:,
																																			 0 ].cumsum().tolist(),
					'b',
					linestyle = style,
					linewidth = width
					)
			ax_act.plot(
					time_axis_actuations[ len( actual_actuations ) - 1:-1 ],
					actual_actuations[ -1 ][ 1 ] - all_actuations[ i ][ 1, 1 ] + all_actuations[ i ][ 1:,
																																			 1 ].cumsum().tolist(),
					'b',
					linestyle = style,
					linewidth = width
					)
			bot.plot(
					all_states[ i ][ 1:, 0 ],
					all_states[ i ][ 1:, 1 ],
					'b',
					linewidth = width
					)

		# plot vertical line from y min to y max
		ax_pos.axvline( color = 'g' )
		ax_ang.axvline( color = 'g' )
		ax_act.axvline( color = 'g' )

		plt.savefig( f'{folder}/{frame}.png', dpi = 100 )
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
			f"{folder}/animation.gif",
			append_images = frames,
			loop = True,
			save_all = True
			)
	print( f'saved at {folder}/animation.gif' )
