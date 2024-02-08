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
			[ [ 1, 0, 0, 0, 0, 0 ], [ 0, 1, 0, 0, 0, 0 ], [ 0, 0, 1, 0, 0, 0 ],
				[ 0, 0, 0, h1, h2 * cos( theta_1 ), h3 * cos( theta_2 ) ],
				[ 0, 0, 0, h2 * cos( theta_1 ), h4, h5 * cos( theta_1 - theta_2 ) ],
				[ 0, 0, 0, h3 * cos( theta_2 ), h5 * cos( theta_1 - theta_2 ), h6 ] ]
			)

	b = np.array(
			[ dx, dtheta_1, dtheta_2,
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
	_, theta_1, theta_2, _, _, _ = x
	g = 9.80665
	l1 = first_arm_length / 2
	l2 = second_arm_length / 2

	E = first_arm_mass * l1 * g * cos( theta_1 ) + second_arm_mass * g * (
			first_arm_length * cos( theta_1 ) + l2 * cos( theta_2 ))

	# objective is to minimize potential energy
	return - E


if __name__ == "__main__":
	# model, initial state and all parameters
	model = double_pendulum
	state = np.array( [ 0., pi, pi, 0., 0., 0. ] )
	time_step = 0.05
	n_frames = 150
	horizon = 150
	max_iter = 1000
	tolerance = 1e-6
	target = np.array( [ 0, 0, 0 ] )
	model_args = {
			"cart_mass"        : 1,
			"first_arm_length" : 1,
			"first_arm_mass"   : 1,
			"second_arm_length": 1,
			"second_arm_mass"  : 1
			}
	command_upper_bound = 200
	command_lower_bound = -200
	command_derivative_upper_bound = 50
	command_derivative_lower_bound = -50

	actual_states = [ ]
	actual_actuations = [ ]

	min_x, max_x = -1, 1

	# create folder for plots
	folder = (f'./plots/{model.__name__}_{time_step}_{horizon}_'
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
		horizon = n_frames - frame if n_frames - frame > 10 else 10

		print( f"frame {frame + 1}/{n_frames}", end = ' ' )

		all_states = [ ]
		all_actuations = [ ]

		ti = time.perf_counter()
		# model predictive control
		actuation = model_predictive_control(
				model = model,
				cost = cost,
				target = target,
				command_dimension = 1,
				horizon = horizon,
				state = state,
				time_step = time_step,
				tolerance = tolerance,
				max_iter = max_iter,
				model_args = model_args,
				bounds = Bounds( command_lower_bound, command_upper_bound ),
				constraints = NonlinearConstraint(
						lambda x: np.diff( x ) / time_step,
						np.ones( horizon - 1 ) * command_derivative_lower_bound,
						np.ones( horizon - 1 ) * command_derivative_upper_bound
						),
				state_history = all_states,
				actuation_history = all_actuations,
				objective = double_pendulum_objective,
				# activate_euclidean_cost = False,
				# activate_final_cost = False
				)

		actual_states.append( state )
		actual_actuations.append( actuation )

		# update state (Euler integration, maybe RK in future?)
		state += model( state, actuation, **model_args ) * time_step
		print(
				f"actuation: {actuation} - state: {state[ :3 ]} - Ep: "
				f"{double_pendulum_objective( state, actuation, **model_args )}", end = ' '
				)

		# ramp
		# target += 1 / n_frames
		# sine
		# target = 1 + np.sin( frame / n_frames * 2 * np.pi )

		tf = time.perf_counter()
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
		ax1.set_ylabel( 'angles' )
		ax2.set_ylabel( 'actuation' )
		ax0.set_ylim( min_x, max_x )
		ax1.set_ylim( -2 * pi, 2 * pi )

		x, theta_1, theta_2, _, _, _ = state
		l1 = model_args[ "first_arm_length" ]
		l2 = model_args[ "second_arm_length" ]
		X = [ x, x + l1 * sin( theta_1 ), x + l1 * sin( theta_1 ) + l2 * sin( theta_2 ) ]
		Y = [ 0, l1 * cos( theta_1 ), l1 * cos( theta_1 ) + l2 * cos( theta_2 ) ]
		pendulum.plot( X[ :2 ], Y[ :2 ], 'b', linewidth = 5 )
		pendulum.plot( X[ 1: ], Y[ 1: ], 'g', linewidth = 5 )
		pendulum.scatter( X, Y, c = 'r', s = 100 )
		pendulum.set_xlim( x - (l1 + l2), x + l1 + l2 )
		pendulum.set_ylim( -(l1 + l2), (l1 + l2) )

		time_axis_states = [ -(len( actual_states ) - 1) * time_step + i * time_step for i in
												 range( len( actual_states ) + len( all_states[ 0 ][ 1:, 0 ] ) ) ]
		time_axis_actuations = [ -(len( actual_actuations ) - 1) * time_step + i * time_step for i in
														 range( len( actual_actuations ) + len( all_actuations[ 0 ] ) ) ]

		ax0.axhline(
				target[ 0 ], color = 'r', linewidth = 5
				)
		ax1.axhline(
				target[ 1 ], color = 'r', linewidth = 5
				)
		ax1.axhline(
				target[ 2 ], color = 'r', linewidth = 5
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
			ax1.plot(
					time_axis_states,
					actual_states[ :, 2 ].tolist() + all_states[ i ][ 1:, 2 ].tolist(),
					'g',
					linewidth = .1
					)
			ax2.plot(
					time_axis_actuations,
					actual_actuations + all_actuations[ i ].tolist(),
					'b',
					linewidth = .1
					)

		min_x = actual_states[ :, 0 ].min() if actual_states[ :, 0 ].min() < -1 else -1
		max_x = actual_states[ :, 0 ].max() if actual_states[ :, 0 ].max() > 1 else 1
		actual_states = actual_states.tolist()

		# plot vertical line from y min to y max
		ax1.axvline( color = 'g' )
		ax2.axvline( color = 'g' )

		plt.savefig( f'{folder}/{frame}.png' )
		plt.close( 'all' )
		print()

	# create gif from frames
	print( 'creating gif ...' )
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
