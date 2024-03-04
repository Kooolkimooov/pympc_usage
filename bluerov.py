import time

import numpy as np
from mpc import *

mass = 11.5
inertial_coefficients = np.array( [ .16, .16, .16, 0.0, 0.0, 0.0 ] )
center_of_mass = np.array( [ 0.0, 0.0, 0.0 ] )
inertial_matrix = np.eye( 6 )
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

bluerov_configuration = {
		"mass"                     : mass,
		"center_of_mass"           : center_of_mass,
		"buoyancy"                 : 120.0,
		"center_of_volume"         : np.array( [ 0.0, 0.0, - 0.02 ] ),
		"inertial_matrix_inv"      : np.linalg.inv( inertial_matrix ),
		"hydrodynamic_coefficients": np.array(
				[ 4.03, 6.22, 5.18, 0.07, 0.07, 0.07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
				),
		"robot_max_actuation"      : np.array( [ 1000, 1000, 1000, 1000, 1000, 1000 ] ),
		"robot_max_actuation_ramp" : np.array( [ 100, 100, 100, 100, 100, 100 ] )
		}


# double pendulum with cart
def robot(
		x: np.ndarray, u: np.ndarray, robot_configuration = None
		) -> np.ndarray:
	if robot_configuration is None:
		global bluerov_configuration
		robot_configuration = bluerov_configuration
	Phi, Theta, Psi = x[ 3 ], x[ 4 ], x[ 5 ]
	cPhi, sPhi = np.cos( Phi ), np.sin( Phi )
	cTheta, sTheta, tTheta = np.cos( Theta ), np.sin( Theta ), np.tan( Theta )
	cPsi, sPsi = np.cos( Psi ), np.sin( Psi )
	J = np.zeros( (6, 6) )
	J[ 0, :3 ] = np.array(
			[ cPsi * cTheta, -sPsi * cPhi + cPsi * sTheta * sPhi, sPsi * sPhi + cPsi * sTheta * cPhi ]
			)
	J[ 1, :3 ] = np.array(
			[ sPsi * cTheta, cPsi * cPhi + sPsi * sTheta * sPhi, -cPsi * sPhi + sPsi * sTheta * cPhi ]
			)
	J[ 2, :3 ] = np.array( [ -sTheta, cTheta * sPhi, cTheta * cPhi ] )
	J[ 3, 3: ] = np.array( [ 1, sPhi * tTheta, cPhi * tTheta ] )
	J[ 4, 3: ] = np.array( [ 0, cPhi, -sPhi ] )
	J[ 5, 3: ] = np.array( [ 0, sPhi / cTheta, cPhi / cTheta ] )

	hydrodynamic_coefficients = robot_configuration[ "hydrodynamic_coefficients" ]
	D = np.diag( hydrodynamic_coefficients[ :6 ] ) + np.diag(
			np.multiply( hydrodynamic_coefficients[ 6: ], abs( x[ 6: ] ) )
			)
	buoyancy = robot_configuration[ "buoyancy" ]
	center_of_volume = robot_configuration[ "center_of_volume" ]
	Fw = mass * np.array( [ 0, 0, 9.80665 ] )
	Fb = buoyancy * np.array( [ 0, 0, 1 ] )
	S = np.zeros( 6 )
	S[ :3 ] = J[ :3, :3 ].T @ (Fw + Fb)
	S[ 3: ] = J[ :3, :3 ].T @ (np.cross( center_of_mass, Fw ) + np.cross( center_of_volume, Fb ))

	I_inv = robot_configuration[ 'inertial_matrix_inv' ]
	xdot = np.zeros( x.shape )
	xdot[ :6 ] = J @ x[ 6: ]
	xdot[ 6: ] = I_inv @ (D @ x[ 6: ] + S + u)

	return xdot


if __name__ == "__main__":
	# model, initial state and all parameters
	model = robot
	state = np.array( [ 0., 0, 0, 0, 0., 0., 0., 0, 0, 0, 0., 0. ] )
	actuation = np.array( [ 0., 0., 0., 0., 0., 0. ] )
	actuation_dim = len( actuation )
	time_step = 0.5
	n_frames = 25
	robust_horizon = 4
	# prediction_horizon = 10
	euclidean_cost = True
	final_cost = True
	result_shape = (robust_horizon, actuation_dim)
	result = np.zeros( result_shape )
	max_iter = 1000
	tolerance = 1e-6
	target = np.array( [ .5, -.5, 1, 0, 0, np.pi ] )
	model_args = { 'robot_configuration': bluerov_configuration }
	command_upper_bound = np.array(
			[ bluerov_configuration[ 'robot_max_actuation' ] ] * robust_horizon
			).flatten()
	command_lower_bound = - np.array(
			[ bluerov_configuration[ 'robot_max_actuation' ] ] * robust_horizon
			).flatten()
	command_derivative_upper_bound = np.array(
			[ bluerov_configuration[ 'robot_max_actuation_ramp' ] ] * robust_horizon
			).flatten()
	command_derivative_lower_bound = - np.array(
			[ bluerov_configuration[ 'robot_max_actuation_ramp' ] ] * robust_horizon
			).flatten()

	note = f'robust_horizon'

	actual_states = [ state ]
	actual_actuations = [ actuation ]

	# create folder for plots
	folder = (f'./plots/'
						f'{model.__name__=}_'
						f'{time_step=}_'
						f'{robust_horizon=}_'
						# f'{prediction_horizon=}_'
						f'{max_iter=}_'
						f'{tolerance=}_'
						f'{n_frames=}_'
						f'{euclidean_cost=}_'
						f'{final_cost=}_'
						f'{note=}')

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

	print( f"{folder = }" )

	for frame in range( n_frames ):

		print( f"{frame+1 = }/{n_frames}", end = ' ', flush = True )

		all_states = [ ]
		all_actuations = [ ]

		print(state)
		ti = time.perf_counter()
		# model predictive control
		result = model_predictive_control(
				model = model,
				cost = cost,
				target = target,
				last_result = result,
				current_actuation = actuation,
				robust_horizon = robust_horizon,
				# prediction_horizon = prediction_horizon,
				state = state,
				time_step = time_step,
				tolerance = tolerance,
				max_iter = max_iter,
				model_args = model_args,
				bounds = Bounds(
						command_derivative_lower_bound, command_derivative_upper_bound
						),
				constraints = (NonlinearConstraint(
						lambda u: (actuation + np.cumsum( u.reshape( result_shape ), axis = 0 )).flatten(),
						command_lower_bound,
						command_upper_bound
						),),
				state_history = all_states,
				actuation_history = all_actuations,
				activate_euclidean_cost = euclidean_cost,
				activate_final_cost = final_cost
				)

		tf = time.perf_counter()
		print(state)
		result = result.reshape( result_shape )
		actuation += result[ 0 ]
		state += model( state, actuation, **model_args ) * time_step
		compute_time = tf - ti

		actual_states.append( state )
		actual_actuations.append( actuation )

		# update state (Euler integration, maybe RK in future?)
		print( f"{actuation = }", end = ' ', flush = True )
		print( f"{state[ :6 ] = }", end = ' ', flush = True )
		print( f"nfeval = {len( all_actuations )}", end = ' ', flush = True )
		print( f"{compute_time = :.6f}s", end = ' ', flush = True )

		ti = time.perf_counter()
		# plot results in subplots
		fig = plt.figure( figsize = (16, 9) )
		ax0, ax1, ax2 = fig.subplots( 3, 1 )
		plt.subplots_adjust( hspace = 0., wspace = .5 )
		fig.suptitle( f"{frame + 1}/{n_frames} - {compute_time = :.6f}s" )
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
		ax1.set_ylim( -2 * np.pi, 2 * np.pi )

		time_axis_states = [ -(len( actual_states ) - 1) * time_step + i * time_step for i in
												 range( len( actual_states ) + len( all_states[ 0 ] ) - 1 ) ]
		time_axis_actuations = [ -(len( actual_actuations ) - 1) * time_step + i * time_step for i in
														 range( len( actual_actuations ) + len( all_actuations[ 0 ] ) - 1 ) ]

		ax0.axhline( target[ 0 ], color = 'r', linewidth = 5 )
		ax0.axhline( target[ 1 ], color = 'g', linewidth = 5 )
		ax0.axhline( target[ 2 ], color = 'b', linewidth = 5 )
		ax1.axhline( target[ 3 ], color = 'r', linewidth = 5 )
		ax1.axhline( target[ 4 ], color = 'g', linewidth = 5 )
		ax1.axhline( target[ 5 ], color = 'b', linewidth = 5 )

		actual_states = np.array( actual_states )
		actual_actuations = np.array( actual_actuations )

		for i in range( len( all_states ) if len( all_states ) < 5000 else 5000 ):
			ax0.plot(
					time_axis_states,
					actual_states[ :, 0 ].tolist() + all_states[ i ][ 1:, 0 ].tolist(),
					'r',
					linewidth = .1
					)
			ax0.plot(
					time_axis_states,
					actual_states[ :, 1 ].tolist() + all_states[ i ][ 1:, 1 ].tolist(),
					'g',
					linewidth = .1
					)
			ax0.plot(
					time_axis_states,
					actual_states[ :, 2 ].tolist() + all_states[ i ][ 1:, 2 ].tolist(),
					'b',
					linewidth = .1
					)
			ax1.plot(
					time_axis_states,
					actual_states[ :, 3 ].tolist() + all_states[ i ][ 1:, 3 ].tolist(),
					'r',
					linewidth = .1
					)
			ax1.plot(
					time_axis_states,
					actual_states[ :, 5 ].tolist() + all_states[ i ][ 1:, 4 ].tolist(),
					'g',
					linewidth = .1
					)
			ax1.plot(
					time_axis_states,
					actual_states[ :, 5 ].tolist() + all_states[ i ][ 1:, 5 ].tolist(),
					'b',
					linewidth = .1
					)
			for axis in range( 6 ):
				ax2.plot(
						time_axis_actuations,
						actual_actuations[ :, axis ].tolist() + all_actuations[ i ][ 1:, axis ].tolist(),
						'g',
						linewidth = .1
						)

		actual_states = actual_states.tolist()
		actual_actuations = actual_actuations.tolist()

		# plot vertical line from y min to y max
		ax0.axvline( color = 'g' )
		ax1.axvline( color = 'g' )
		ax2.axvline( color = 'g' )

		plt.savefig( f'{folder}/{frame}.png' )
		tf = time.perf_counter()
		plt.close( 'all' )
		del fig
		print( f'fig saved in {tf - ti:.6f}' )

	# create gif from frames
	print( 'creating gif ...', end = ' ' )
	names = [ image for image in glob.glob( f"{folder}/*.png" ) ]
	names.sort( key = lambda x: os.path.getmtime( x ) )
	frames = [ Image.open( name ) for name in names ]
	frame_one = frames[ 0 ]
	frame_one.save(
			f"{folder}/gif.gif", append_images = frames, loop = True, save_all = True
			)
	print( f'saved at {folder}/gif.gif' )

# with scipy.optimize.minimize	# Days: 0	# Hours: 0	# Minutes: 4	# Seconds: 56	# Milliseconds:
# 874	# Ticks: 2968743334	# TotalDays: 0, 00343604552546296	# TotalHours: 0, 0824650926111111	#
# TotalMinutes: 4, 94790555666667	# TotalSeconds: 296, 8743334	# TotalMilliseconds: 296874, 3334

# with ipopt	# Days: 0	# Hours: 4	# Minutes: 18	# Seconds: 41	# Milliseconds: 571	# Ticks:
# 155215710386	# TotalDays: 0, 179647812946759	# TotalHours: 4, 31154751072222	# TotalMinutes:
# 258, 692850643333	# TotalSeconds: 15521, 5710386	# TotalMilliseconds: 15521571, 03 86
