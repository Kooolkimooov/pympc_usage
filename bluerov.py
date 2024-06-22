import time

import numpy as np
import scipy
import scipy.linalg
import scipy.spatial
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
		"robot_max_actuation"      : np.array( [ 40, 40, 40, 40, 40, 40 ] ),
		"robot_max_actuation_ramp" : np.array( [ 80, 80, 80, 80, 80, 80 ] )
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
	Fb = buoyancy * np.array( [ 0, 0, -1 ] )
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
	time_step = 0.05
	n_frames = 100
	robust_horizon = 7
	prediction_horizon = 10
	euclidean_cost = True
	final_cost = True
	result_shape = (robust_horizon, actuation_dim)
	result = np.zeros( result_shape )
	max_iter = 1000
	tolerance = 1e-3
	target = np.array( [ 1, 1, 1, 0, 0, np.pi ] )
	error_weight_matrix = np.eye(target.size)
	error_weight_matrix[:3,:3] *= 2.
	# error_weight_matrix[3:,3:] *= .1
	final_cost_weight = 2.
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

	note = 'cost_and_final_weight'

	actual_states = [ state ]
	actual_actuations = [ actuation ]

	# create folder for plots
	folder = (f'./plots/'
						f'{model.__name__}_'
						f'{note}_'
						f'{time_step=}_'
						f'{robust_horizon=}_'
						f'{prediction_horizon=}_'
						f'{max_iter=}_'
						f'{tolerance=}_'
						f'{n_frames=}_'
						f'{euclidean_cost=}_'
						f'{final_cost=}'
      )

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

	print( f"{folder = }" )

	for frame in range( n_frames ):

		print( f"{frame+1 = }/{n_frames}", end = ' ', flush = True )

		all_states = [ ]
		all_actuations = [ ]

		ti = time.perf_counter()
		# model predictive control
		result = optimize(
				model = model,
				cost_function = model_predictive_control_cost_function,
				target = target,
				initial_guess = result,
				current_actuation = actuation,
				optimization_horizon = robust_horizon,
				prediction_horizon = prediction_horizon,
				error_weight_matrix=error_weight_matrix,
				state = state,
				time_step = time_step,
				tolerance = tolerance,
				max_iter = max_iter,
				model_args = model_args,
				bounds = Bounds(
						command_derivative_lower_bound * time_step, command_derivative_upper_bound * time_step
						),
				constraints = (NonlinearConstraint(
						lambda u: (actuation + np.cumsum( u.reshape( result_shape ), axis = 0 )).flatten(),
						command_lower_bound,
						command_upper_bound
						),),
				state_history = all_states,
				actuation_history = all_actuations,
				activate_euclidean_cost = euclidean_cost,
				activate_final_cost = final_cost,
    			final_cost_weight=final_cost_weight
				)

		tf = time.perf_counter()
		result = result.reshape( result_shape )
		actuation += result[ 0 ]
		state += model( state, actuation, **model_args ) * time_step
		compute_time = tf - ti

		actual_states.append( state )
		actual_actuations.append( actuation )

		print( f"\tnf_eval = {len( all_actuations )}", end = ' ', flush = True )
		print( f"\t{compute_time = :.6f}s", end = ' ', flush = True )
		print( f"\t{state[:3].T = }", end = ' ', flush = True )
		print( f"\t{actuation[:3].T = }", end = ' ', flush = True )

		ti = time.perf_counter()
  
		# plot results in subplots
		fig = plt.figure()#( figsize = (16, 9) )  # subplot shape is (y, x)
		bot = plt.subplot2grid( (3, 5), (0, 0), 3, 3, fig, projection='3d' )
		bot.view_init(elev=15, azim=45+180)
		bot.set_xlabel("x")
		bot.set_ylabel("y")
		bot.set_zlabel("z")
		bot.set_xlim( [ -2, 2 ] )
		bot.set_ylim( [ -2, 2 ] )
		bot.set_zlim( [ 0, 4 ] )
		bot.invert_yaxis()
		bot.invert_zaxis()
  
		ax_pos = plt.subplot2grid( (3, 5), (0, 3), 1, 2, fig )
		ax_pos.set_ylabel( 'position' )
		ax_pos.yaxis.set_label_position("right")
		ax_pos.yaxis.tick_right()
		ax_pos.set_ylim( -1, 2 )
  
		ax_ang = plt.subplot2grid( (3, 5), (1, 3), 1, 2, fig )
		ax_ang.set_ylabel( 'angle' )
		ax_ang.yaxis.set_label_position("right")
		ax_ang.yaxis.tick_right()
		ax_ang.set_ylim( -2 * np.pi, 2 * np.pi )
  
		ax_act = plt.subplot2grid( (3, 5), (2, 3), 1, 2, fig )
		ax_act.set_ylabel( 'actuation' )
		ax_act.yaxis.set_label_position("right")
		ax_act.yaxis.tick_right()
		# ax_act.set_ylim( 
        #           -max(bluerov_configuration[ 'robot_max_actuation' ]), 
        #           max(bluerov_configuration[ 'robot_max_actuation' ])
        #           )
  
		plt.subplots_adjust( hspace = 0., wspace = 0.5 )
		fig.suptitle( f"{frame + 1}/{n_frames} - compute time: {compute_time:.6f}s" )

		state_r = scipy.spatial.transform.Rotation.from_euler('xyz', state[3:6]).as_matrix()
		target_r = scipy.spatial.transform.Rotation.from_euler('xyz', target[3:]).as_matrix()

		quiver_scale = .3
		bot.quiver( *state[:3], *(state_r@(quiver_scale * np.ones((3, 1)))))
		bot.quiver( *target[:3], *(target_r@(quiver_scale * np.ones((3, 1)))), color='r')
  
		time_axis_states = [ -(len( actual_states ) - 1) * time_step + i * time_step for i in
												 range( len( actual_states ) + len( all_states[ 0 ] ) - 1 ) ]
		time_axis_actuations = [ -(len( actual_actuations ) - 1) * time_step + i * time_step for i in
														 range( len( actual_actuations ) + len( all_actuations[ 0 ] ) - 1 ) ]

		t1 = 0.
		timespan = time_axis_states[-1] - time_axis_states[0]
		for axis in range(3):
			ax_pos.axhline(
				target[ axis ], t1, (time_axis_states[-1] + frame * time_step) / timespan, linewidth = 5
				)
			ax_ang.axhline(
				target[ axis + 3 ], t1, (time_axis_states[-1] + frame * time_step) / timespan, linewidth = 5
				)
		
		actual_states = np.array( actual_states )
		actual_actuations = np.array( actual_actuations )
  
		for i in range( len( all_states ) ):
			for axis in range(3):
				ax_pos.plot(
					time_axis_states,
					actual_states[ :, axis ].tolist() + all_states[ i ][ 1:, axis ].tolist(),
					'b',
					linewidth = .1
					)
				ax_ang.plot(
					time_axis_states,
					actual_states[ :, axis + 3 ].tolist() + all_states[ i ][ 1:, axis + 3 ].tolist(),
					'b',
					linewidth = .1
					)
				ax_act.plot(
					time_axis_actuations,
					actual_actuations[:, axis].tolist() + [ actuation[axis] + a for a in all_actuations[ i ][1:, axis].cumsum() ],
					'b',
					linewidth = .1
					)
				ax_act.plot(
					time_axis_actuations,
					actual_actuations[:, axis + 3].tolist() + [ actuation[axis + 3] + a for a in all_actuations[ i ][1:, axis + 3].cumsum() ],
					'b',
					linewidth = .1
					)
			bot.plot(
					all_states[ i ][ 1:, 0 ],
					all_states[ i ][ 1:, 1 ],
					all_states[ i ] [1:, 2],
					'b',
					linewidth = .1
					)

		actual_states = actual_states.tolist()
		actual_actuations = actual_actuations.tolist()
  
		# plot vertical line from y min to y max
		ax_pos.axvline( color = 'g' )
		ax_ang.axvline( color = 'g' )
		ax_act.axvline( color = 'g' )

		plt.savefig( f'{folder}/{frame}.png', dpi = 100 )
		plt.close( 'all' )
		del fig
  
		tf = time.perf_counter()
		save_time = tf - ti
		print( f"\t{save_time = :.6f}s", end = ' ', flush = True )
		print()

	# create gif from frames
	print( 'creating gif ...', end = ' ' )
	names = [ image for image in glob.glob( f"{folder}/*.png" ) ]
	names.sort( key = lambda x: os.path.getmtime( x ) )
	frames = [ Image.open( name ) for name in names ]
	frame_one = frames[ 0 ]
	frame_one.save(
			f"{folder}/animation.gif", append_images = frames, loop = True, save_all = True
			)
	print( f'saved at {folder}/animation.gif' )

