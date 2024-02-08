import glob
import os
import time
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.optimize import Bounds, LinearConstraint, minimize, NonlinearConstraint


# actuated 1d mass spring damper system
def mass_spring_damper(
		x: np.ndarray, u: float, mass: float = 1, dampening: float = 1, spring: float = 1
		) -> np.ndarray:

	# dynamics
	xdot = np.zeros( 2 )
	xdot[ 0 ] = x[ 1 ]
	xdot[ 1 ] = (u - dampening * x[ 1 ] - spring * x[ 0 ]) / mass

	return xdot


def mass_spring_nl_damper(
		x: np.ndarray, u: float, mass: float = 1, dampening: float = 1, spring: float = 1
		) -> np.ndarray:

	# dynamics
	xdot = np.zeros( 2 )
	xdot[ 0 ] = x[ 1 ]
	xdot[ 1 ] = (u - dampening * abs( x[ 1 ] ) * x[ 1 ] - spring * x[ 0 ]) / mass

	return xdot


# cost function for one horizon pass with given model and actuations
def cost(
		actuations: np.ndarray,
		model: callable,
		model_args: dict,
		target: np.ndarray,
		horizon: int,
		state: np.ndarray,
		time_step: float,
		objective: callable,
		activate_euclidean_cost: bool,
		activate_final_cost: bool,
		state_history: list,
		actuation_history: list,
		verb: bool
		) -> float:

	if not activate_euclidean_cost and objective is None:
		raise ValueError( "Cannot compute cost" )

	local_state = deepcopy( state )
	cost = 0.

	if actuation_history is not None:
		actuation_history.append( actuations )

	states = np.zeros( (horizon, len( state )) )

	for i in range( horizon ):
		states[ i ] = local_state
		local_state = local_state + model(
				local_state, actuations[ i ], **model_args
				) * time_step
		if activate_euclidean_cost:
			cost += np.linalg.norm( local_state[ :len( state ) // 2 ] - target ) ** 2
		if objective is not None:
			cost += objective( local_state, actuations[ i ], **model_args )

	cost /= horizon

	if activate_final_cost:
		cost += np.linalg.norm( local_state[ :len( state ) // 2 ] - target ) ** 2
		if objective is not None:
			cost += objective( local_state, actuations[ -1 ], **model_args )

	if state_history is not None:
		state_history.append( states )

	if verb:
		print( f"cost: {cost}; u: {actuations}" )
	return cost


# model predictive control returns the optimal actuation
def model_predictive_control(
		cost: callable,
		model: callable,
		state: np.ndarray,
		target: np.ndarray,
		command_dimension: int,
		horizon: int,
		time_step: float,
		tolerance: float = 1e-6,
		max_iter: int = 100,
		model_args: dict = None,
		bounds: Bounds = None,
		constraints: NonlinearConstraint | LinearConstraint = None,
		objective: callable = None,
		activate_euclidean_cost: bool = True,
		activate_final_cost: bool = True,
		state_history: list = None,
		actuation_history: list = None,
		verb: bool = False
		) -> float:

	initial_guess = np.zeros( (horizon, command_dimension) )
	if command_dimension == 1:
		initial_guess = np.zeros( horizon )

	result = minimize(
			fun = cost,
			x0 = initial_guess,
			args = (
			model, model_args, target, horizon, state, time_step, objective, activate_euclidean_cost,
			activate_final_cost, state_history, actuation_history, verb),
			tol = tolerance,
			bounds = bounds,
			constraints = constraints,
			options = { 'maxiter': max_iter }
			)

	if verb:
		print( result )
	return result.x[ 0 ]


if __name__ == '__main__':
	# model, initial state and all parameters
	model = mass_spring_nl_damper
	state = np.array( [ 0., 0. ] )
	time_step = 0.1
	horizon = 10
	max_iter = 25
	tolerance = 1e-6
	target = 2
	n_frames = 100
	model_args = { 'mass': 5, 'dampening': 4, 'spring': 3 }
	command_upper_bound = 50
	command_lower_bound = -50
	command_derivative_upper_bound = 1
	command_derivative_lower_bound = -1

	actual_states = [ ]
	actual_actuations = [ ]

	# create folder for plots
	folder = (f'./plots/{model.__name__}_{state[ 0 ]}_{state[ 1 ]}_{time_step}_{horizon}_'
						f'{max_iter}_{tolerance}_{target}_{n_frames}_{model_args[ "mass" ]}_'
						f'{model_args[ "dampening" ]}_{model_args[ "spring" ]}_{command_lower_bound}_'
						f'{command_upper_bound}_{command_derivative_lower_bound}_'
						f'{command_derivative_upper_bound}')

	if os.path.exists( folder ):
		files_in_dir = glob.glob( f'{folder}/*.png' )
		if len( files_in_dir ) > 0:
			if input( f"{folder} contains data. Remove? (y/n) " ) == 'y':
				for f in files_in_dir:
					os.remove( f )
			else:
				exit()
	else:
		os.mkdir( folder )

	for frame in range( n_frames ):
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
				activate_final_cost = False
				)

		actual_states.append( state[ 0 ] )
		actual_actuations.append( actuation )

		# update state (Euler integration, maybe RK in future?)
		state += model( state, actuation, **model_args ) * time_step

		# ramp
		# target += 1 / n_frames
		# sine
		# target = 1 + np.sin( frame / n_frames * 2 * np.pi )

		tf = time.perf_counter()
		print( f"- {tf - ti:.6f}s", end = ' ' )

		# plot results in subplots
		fig = plt.figure( figsize = (16, 9) )
		fig.suptitle( f"{frame + 1}/{n_frames} - compute time: {tf - ti:.6f}s" )
		ax1, ax2 = plt.subplot( 211 ), plt.subplot( 212 )
		ax1.set_ylabel( 'position' )
		ax2.set_ylabel( 'actuation' )

		time_axis_states = [ -(len( actual_states ) - 1) * time_step + i * time_step for i in
												 range( len( actual_states ) + len( all_states[ 0 ][ 1:, 0 ] ) ) ]
		time_axis_actuations = [ -(len( actual_actuations ) - 1) * time_step + i * time_step for i in
														 range( len( actual_actuations ) + len( all_actuations[ 0 ] ) ) ]

		ax1.axhline(
				target, color = 'r', linewidth = 5
				)

		for i in range( len( all_states ) ):
			ax1.plot(
					time_axis_states, actual_states + all_states[ i ][ 1:, 0 ].tolist(), 'b', linewidth = .1
					)
			ax2.plot(
					time_axis_actuations,
					actual_actuations + all_actuations[ i ].tolist(),
					'b',
					linewidth = .1
					)

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
