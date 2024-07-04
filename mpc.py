from copy import deepcopy
from inspect import signature

import matplotlib.pyplot as plt
from numpy import ndarray, pi, zeros
from numpy.linalg import norm
from scipy.optimize import Bounds, LinearConstraint, minimize, NonlinearConstraint


def model_predictive_control_cost_function(
		candidate_actuations_derivative: ndarray,
		candidate_shape: tuple,
		model: callable,
		initial_state: ndarray,
		initial_actuation: ndarray,
		model_kwargs: dict,
		target_trajectory: list[ tuple[ float, list ] ],
		objective_function: callable,
		optimization_horizon: int,
		prediction_horizon: int,
		time_step: float,
		time_steps_per_actuation: int,
		pose_weight_matrix: ndarray,
		actuation_weight_matrix: ndarray,
		objective_weight: float,
		final_cost_weight: float,
		state_record: list,
		actuation_record: list,
		objective_record: list,
		verbose: bool
		) -> float:

	if norm( pose_weight_matrix ) == 0 and final_cost_weight == 0 and objective_function is None:
		raise ValueError( "Cannot compute cost" )

	candidate_actuations_derivative = candidate_actuations_derivative.reshape( candidate_shape )
	actuation = deepcopy( initial_actuation )
	state = deepcopy( initial_state )
	horizon = optimization_horizon + prediction_horizon
	cost = 0.
	trajectory_index = 0

	if state_record is not None:
		states = zeros( (horizon, initial_state.shape[ 0 ]) )
	if actuation_record is not None:
		actuations = zeros( (horizon, initial_actuation.shape[ 0 ]) )
	if objective_function is not None:
		objectives = zeros( (horizon, 1) )

	for i in range( optimization_horizon ):

		if i % time_steps_per_actuation == 0:
			actuation_derivative = candidate_actuations_derivative[ i // time_steps_per_actuation ]
			cost += actuation_derivative @ actuation_weight_matrix @ actuation_derivative.T
			actuation += actuation_derivative

		if i * time_step > target_trajectory[ trajectory_index ][ 0 ]:
			trajectory_index += 1

		target_pose = target_trajectory[ trajectory_index ][ 1 ]

		state += model( state, actuation, **model_kwargs ) * time_step

		error = state[ :len( state ) // 2 ] - target_pose
		cost += error @ pose_weight_matrix @ error.T

		if state_record is not None:
			states[ i ] = state

		if actuation_record is not None:
			actuations[ i ] = actuation

		if objective_function is not None:
			objective = objective_function( state, actuation, **model_kwargs )
			cost += objective_weight * objective
			if objective_record is not None:
				objectives[ i ] = objective

	for i in range( prediction_horizon ):

		if (i + optimization_horizon) * time_step > target_trajectory[ trajectory_index ][ 0 ]:
			trajectory_index += 1

		target_pose = target_trajectory[ trajectory_index ][ 1 ]

		state += model( state, actuation, **model_kwargs ) * time_step

		error = state[ :len( state ) // 2 ] - target_pose
		cost += error @ pose_weight_matrix @ error.T

		if state_record is not None:
			states[ optimization_horizon + i ] = state

		if actuation_record is not None:
			actuations[ optimization_horizon + i ] = actuation

		if objective_function is not None:
			objective = objective_function( state, actuation, **model_kwargs )
			cost += objective_weight * objective
			if objective_record is not None:
				objectives[ optimization_horizon + i ] = objective

	error = state[ :len( state ) // 2 ] - target_pose
	cost += final_cost_weight * pow( norm( error @ pose_weight_matrix @ error.T ), 2 )
	if objective_function is not None:
		cost += final_cost_weight * objective_weight * objective_function(
				state, actuation, **model_kwargs
				)

	cost /= horizon

	if state_record is not None:
		state_record.append( states )
	if actuation_record is not None:
		actuation_record.append( actuations )
	if objective_record is not None:
		objective_record.append( objectives )

	if verbose:
		print( f'{cost=}; {candidate_actuations_derivative}' )

	return cost


def optimize(
		cost_function: callable,
		cost_kwargs: dict,
		initial_guess: ndarray,
		tolerance: float = 1e-6,
		max_iter: int = 100,
		bounds: Bounds = None,
		constraints: tuple[ NonlinearConstraint ] | tuple[ LinearConstraint ] = None,
		verbose: bool = False
		) -> ndarray:

	cost_args = ()
	for parameter in signature( cost_function ).parameters:
		if parameter != 'candidate_actuations_derivative':
			cost_args += (cost_kwargs[ parameter ],)

	result = minimize(
			fun = cost_function,
			x0 = initial_guess.flatten(),
			args = cost_args,
			tol = tolerance,
			bounds = bounds,
			constraints = constraints,
			options = { 'maxiter': max_iter }
			)

	if verbose:
		print( result )

	return result.x.reshape( cost_kwargs[ 'candidate_shape' ] )


def generate_trajectory(
		key_frames: list[ tuple[ float, list ] ], n_points: int
		):
	assert key_frames[ 0 ][ 0 ] == 0., "trajectory doesn't start at t = 0."

	n_dim = len( key_frames[ 0 ][ 1 ] )
	timespan = key_frames[ -1 ][ 0 ]
	dt = timespan / n_points
	trajectory = [ (i * dt, [ 0. ] * n_dim) for i in range( n_points ) ]
	start_point = 0

	for frame_index in range( len( key_frames ) - 1 ):
		frame_0 = key_frames[ frame_index ]
		frame_1 = key_frames[ frame_index + 1 ]
		sub_timespan = frame_1[ 0 ] - frame_0[ 0 ]
		sub_n_points = int( n_points * sub_timespan / timespan )

		funcs = [ ]
		for dim in range( n_dim ):
			funcs += [ cubic_interpolation_function( frame_0[ 1 ][ dim ], frame_1[ 1 ][ dim ], 0., 0. ) ]

		for point in range( sub_n_points ):
			for dim in range( n_dim ):
				trajectory[ start_point + point ][ 1 ][ dim ] = funcs[ dim ]( point / sub_n_points )

		start_point += sub_n_points
	for dim in range( n_dim ):
		trajectory[ -1 ][ 1 ][ dim ] = key_frames[ -1 ][ 1 ][ dim ]
	return trajectory


def cubic_interpolation_function( f_0: float, f_1: float, f_0p: float, f_1p: float ):
	a = 2 * f_0 - 2 * f_1 + f_0p + f_1p
	b = -3 * f_0 + 3 * f_1 - 2 * f_0p - f_1p
	c = f_0p
	d = f_0

	def function( x: float ) -> float:
		return a * pow( x, 3 ) + b * pow( x, 2 ) + c * x + d

	return function


def serialize_others( obj: any ) -> str:
	if callable( obj ):
		return obj.__name__
	if isinstance( obj, ndarray ):
		return obj.tolist()


class Logger:
	def __init__( self ):
		self.logs: str = ''

	def log( self, log: str ):
		'''
		:param log: text to be printed and saved. ends with a tabulation
		:return: None
		'''
		print( log, end = '\t' )
		self.logs += log
		self.logs += '\t'

	def lognl( self, log: str ):
		'''
		:param log: text to be printed and saved. ends with a new line
		:return: None
		'''
		print( log )
		self.logs += log
		self.logs += '\n'

	def logrl( self, log: str ):
		'''
		:param log: text to be printed and saved. ends with a return to the beginning of the line,
		the saved text goes to a new line
		:return: None
		'''
		print( log, end = '\r' )
		self.logs += log
		self.logs += '\n'

	def save_at( self, path: str, file: str = 'logs' ):
		"""
		:param path: folder in which to save the current log
		:param file: name of the file
		"""
		self.log( 'saving logs ...' )
		with open( f'{path}/logs.txt', 'w' ) as f:
			f.write( self.logs )
		self.lognl( f'saved at {path}/{file}.txt' )


if __name__ == "__main__":

	n_frames = 400
	time_step = 0.025

	trajectory = [ (time_step * .0 * n_frames, [ 0., 0., 0., 0., 0., 0. ]),
								 (time_step * .25 * n_frames, [ 0., 0., 1., 0., 0., pi ]),
								 (time_step * .50 * n_frames, [ 1., 1., 1., 0., 0., -pi ]),
								 (time_step * .75 * n_frames, [ -1., -1., 1., 0., 0., pi ]),
								 (time_step * 1. * n_frames, [ 0., 0., 1., 0., 0., 0. ]) ]

	dim = 1

	X0 = [ point[ 0 ] for point in trajectory ]
	Y0 = [ point[ 1 ][ dim ] for point in trajectory ]

	trajectory = generate_trajectory( trajectory, n_frames )

	X1 = [ point[ 0 ] for point in trajectory ]
	Y1 = [ point[ 1 ][ dim ] for point in trajectory ]

	plt.plot( X0, Y0, X1, Y1 )
	plt.show()
