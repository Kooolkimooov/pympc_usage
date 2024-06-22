from copy import deepcopy
from inspect import signature

import numpy as np
from numpy import ndarray, zeros
from numpy.linalg import norm
from scipy.optimize import Bounds, LinearConstraint, minimize, NonlinearConstraint


def model_predictive_control_cost_function(
		candidate_actuations: ndarray,
		candidate_shape: tuple,
		model: callable,
		initial_state: ndarray,
		initial_actuation: ndarray,
		model_kwargs: dict,
		target: ndarray,
		objective_function: callable,
		optimization_horizon: int,
		prediction_horizon: int,
		time_step: float,
		time_steps_per_actuation: int,
		error_weight_matrix: ndarray,
		objective_weight: float,
		final_cost_weight: float,
		state_record: list,
		actuation_record: list,
		objective_record: list,
		verbose: bool
		) -> float:

	if norm( error_weight_matrix ) == 0 and final_cost_weight == 0 and objective_function is None:
		raise ValueError( "Cannot compute model_predictive_control_cost_function" )

	candidate_actuations = candidate_actuations.reshape( candidate_shape )

	actuation = deepcopy( initial_actuation )
	state = deepcopy( initial_state )
	cost = 0.
	horizon = optimization_horizon + prediction_horizon

	if state_record is not None:
		states = zeros( (horizon, initial_state.shape[ 0 ]) )
	if actuation_record is not None:
		actuations = zeros( (horizon, initial_actuation.shape[ 0 ]) )
	if objective_function is not None:
		objectives = zeros( (horizon, 1) )

	time_steps_count = time_steps_per_actuation

	for i in range( optimization_horizon ):

		if time_steps_count == time_steps_per_actuation:
			actuation += candidate_actuations[ i // time_steps_per_actuation ]
			time_steps_count = 0

		state += model( state, actuation, **model_kwargs ) * time_step

		error = state[ :len( state ) // 2 ] - target
		cost += error @ error_weight_matrix @ error.T

		time_steps_count += 1

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

		state += model( state, actuation, **model_kwargs ) * time_step

		error = state[ :len( state ) // 2 ] - target
		cost += error @ error_weight_matrix @ error.T

		if state_record is not None:
			states[ optimization_horizon + i ] = state

		if actuation_record is not None:
			actuations[ optimization_horizon + i ] = actuation

		if objective_function is not None:
			objective = objective_function( state, actuation, **model_kwargs )
			cost += objective_weight * objective
			if objective_record is not None:
				objectives[ optimization_horizon + i ] = objective

	error = state[ :len( state ) // 2 ] - target
	cost += final_cost_weight * pow( norm( error @ error_weight_matrix @ error.T ), 2 )
	if objective_function is not None:
		cost += final_cost_weight * objective_weight * objective_function(
			state,
			actuation,
			**model_kwargs
			)

	cost /= horizon

	if state_record is not None:
		state_record.append( states )
	if actuation_record is not None:
		actuation_record.append( actuations )
	if objective_record is not None:
		objective_record.append( objectives )

	if verbose:
		print( f'{horizon=}; {cost=}' )

	return cost


# model predictive control returns the optimal actuation
def optimize(
		cost_function: callable,
		cost_kwargs: dict,
		initial_guess: ndarray,
		tolerance: float = 1e-6,
		max_iter: int = 100,
		bounds: Bounds = None,
		constraints: tuple[ NonlinearConstraint ] | tuple[ LinearConstraint ] = None, ) -> ndarray:

	cost_args = ()
	for parameter in signature( cost_function ).parameters:
		if parameter != 'candidate_actuations':
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

	if cost_kwargs[ 'verbose' ]:
		print( result )

	return result.x.reshape( cost_kwargs[ 'candidate_shape' ] )


def serialize_others( obj: any ) -> str:
	if callable( obj ):
		return obj.__name__
	if isinstance( obj, ndarray ):
		return obj.tolist()
