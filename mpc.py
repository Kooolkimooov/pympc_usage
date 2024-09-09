from copy import deepcopy
from inspect import signature
from time import perf_counter

from numpy import eye, ndarray, zeros
from scipy.optimize import Bounds, LinearConstraint, minimize, NonlinearConstraint

from model import Model


class MPC:
	def __init__(
			self,
			model: Model,
			horizon: int,
			target_trajectory: ndarray,
			objective: callable = None,
			time_steps_per_actuation: int = 1,
			guess_from_last_solution: bool = True,
			tolerance: float = 1e-6,
			max_iter: int = 1000,
			bounds: tuple[ Bounds ] = None,
			constraints: tuple[ NonlinearConstraint ] | tuple[ LinearConstraint ] = None,
			pose_weight_matrix: ndarray = None,
			actuation_derivative_weight_matrix: ndarray = None,
			objective_weight: float = 0.,
			final_weight: float = 0.,
			record: bool = False,
			verbose: bool = False
			):

		assert objective is None or list( signature( objective ).parameters ) == [ 'trajectory',
																																							 'actuation' ]

		self.model = model
		self.horizon = horizon
		self.target_trajectory = target_trajectory
		self.objective = objective
		self.time_steps_per_actuation = time_steps_per_actuation
		self.guess_from_last_solution = guess_from_last_solution
		self.tolerance = tolerance
		self.max_iter = max_iter
		self.bounds = bounds
		self.constraints = constraints

		self.result_shape = (self.horizon // self.time_steps_per_actuation + (
				1 if self.horizon % self.time_steps_per_actuation != 0 else 0), 1,
												 self.model.actuation.shape[ 0 ])

		self.raw_result = None
		self.result = zeros( self.result_shape )

		self.pose_weight_matrix: ndarray = zeros(
				(self.horizon, self.model.state.shape[ 0 ] // 2, self.model.state.shape[ 0 ] // 2)
				)
		self.actuation_derivative_weight_matrix: ndarray = zeros(
				(self.result_shape[ 0 ], self.result_shape[ 2 ], self.result_shape[ 2 ])
				)

		if pose_weight_matrix is None:
			self.pose_weight_matrix[ : ] = eye( self.model.state.shape[ 0 ] // 2 )
		else:
			self.pose_weight_matrix[ : ] = pose_weight_matrix

		if actuation_derivative_weight_matrix is None:
			self.actuation_derivative_weight_matrix[ : ] = eye( self.result_shape[ 2 ] )
		else:
			self.actuation_derivative_weight_matrix[ : ] = actuation_derivative_weight_matrix

		self.objective_weight = objective_weight
		self.final_weight = final_weight

		self.record = record
		if self.record:
			self.predicted_trajectories = [ ]
			self.candidate_actuations = [ ]
			self.times = [ ]

		self.verbose = verbose

	def predict( self, actuation: ndarray, with_speed = False ) -> ndarray:
		p_state = deepcopy( self.model.state )
		vec_size = (self.model.state.shape[ 0 ]) if with_speed else (self.model.state.shape[ 0 ] // 2)
		predicted_trajectory = zeros( (self.horizon, 1, vec_size) )

		for i in range( self.horizon ):
			p_state += self.model.dynamics( p_state, actuation[ i, 0 ] ) * self.model.time_step
			predicted_trajectory[ i ] = p_state[ :vec_size ]

		return predicted_trajectory

	def apply_result( self ):
		self.model.actuation += self.result[ 0, 0 ]

	def compute_actuation( self ):

		if self.record:
			self.predicted_trajectories.clear()
			self.candidate_actuations.clear()
			ti = perf_counter()

		self.raw_result = minimize(
				fun = self.cost,
				x0 = self.result.flatten(),
				tol = self.tolerance,
				bounds = self.bounds,
				constraints = self.constraints,
				options = {
						'maxiter': self.max_iter, 'disp': self.verbose
						}
				)

		if self.record:
			self.times.append( perf_counter() - ti )

		if self.raw_result.success:
			self.result = self.raw_result.x.reshape( self.result_shape )

	def cost( self, actuations_derivative: ndarray ) -> float:
		actuations_derivative = actuations_derivative.reshape( self.result_shape )
		actuations = actuations_derivative.cumsum( axis = 0 ) + self.model.actuation

		actuations = actuations.repeat( self.time_steps_per_actuation, axis = 0 )
		actuations = actuations[ :self.horizon ]

		cost = 0.

		predicted_trajectory = self.predict( actuations )
		error = predicted_trajectory - self.target_trajectory
		cost += (error @ self.pose_weight_matrix @ error.transpose( (0, 2, 1) )).sum()
		cost += (
				actuations_derivative @ self.actuation_derivative_weight_matrix @
				actuations_derivative.transpose(
				(0, 2, 1)
				)).sum()
		cost += 0. if self.objective is None else self.objective_weight * self.objective(
				predicted_trajectory, actuations
				)

		cost /= self.horizon

		cost += self.final_weight * (error[ -1 ] @ self.pose_weight_matrix[ -1 ] @ error[ -1 ].T)

		if self.record:
			self.predicted_trajectories.append( predicted_trajectory )
			self.candidate_actuations.append( actuations )

		return cost
