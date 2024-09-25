from copy import deepcopy
from time import perf_counter

from numpy import diff, eye, ndarray, zeros
from scipy.optimize import Bounds, LinearConstraint, minimize, NonlinearConstraint

from model import Model


class MPC:

	def __init__(
			self,
			model: Model,
			horizon: int,
			target_trajectory: ndarray,
			model_type: str = 'nonlinear',
			optimize_on: str = 'actuation_derivative',
			objective: callable = None,
			time_step: float = None,
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
		"""
		PREDICTION ASSUMES THAT THE MODELS STATE IS X = [POSE, POSE_DERIVATIVE]
		:param model: model of the system
		:param horizon: prediction horizon
		:param target_trajectory: target trajectory
		:param objective: objective function, must have the following signature: f(trajectory,
		actuation)
		:param time_step: time step of the mpc prediction if None get the model simulation time step
		:param time_steps_per_actuation: number of time steps per proposed actuation over the horizon
		:param guess_from_last_solution: whether to use the last solution as the initial guess
		:param tolerance: tolerance for the optimization algorithm
		:param max_iter: maximum number of iterations for the optimization algorithm
		:param bounds: bounds for the optimization variables
		:param constraints: constraints for the optimization variables
		:param pose_weight_matrix: weight matrix for the pose error; shape: (state_dim//2,
		state_dim//2)
		:param actuation_derivative_weight_matrix: weight matrix for the actuation derivative; shape:
		(actuation_dim, actuation_dim)
		:param objective_weight: weight for the objective function
		:param final_weight: weight for the final pose error
		:param record: whether to record the computation times, predicted trajectories and candidate
		actuations
		:param verbose: whether to print the optimization results
		"""

		assert model_type in self.__MODEL_TYPE
		assert optimize_on in self.__OPTIMIZE_ON

		self.predict = self.__MODEL_TYPE[ model_type ][ 'predict' ].__get__( self, MPC )
		self.get_actuation = self.__OPTIMIZE_ON[ optimize_on ][ 'get_actuation' ].__get__( self, MPC )
		self.get_result = self.__OPTIMIZE_ON[ optimize_on ][ 'get_result' ].__get__( self, MPC )

		self.model = model
		self.horizon = horizon
		self.target_trajectory = target_trajectory
		self.objective = objective

		if time_step is None:
			self.time_step = self.model.time_step
		else:
			self.time_step = time_step

		self.time_steps_per_actuation = time_steps_per_actuation
		self.guess_from_last_solution = guess_from_last_solution
		self.tolerance = tolerance
		self.max_iter = max_iter
		self.bounds = bounds
		self.constraints = constraints

		add_one = (1 if self.horizon % self.time_steps_per_actuation != 0 else 0)
		self.result_shape = (self.horizon // self.time_steps_per_actuation + add_one, 1, self.model.actuation.shape[ 0 ])

		self.raw_result = None
		self.result = zeros( self.model.actuation.shape )

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
			self.compute_times = [ ]

		self.verbose = verbose

	def compute_actuation( self ):
		"""
		computes the best actuation for the current state with a given horizon. records the computation
		time if record is True
		"""

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
			self.compute_times.append( perf_counter() - ti )

		if self.raw_result.success:
			self.get_result()

	def get_result( self ):
		raise NotImplementedError( 'predict method should have been implemented in __init__' )

	def apply_result( self ):
		"""
		applies the first actuation derivative of the current result to the model
		"""
		self.model.actuation = self.result

	def cost( self, candidate: ndarray ) -> float:
		"""
		cost function for the optimization. records the predicted trajectories and candidate actuation
		:param candidate: proposed actuation derivative over the horizon
		:return: cost
		"""

		actuation, actuation_derivatives = self.get_actuation( candidate )

		cost = 0.

		prediction = self.predict( actuation )
		predicted_trajectory = prediction[ :, :, :self.model.state.shape[ 0 ] // 2 ]

		error = predicted_trajectory - self.target_trajectory
		cost += (error @ self.pose_weight_matrix @ error.transpose( (0, 2, 1) )).sum()
		cost += (actuation_derivatives @ self.actuation_derivative_weight_matrix @ actuation_derivatives.transpose(
				(0, 2, 1)
				)).sum()
		cost += 0. if self.objective is None else self.objective_weight * self.objective(
				prediction, actuation
				)

		cost /= self.horizon

		cost += self.final_weight * (error[ -1 ] @ self.pose_weight_matrix[ -1 ] @ error[ -1 ].T)

		if self.record:
			self.predicted_trajectories.append( predicted_trajectory )
			self.candidate_actuations.append( actuation )

		return cost

	def predict( self, actuation: ndarray ) -> ndarray:
		"""
		predicts the trajectory given the proposed actuation over the horizon
		ASSUMES THAT THE MODELS STATE IS X = [POSE, POSE_DERIVATIVE]
		:param actuation: proposed actuation over the horizon
		:param with_speed: whether to return the predicted speed as well
		:return: predicted trajectory
		"""
		raise NotImplementedError( 'predict method should have been implemented in __init__' )

	def get_actuation( self, candidate: ndarray ) -> tuple[ ndarray, ndarray ]:
		raise NotImplementedError( 'predict method should have been implemented in __init__' )

	def _predict_non_linear( self, actuation: ndarray ) -> ndarray:

		p_state = deepcopy( self.model.state )
		predicted_trajectory = zeros( (self.horizon, 1, self.model.state.shape[ 0 ]) )

		for i in range( self.horizon ):
			p_state += self.model.dynamics( p_state, actuation[ i, 0 ] ) * self.time_step
			predicted_trajectory[ i ] = p_state

		return predicted_trajectory

	def _predict_linear( self, actuation: ndarray ) -> ndarray:
		raise NotImplementedError( 'predict method should have been implemented in __init__' )

	def _get_actuation_from_derivative( self, candidate: ndarray ) -> tuple[ ndarray, ndarray ]:
		actuation_derivatives = candidate.reshape( self.result_shape )
		actuation = actuation_derivatives.cumsum( axis = 0 ) * self.time_step + self.model.actuation
		actuation = actuation.repeat( self.time_steps_per_actuation, axis = 0 )
		actuation = actuation[ :self.horizon ]

		return actuation, actuation_derivatives

	def _get_actuation_from_actual( self, candidate: ndarray ) -> tuple[ ndarray, ndarray ]:
		actuation = candidate.reshape( self.result_shape )
		actuation_derivatives = diff( actuation, prepend = [ [ self.model.actuation ] ], axis = 0 ) / self.time_step
		actuation = actuation.repeat( self.time_steps_per_actuation, axis = 0 )
		actuation = actuation[ :self.horizon ]

		return actuation, actuation_derivatives

	def _get_result_from_derivative( self ):
		self.result = self.raw_result.x.reshape( self.result_shape )[ 0, 0 ] * self.model.time_step + self.model.actuation

	def _get_result_from_actual( self ):
		self.result = self.raw_result.x.reshape( self.result_shape )[ 0, 0 ]

	__MODEL_TYPE = {
			'nonlinear': { 'predict': _predict_non_linear }, 'linear': { 'predict': _predict_linear }
			}

	__OPTIMIZE_ON = {
			'actuation'           : { 'get_actuation': _get_actuation_from_actual, 'get_result': _get_result_from_actual },
			'actuation_derivative': {
					'get_actuation': _get_actuation_from_derivative, 'get_result': _get_result_from_derivative
					}
			}
