from copy import deepcopy
from inspect import signature

from numpy import ndarray


def rungeKutta4(
		function: callable, time_step: float, current_state: ndarray, *args, **kwargs
		):
	'''
	Runge-Kutta 4th order method
	:param function: function to integrate, must have the following signature: f(x, *args, **kwargs)
	:param time_step: time step
	:param current_state: initial position
	:param args: additional arguments for f
	:param kwargs: additional keyword arguments for f
	'''

	# coefficients of the Butcher tableau
	a21 = .4
	a31 = .29697761
	a32 = .15875964
	a41 = .21810040
	a42 = -3.05096516
	a43 = 3.83286476
	b1 = .17476028
	b2 = -.55148066
	b3 = 1.20553560
	b4 = .17118478

	k1 = function( current_state, *args, **kwargs )
	k2 = function( current_state + a21 * k1 * time_step, *args, **kwargs )
	k3 = function( current_state + a31 * k1 * time_step + a32 * k2 * time_step, *args, **kwargs )
	k4 = function(
			current_state + a41 * k1 * time_step + a42 * k2 * time_step + a43 * k3 * time_step,
			*args,
			**kwargs
			)

	new_state = (
			current_state + b1 * k1 * time_step + b2 * k2 * time_step + b3 * k3 * time_step + b4 * k4 *
			time_step)

	return new_state


class Model:
	def __init__(
			self,
			dynamics: callable,
			time_step: float,
			initial_state: ndarray,
			initial_actuation: ndarray,
			kwargs = None,
			record: bool = False
			):

		if kwargs is None:
			kwargs = { }

		assert list( signature( dynamics ).parameters )[ :2 ] == [ 'state', 'actuation' ]

		self.model_dynamics = dynamics
		self.time_step = time_step

		self.kwargs = kwargs

		self.state = deepcopy( initial_state )
		self.actuation = deepcopy( initial_actuation )

		self.record = record
		if self.record:
			self.previous_states = [ deepcopy( self.state ) ]
			self.previous_actuations = [ deepcopy( self.actuation ) ]

	def dynamics( self, state: ndarray, actuation: ndarray ) -> ndarray:
		return self.model_dynamics( state, actuation, **self.kwargs )

	def step( self ):
		self.state = rungeKutta4( self.dynamics, self.time_step, self.state, self.actuation )

		if self.record:
			self.previous_states.append( deepcopy( self.state ) )
			self.previous_actuations.append( deepcopy( self.actuation ) )
