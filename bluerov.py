from numpy import array, cross, diag, ndarray, zeros
from numpy.linalg import inv

from utils import (
	build_inertial_matrix, build_transformation_matrix,
	)


class Bluerov:
	"""
	Bluerov model, based on the BlueROV model from Blue Robotics
	paramters of the model are based on the BlueROV2 Heavy configuration
	and are stored in the class as class variables
	"""

	state_size = 12
	actuation_size = 6

	mass = 11.5
	center_of_mass = array( [ 0.0, 0.0, 0.0 ] )
	weight = array( [ 0., 0., mass * 9.80665 ] )

	center_of_volume = array( [ 0.0, 0.0, -0.02 ] )
	buoyancy = array( [ 0, 0, -120. ] )

	inverse_inertial_matrix = inv(
			build_inertial_matrix( mass, center_of_mass, [ .16, .16, .16, 0., 0., 0. ] )
			)

	hydrodynamic_matrix = diag( [ 4.03, 6.22, 5.18, 0.07, 0.07, 0.07 ] )

	def __call__( self, state: ndarray, actuation: ndarray, perturbation: ndarray ) -> ndarray:
		"""
		evalutes the dynamics of the Bluerov model
		:param state: current state of the system
		:param actuation: current actuation of the system
		:return: state derivative of the system
		"""

		transform_matrix = build_transformation_matrix( *state[ 3:6 ] )

		hydrostatic_forces = zeros( 6 )
		hydrostatic_forces[ :3 ] = transform_matrix[ :3, :3 ].T @ (self.weight + self.buoyancy)
		hydrostatic_forces[ 3: ] = cross(
				self.center_of_mass, transform_matrix[ :3, :3 ].T @ self.weight
				) + cross(
				self.center_of_volume, transform_matrix[ :3, :3 ].T @ self.buoyancy
				)

		xdot = zeros( state.shape )
		xdot[ :6 ] = transform_matrix @ state[ 6: ]
		xdot[ 6: ] = self.inverse_inertial_matrix @ (
				self.hydrodynamic_matrix @ state[ 6: ] + hydrostatic_forces + actuation + perturbation)

		return xdot


class BluerovNoAngularActuation( Bluerov ):

	actuation_size = 3

	def __call__( self, state: ndarray, actuation: ndarray, perturbation ) -> ndarray:
		six_dof_actuation = zeros( (6,) )
		six_dof_actuation[ :3 ] = actuation

		return Bluerov.__call__( self, state, six_dof_actuation, perturbation )
