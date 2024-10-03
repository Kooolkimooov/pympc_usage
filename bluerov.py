from numpy import array, concat, cos, cross, diag, exp, eye, ndarray, sin, tan, zeros
from numpy.linalg import inv


class Bluerov:
	"""
	Bluerov model, based on the BlueROV model from Blue Robotics
	paramters of the model are based on the BlueROV2 Heavy configuration
	and are stored in the class as class variables
	"""

	state_size = 12
	actuation_size = 6

	def __init__( self, water_surface_z: float = 0., water_current: ndarray = None ):

		self.mass = 11.5
		self.center_of_mass = array( [ 0.0, 0.0, 0.0 ] )
		self.weight = self.mass * array( [ 0., 0., 9.80665 ] )

		self.buoyancy_norm = 120.
		self.center_of_volume = array( [ 0.0, 0.0, -0.02 ] )
		self.water_surface_z = water_surface_z

		# water speed should be on [3:6]
		water_current = zeros( (6,) ) if water_current is None else water_current
		if water_current.shape == (3,):
			water_current = concat( (water_current, array( [ 0., 0., 0. ] )) )

		assert water_current.shape == (6,)

		self.water_current = water_current

		self.inertial_coefficients = [ .16, .16, .16, 0., 0., 0. ]
		self.hydrodynamic_coefficients = [ 4.03, 6.22, 5.18, 0.07, 0.07, 0.07 ]
		self.added_mass_coefficients = [ 5.5, 12.7, 14.57, .12, .12, .12 ]

		self.inertial_matrix = self.build_inertial_matrix(
				self.mass,
				self.center_of_mass,
				self.inertial_coefficients
				) + diag( self.added_mass_coefficients )

		self.inverse_inertial_matrix = inv( self.inertial_matrix )

		self.hydrodynamic_matrix = diag( self.hydrodynamic_coefficients )

	def __call__( self, state: ndarray, actuation: ndarray, perturbation: ndarray ) -> ndarray:
		"""
		evalutes the dynamics of the Bluerov model
		:param state: current state of the system
		:param actuation: current actuation of the system
		:return: state derivative of the system
		"""

		transform_matrix = self.build_transformation_matrix( *state[ 3:6 ] )

		# sigmoid the buoyancy to smooth out the discontinuity
		# buoyancy = self.buoyancy_norm * array( [ 0., 0., -1. ] ) / (
		# 		1 + exp( 10. * (self.water_surface_z - state[ 2 ]) - 2. )
		# )

		buoyancy = self.buoyancy_norm * array( [ 0., 0., 1. ] ) * (-.5 - .5 / (
				1 + exp( 10. * (self.water_surface_z - state[ 2 ]) - 2. )
		))

		hydrostatic_forces = zeros( 6 )
		hydrostatic_forces[ :3 ] = transform_matrix[ :3, :3 ].T @ (self.weight + buoyancy)
		hydrostatic_forces[ 3: ] = cross(
				self.center_of_mass, transform_matrix[ :3, :3 ].T @ self.weight
				) + cross(
				self.center_of_volume, transform_matrix[ :3, :3 ].T @ buoyancy
				)

		xdot = zeros( state.shape )
		xdot[ :6 ] = transform_matrix @ state[ 6: ]
		xdot[ 6: ] = self.inverse_inertial_matrix @ (
				self.hydrodynamic_matrix @ (state[ 6: ] - self.water_current) + hydrostatic_forces + actuation + perturbation)

		return xdot

	@staticmethod
	def build_transformation_matrix( phi: float, theta: float, psi: float ) -> ndarray:
		cPhi, sPhi = cos( phi ), sin( phi )
		cTheta, sTheta, tTheta = cos( theta ), sin( theta ), tan( theta )
		cPsi, sPsi = cos( psi ), sin( psi )

		matrix = zeros( (6, 6) )
		matrix[ 0, :3 ] = array(
				[ cPsi * cTheta, -sPsi * cPhi + cPsi * sTheta * sPhi, sPsi * sPhi + cPsi * sTheta * cPhi ]
				)
		matrix[ 1, :3 ] = array(
				[ sPsi * cTheta, cPsi * cPhi + sPsi * sTheta * sPhi, -cPsi * sPhi + sPsi * sTheta * cPhi ]
				)
		matrix[ 2, :3 ] = array( [ -sTheta, cTheta * sPhi, cTheta * cPhi ] )
		matrix[ 3, 3: ] = array( [ 1, sPhi * tTheta, cPhi * tTheta ] )
		matrix[ 4, 3: ] = array( [ 0, cPhi, -sPhi ] )
		matrix[ 5, 3: ] = array( [ 0, sPhi / cTheta, cPhi / cTheta ] )
		return matrix

	@staticmethod
	def build_inertial_matrix(
			mass: float, center_of_mass: ndarray, inertial_coefficients: list[ float ]
			) -> ndarray:
		inertial_matrix = eye( 6 )
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

		return inertial_matrix


class BluerovNoAngularActuation( Bluerov ):

	actuation_size = 3

	def __init__( self ):
		super().__init__()

	def __call__( self, state: ndarray, actuation: ndarray, perturbation ) -> ndarray:
		six_dof_actuation = zeros( (6,) )
		six_dof_actuation[ :3 ] = actuation

		return Bluerov.__call__( self, state, six_dof_actuation, perturbation )


class USV( Bluerov ):
	actuation_size = 2

	def __init__( self, water_surface_z: float = 0. ):
		super().__init__( water_surface_z )

	def __call__( self, state, actuation, perturbation ):
		six_dof_actuation = zeros( (6,) )
		# linear actuation on x
		six_dof_actuation[ 0 ] = actuation[ 0 ]
		# angular actuation around z
		six_dof_actuation[ 5 ] = actuation[ 1 ]

		return Bluerov.__call__( self, state, six_dof_actuation, perturbation )
