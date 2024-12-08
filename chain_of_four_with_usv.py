from numpy import dot, ndarray, r_, zeros
from numpy.linalg import norm

from bluerov import BluerovXYZPsi as Bluerov, USV
from catenary import Catenary
from mpc import MPC
from seafloor import Seafloor


class ChainOf4WithUSV:

	state_size = 4 * Bluerov.state_size
	actuation_size = 3 * Bluerov.actuation_size + USV.actuation_size

	def __init__(
			self,
			water_surface_depth: float,
			water_current: ndarray,
			seafloor: Seafloor,
			cables_length: float,
			cables_linear_mass: float,
			get_cable_parameter_method
			):

		self.br_0 = Bluerov( water_surface_depth, water_current )
		self.c_01 = Catenary( cables_length, cables_linear_mass, get_cable_parameter_method )
		self.br_1 = Bluerov( water_surface_depth, water_current )
		self.c_12 = Catenary( cables_length, cables_linear_mass, get_cable_parameter_method )
		self.br_2 = Bluerov( water_surface_depth, water_current )
		self.c_23 = Catenary( cables_length, cables_linear_mass, get_cable_parameter_method )
		self.br_3 = USV( water_surface_depth )

		self.sf = seafloor

		self.last_perturbation_01_0 = zeros( (Bluerov.pose_size,) )
		self.last_perturbation_01_1 = zeros( (Bluerov.pose_size,) )
		self.last_perturbation_12_1 = zeros( (Bluerov.pose_size,) )
		self.last_perturbation_12_2 = zeros( (Bluerov.pose_size,) )
		self.last_perturbation_23_2 = zeros( (Bluerov.pose_size,) )
		self.last_perturbation_23_3 = zeros( (Bluerov.pose_size,) )

		self.br_0_pose = slice( 0, 6 )
		self.br_0_position = slice( 0, 3 )
		self.br_0_xy = slice( 0, 2 )
		self.br_0_z = 2
		self.br_0_orientation = slice( 3, 6 )

		self.br_1_pose = slice( 6, 12 )
		self.br_1_position = slice( 6, 9 )
		self.br_1_xy = slice( 6, 8 )
		self.br_1_z = 8
		self.br_1_orientation = slice( 9, 12 )

		self.br_2_pose = slice( 12, 18 )
		self.br_2_position = slice( 12, 15 )
		self.br_2_xy = slice( 12, 14 )
		self.br_2_z = 14
		self.br_2_orientation = slice( 15, 18 )

		self.br_3_pose = slice( 18, 24 )
		self.br_3_position = slice( 18, 21 )
		self.br_3_xy = slice( 18, 20 )
		self.br_3_z = 20
		self.br_3_orientation = slice( 21, 24 )

		self.br_0_speed = slice( 24, 30 )
		self.br_0_linear_speed = slice( 24, 27 )
		self.br_0_angular_speed = slice( 27, 30 )

		self.br_1_speed = slice( 30, 36 )
		self.br_1_linear_speed = slice( 30, 33 )
		self.br_1_angular_speed = slice( 33, 36 )

		self.br_2_speed = slice( 36, 42 )
		self.br_2_linear_speed = slice( 36, 39 )
		self.br_2_angular_speed = slice( 39, 42 )

		self.br_3_speed = slice( 42, 48 )
		self.br_3_linear_speed = slice( 42, 45 )
		self.br_3_angular_speed = slice( 45, 48 )

		self.br_0_actuation_start = 0
		self.br_0_actuation = slice( self.br_0_actuation_start, self.br_0_actuation_start + Bluerov.actuation_size )
		self.br_0_linear_actuation = slice(
				self.br_0_actuation_start, self.br_0_actuation_start + Bluerov.linear_actuation_size
				)
		self.br_0_angular_actuation = slice(
				self.br_0_actuation_start + Bluerov.linear_actuation_size, self.br_0_actuation_start + Bluerov.actuation_size
				)

		self.br_1_actuation_start = Bluerov.actuation_size
		self.br_1_actuation = slice( self.br_1_actuation_start, self.br_1_actuation_start + Bluerov.actuation_size )
		self.br_1_linear_actuation = slice(
				self.br_1_actuation_start, self.br_1_actuation_start + Bluerov.linear_actuation_size
				)
		self.br_1_angular_actuation = slice(
				self.br_1_actuation_start + Bluerov.linear_actuation_size, self.br_1_actuation_start + Bluerov.actuation_size
				)

		self.br_2_actuation_start = 2 * Bluerov.actuation_size
		self.br_2_actuation = slice( self.br_2_actuation_start, self.br_2_actuation_start + Bluerov.actuation_size )
		self.br_2_linear_actuation = slice(
				self.br_2_actuation_start, self.br_2_actuation_start + Bluerov.linear_actuation_size
				)
		self.br_2_angular_actuation = slice(
				self.br_2_actuation_start + Bluerov.linear_actuation_size, self.br_2_actuation_start + Bluerov.actuation_size
				)

		self.br_3_actuation_start = 3 * Bluerov.actuation_size
		self.br_3_actuation = slice( self.br_3_actuation_start, self.br_3_actuation_start + USV.actuation_size )
		self.br_3_linear_actuation = slice(
				self.br_3_actuation_start, self.br_3_actuation_start + USV.linear_actuation_size
				)
		self.br_3_angular_actuation = slice(
				self.br_3_actuation_start + USV.linear_actuation_size, self.br_3_actuation_start + USV.actuation_size
				)

		self.br_0_state = r_[ self.br_0_pose, self.br_0_speed ]
		self.br_1_state = r_[ self.br_1_pose, self.br_1_speed ]
		self.br_2_state = r_[ self.br_2_pose, self.br_2_speed ]
		self.br_3_state = r_[ self.br_3_pose, self.br_3_speed ]

	def __call__( self, state: ndarray, actuation: ndarray ) -> ndarray:
		"""
		evaluates the dynamics of each robot of the chain
		:param state: current state of the system
		:param actuation: current actuation of the system
		:return: state derivative of the system
		"""
		state_derivative = zeros( state.shape )

		perturbation_01_0, perturbation_01_1 = self.c_01.get_perturbations(
				state[ self.br_0_position ], state[ self.br_1_position ]
				)
		perturbation_12_1, perturbation_12_2 = self.c_12.get_perturbations(
				state[ self.br_1_position ], state[ self.br_2_position ]
				)
		perturbation_23_2, perturbation_23_3 = self.c_23.get_perturbations(
				state[ self.br_2_position ], state[ self.br_3_position ]
				)

		# if the cable is taunt the perturbation is None
		# here we should consider any pair with a taunt cable as a single body
		if perturbation_01_0 is not None:
			self.last_perturbation_01_0[ :3 ] = perturbation_01_0
			self.last_perturbation_01_1[ :3 ] = perturbation_01_1
		else:
			perturbation_01_0, perturbation_01_1 = self.get_taunt_cable_perturbations( state, actuation, 0 )

		if perturbation_12_1 is not None:
			self.last_perturbation_12_1[ :3 ] = perturbation_12_1
			self.last_perturbation_12_2[ :3 ] = perturbation_12_2
		else:
			perturbation_12_1, perturbation_12_2 = self.get_taunt_cable_perturbations( state, actuation, 1 )

		if perturbation_23_2 is not None:
			self.last_perturbation_23_2[ :3 ] = perturbation_23_2
			self.last_perturbation_23_3[ :3 ] = perturbation_23_3
		else:
			perturbation_23_2, perturbation_23_3 = self.get_taunt_cable_perturbations( state, actuation, 2 )

		perturbation_01_0.resize( (Bluerov.pose_size,), refcheck = False )
		perturbation_01_1.resize( (Bluerov.pose_size,), refcheck = False )
		perturbation_12_1.resize( (Bluerov.pose_size,), refcheck = False )
		perturbation_12_2.resize( (Bluerov.pose_size,), refcheck = False )
		perturbation_23_2.resize( (Bluerov.pose_size,), refcheck = False )
		perturbation_23_3.resize( (Bluerov.pose_size,), refcheck = False )

		# perturbation is in world frame, should be applied robot frame instead
		br_0_transformation_matrix = self.br_0.build_transformation_matrix( *state[ self.br_0_orientation ] )
		br_1_transformation_matrix = self.br_1.build_transformation_matrix( *state[ self.br_1_orientation ] )
		br_2_transformation_matrix = self.br_2.build_transformation_matrix( *state[ self.br_2_orientation ] )
		br_3_transformation_matrix = self.br_3.build_transformation_matrix( *state[ self.br_3_orientation ] )

		perturbation_01_0 = br_0_transformation_matrix.T @ perturbation_01_0
		perturbation_01_1 = br_1_transformation_matrix.T @ perturbation_01_1
		perturbation_12_1 = br_1_transformation_matrix.T @ perturbation_12_1
		perturbation_12_2 = br_2_transformation_matrix.T @ perturbation_12_2
		perturbation_23_2 = br_2_transformation_matrix.T @ perturbation_23_2
		perturbation_23_3 = br_3_transformation_matrix.T @ perturbation_23_3

		state_derivative[ self.br_0_state ] = self.br_0(
				state[ self.br_0_state ], actuation[ self.br_0_actuation ], perturbation_01_0
				)
		state_derivative[ self.br_1_state ] = self.br_1(
				state[ self.br_1_state ], actuation[ self.br_1_actuation ], perturbation_01_1 + perturbation_12_1
				)
		state_derivative[ self.br_2_state ] = self.br_2(
				state[ self.br_2_state ], actuation[ self.br_2_actuation ], perturbation_12_2 + perturbation_23_2
				)
		state_derivative[ self.br_3_state ] = self.br_3(
				state[ self.br_3_state ], actuation[ self.br_3_actuation ], perturbation_23_3
				)

		return state_derivative

	def get_taunt_cable_perturbations(
			self, state: ndarray, actuation: ndarray, pair: int
			) -> tuple[ ndarray, ndarray ]:
		match pair:
			case 0:
				br_0 = self.br_0
				br_1 = self.br_1
				br_0_state = self.br_0_state
				br_0_position = self.br_0_position
				br_0_orientation = self.br_0_orientation
				br_0_actuation = self.br_0_actuation
				br_0_perturbation = self.last_perturbation_01_0
				br_1_state = self.br_1_state
				br_1_position = self.br_1_position
				br_1_orientation = self.br_1_orientation
				br_1_actuation = self.br_1_actuation
				br_1_perturbation = self.last_perturbation_01_1
			case 1:
				br_0 = self.br_1
				br_1 = self.br_2
				br_0_state = self.br_1_state
				br_0_position = self.br_1_position
				br_0_orientation = self.br_1_orientation
				br_0_actuation = self.br_1_actuation
				br_0_perturbation = self.last_perturbation_12_1
				br_1_state = self.br_2_state
				br_1_position = self.br_2_position
				br_1_orientation = self.br_2_orientation
				br_1_actuation = self.br_2_actuation
				br_1_perturbation = self.last_perturbation_12_2
			case 2:
				br_0 = self.br_2
				br_1 = self.br_3
				br_0_state = self.br_2_state
				br_0_position = self.br_2_position
				br_0_orientation = self.br_2_orientation
				br_0_actuation = self.br_2_actuation
				br_0_perturbation = self.last_perturbation_23_2
				br_1_state = self.br_3_state
				br_1_position = self.br_3_position
				br_1_orientation = self.br_3_orientation
				br_1_actuation = self.br_3_actuation
				br_1_perturbation = self.last_perturbation_23_3
			case _:
				raise RuntimeError( f'unknown {pair=}' )

		# from br_0 to br_1
		direction = state[ br_1_position ] - state[ br_0_position ]
		direction /= norm( direction )

		null = zeros( (br_0.pose_size,) )

		br_0_transformation_matrix = br_0.build_transformation_matrix( *state[ br_0_orientation ] )
		br_1_transformation_matrix = br_1.build_transformation_matrix( *state[ br_1_orientation ] )

		# in robot frame
		br_0_acceleration = br_0( state[ br_0_state ], actuation[ br_0_actuation ], null )[ 6: ]
		br_0_forces = (br_0.inertial_matrix[ :3, :3 ] @ br_0_acceleration[ :3 ])

		br_1_acceleration = br_1( state[ br_1_state ], actuation[ br_1_actuation ], null )[ 6: ]
		br_1_forces = (br_1.inertial_matrix[ :3, :3 ] @ br_1_acceleration[ :3 ])

		all_forces = dot( br_0_transformation_matrix[ :3, :3 ] @ br_0_forces, -direction )
		all_forces += dot( br_1_transformation_matrix[ :3, :3 ] @ br_1_forces, direction )
		all_forces += dot( br_0_perturbation[ :3 ], direction )
		all_forces += dot( br_1_perturbation[ :3 ], -direction )

		perturbation = direction * all_forces

		# in world frame
		return perturbation, -perturbation


def chain_of_4_constraints( self: MPC, candidate ):

	chain: ChainOf4WithUSV = self.model.dynamics

	actuation, _ = self.get_actuation( candidate )

	prediction = self.predict( actuation )
	prediction = prediction[ :, 0 ]

	# 3 constraints on cables (distance of lowest point to seafloor)
	# 4 constraints on robots (distance of lowest point to seafloor)
	# 6 on inter robot_distance (3 horizontal, 2 3d)
	n_constraints = 3 + 4 + 6
	constraints = zeros( (self.horizon, n_constraints) )

	for i, state in enumerate( prediction ):
		lp01 = chain.c_01.get_lowest_point( state[ chain.br_0_position ], state[ chain.br_1_position ] )
		lp12 = chain.c_12.get_lowest_point( state[ chain.br_1_position ], state[ chain.br_2_position ] )
		lp23 = chain.c_23.get_lowest_point( state[ chain.br_2_position ], state[ chain.br_3_position ] )

		# cables distance from seafloor [0, 3[
		constraints[ i, 0 ] = chain.sf.get_distance_to_seafloor( lp01 )
		constraints[ i, 1 ] = chain.sf.get_distance_to_seafloor( lp12 )
		constraints[ i, 2 ] = chain.sf.get_distance_to_seafloor( lp23 )

		lp0 = zeros( (3,) )
		lp0[ :2 ] = state[ chain.br_0_xy ]
		lp0[ 2 ] = max( lp01[ 2 ], state[ chain.br_0_z ] )

		lp1 = zeros( (3,) )
		lp1[ :2 ] = state[ chain.br_1_xy ]
		lp1[ 2 ] = max( lp01[ 2 ], lp12[ 2 ], state[ chain.br_1_z ] )

		lp2 = zeros( (3,) )
		lp2[ :2 ] = state[ chain.br_2_xy ]
		lp2[ 2 ] = max( lp12[ 2 ], lp23[ 2 ], state[ chain.br_2_z ] )

		lp3 = zeros( (3,) )
		lp3[ :2 ] = state[ chain.br_3_xy ]
		lp3[ 2 ] = max( lp23[ 2 ], state[ chain.br_3_z ] )

		# robot distance from seafloor, taking into accout the cables [3, 7[
		constraints[ i, 3 ] = chain.sf.get_distance_to_seafloor( lp0 )
		constraints[ i, 4 ] = chain.sf.get_distance_to_seafloor( lp1 )
		constraints[ i, 5 ] = chain.sf.get_distance_to_seafloor( lp2 )
		constraints[ i, 6 ] = chain.sf.get_distance_to_seafloor( lp3 )

	# horizontal distance between consecutive robots [7, 10[
	constraints[ :, 7 ] = norm(
			prediction[ :, chain.br_1_xy ] - prediction[ :, chain.br_0_xy ], axis = 1
			)
	constraints[ :, 8 ] = norm(
			prediction[ :, chain.br_2_xy ] - prediction[ :, chain.br_1_xy ], axis = 1
			)
	constraints[ :, 9 ] = norm(
			prediction[ :, chain.br_3_xy ] - prediction[ :, chain.br_2_xy ], axis = 1
			)

	# distance between consecutive robots [10, 13[
	constraints[ :, 10 ] = norm(
			prediction[ :, chain.br_1_position ] - prediction[ :, chain.br_0_position ], axis = 1
			)
	constraints[ :, 11 ] = norm(
			prediction[ :, chain.br_2_position ] - prediction[ :, chain.br_1_position ], axis = 1
			)
	constraints[ :, 12 ] = norm(
			prediction[ :, chain.br_3_position ] - prediction[ :, chain.br_2_position ], axis = 1
			)

	return constraints.flatten()


def chain_of_4_objective( self: MPC, prediction: ndarray, actuation: ndarray ):

	chain: ChainOf4WithUSV = self.model.dynamics
	desired_distance = chain.c_01.length / 2

	objective = 0.

	# objective += pow( norm( prediction[ :, 0, chain.br_0_speed ], axis = 1 ).sum(), 2 )
	objective += pow( norm( prediction[ :, 0, chain.br_1_speed ], axis = 1 ).sum(), 2 )
	objective += pow( norm( prediction[ :, 0, chain.br_2_speed ], axis = 1 ).sum(), 2 )
	objective += pow( norm( prediction[ :, 0, chain.br_3_speed ], axis = 1 ).sum(), 2 )

	objective += abs(
			norm(
					prediction[ :, 0, chain.br_0_position ] - prediction[ :, 0, chain.br_1_position ], axis = 1
					) - desired_distance
			).sum()
	objective += abs(
			norm(
					prediction[ :, 0, chain.br_1_position ] - prediction[ :, 0, chain.br_2_position ], axis = 1
					) - desired_distance
			).sum()
	objective += abs(
			norm(
					prediction[ :, 0, chain.br_2_position ] - prediction[ :, 0, chain.br_3_position ], axis = 1
					) - desired_distance
			).sum()

	return objective
