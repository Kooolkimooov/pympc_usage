from warnings import simplefilter

from numpy import dot, ndarray, r_, zeros
from numpy.linalg import norm

from bluerov import BluerovXYZPsi as Bluerov
from catenary import Catenary
from seafloor import Seafloor

simplefilter( 'ignore', RuntimeWarning )


class ChainOf3WithFixedPoint:

	state_size = 3 * Bluerov.state_size
	actuation_size = 2 * Bluerov.actuation_size

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

		self.sf = seafloor

		self.last_perturbation_01_0 = zeros( (Bluerov.pose_size,) )
		self.last_perturbation_01_1 = zeros( (Bluerov.pose_size,) )
		self.last_perturbation_12_1 = zeros( (Bluerov.pose_size,) )

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

		self.br_0_speed = slice( 12, 18 )
		self.br_0_linear_speed = slice( 12, 15 )
		self.br_0_angular_speed = slice( 15, 18 )

		self.br_1_speed = slice( 18, 24 )
		self.br_1_linear_speed = slice( 18, 21 )
		self.br_1_angular_speed = slice( 21, 24 )

		self.br_0_actuation_start = 0
		self.br_0_actuation = slice( 0, 6 )
		self.br_0_linear_actuation = slice( 0, 3 )
		self.br_0_angular_actuation = slice( 3, 6 )

		self.br_1_actuation_start = 6
		self.br_1_actuation = slice( 6, 12 )
		self.br_1_linear_actuation = slice( 6, 9 )
		self.br_1_angular_actuation = slice( 9, 12 )

		self.br_0_state = r_[ self.br_0_pose, self.br_0_speed ]
		self.br_1_state = r_[ self.br_1_pose, self.br_1_speed ]

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
		perturbation_12_1, _ = self.c_12.get_perturbations(
				state[ self.br_0_position ], state[ self.br_1_position ]
				)

		# if the cable is taunt the perturbation is None
		# here we should consider any pair with a taunt cable as a single body
		if perturbation_01_0 is not None:
			self.last_perturbation_01_0[ :3 ] = perturbation_01_0
			self.last_perturbation_01_1[ :3 ] = perturbation_01_1
		else:
			perturbation_01_0, perturbation_01_1 = self.get_taunt_cable_perturbations( state, actuation, 0 )

		perturbation_01_0.resize( (Bluerov.actuation_size,), refcheck = False )
		perturbation_01_1.resize( (Bluerov.actuation_size,), refcheck = False )

		# perturbation is in world frame, should be applied robot frame instead
		br_0_transformation_matrix = self.br_0.build_transformation_matrix( *state[ self.br_0_orientation ] )
		br_1_transformation_matrix = self.br_1.build_transformation_matrix( *state[ self.br_1_orientation ] )
		perturbation_01_0 = br_0_transformation_matrix.T @ perturbation_01_0
		perturbation_01_1 = br_1_transformation_matrix.T @ perturbation_01_1

		state_derivative[ self.br_0_state ] = self.br_0(
				state[ self.br_0_state ], actuation[ self.br_0_actuation ], perturbation_01_0
				)
		state_derivative[ self.br_1_state ] = self.br_1(
				state[ self.br_1_state ], actuation[ self.br_1_actuation ], perturbation_01_1
				)

		return state_derivative

	def get_taunt_cable_perturbations(
			self, state: ndarray, actuation: ndarray, pair: int
			) -> tuple[ ndarray, ndarray ]:
		match pair:
			case 0:
				br_0 = self.br_0
				c_01 = self.c_01
				br_1 = self.br_1
				br_0_state = state[ self.br_0_state ]
				br_0_position = state[ self.br_0_position ]
				br_0_orientation = state[ self.br_0_orientation ]
				br_0_actuation = actuation[ self.br_0_actuation ]
				br_0_perturbation = self.last_perturbation_0
				br_1_state = state[ self.br_1_state ]
				br_1_position = state[ self.br_1_position ]
				br_1_orientation = state[ self.br_1_orientation ]
				br_1_actuation = actuation[ self.br_1_actuation ]
				br_1_perturbation = self.last_perturbation_1
			case 1:
				br_0 = self.br_1
				c_01 = self.c_12
				br_1 = self.br_2
				br_0_state = state[self.br_0_state]
				br_0_position = state[self.br_0_position]
				br_0_orientation = state[self.br_0_orientation]
				br_0_actuation = [self.br_0_actuation]
				br_0_perturbation = self.last_perturbation_0
				br_1_state = self.br_1_state
				br_1_position = self.br_1_position
				br_1_orientation = self.br_1_orientation
				br_1_actuation = self.br_1_actuation
				br_1_perturbation = self.last_perturbation_1
			case _:
				raise RuntimeError( f'unknown {pair=}' )

		# from br_0 to br_1
		direction = br_1_position - br_0_position
		direction /= norm( direction )

		null = zeros( (br_0.pose_size,) )

		br_0_transformation_matrix = br_0.build_transformation_matrix( *br_0_orientation )
		br_1_transformation_matrix = br_1.build_transformation_matrix( *br_1_orientation )

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
