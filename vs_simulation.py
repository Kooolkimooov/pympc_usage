from json import dump, load
from os.path import join, split
from time import perf_counter, time
from warnings import simplefilter

# noinspection PyUnresolvedReferences
from numpy import array, concatenate, cos, diff, eye, inf, max, min, pi, set_printoptions, sqrt, zeros, any, isnan
from numpy.linalg import norm

from pympc.controllers.vs import VS
from pympc.controllers.pid import PID
from pympc.models.dynamics.chain_of_four_with_usv import ChainOf4WithUSV
from pympc.models.model import Model
from pympc.models.seafloor import SeafloorFromFunction, seafloor_function_1
from pympc.utils import Logger, check, generate_trajectory, get_computer_info, print_dict, serialize_others

from pympc.models.dynamics.chain_of_four_with_usv import chain_of_4_constraints_pp

if __name__ == "__main__":
    simplefilter( 'ignore', RuntimeWarning )
    set_printoptions( precision=2, linewidth=10000, suppress=True )

    ti = perf_counter()

    record = True
    seafloor = SeafloorFromFunction( seafloor_function_1 )

    dynamics = ChainOf4WithUSV(
            water_surface_depth=0.,
            water_current=None, #array( [ sqrt( 2.0 ), 0., 0. ] ),
            seafloor=seafloor,
            cables_length=3.0,
            cables_linear_mass=0.01,
            get_cable_parameter_method='precompute',
            reference_frame='ENU'
    )

    time_step = 0.1

    initial_actuation = zeros( (dynamics.actuation_size,) )

    initial_state = zeros( (dynamics.state_size,) )
    initial_state[ dynamics.br_0_position[ 0 ] ] = 2.
    initial_state[ dynamics.br_0_position[ 2 ] ] = -1.
    initial_state[ dynamics.br_1_position[ 0 ] ] = 2.5
    initial_state[ dynamics.br_1_position[ 2 ] ] = -1.
    initial_state[ dynamics.br_2_position[ 0 ] ] = 3.
    initial_state[ dynamics.br_2_position[ 2 ] ] = -1.
    initial_state[ dynamics.br_3_position[ 0 ] ] = 3.5
    initial_state[ dynamics.br_3_orientation[ 2 ] ] = pi / 2 - 1e-2

    model = Model(
            dynamics=dynamics,
            time_step=time_step,
            initial_state=initial_state,
            initial_actuation=initial_actuation,
            record=True
    )

    n_frames = 500

    key_frames = [
            (0., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 18),
            (.5, [ -5., 0., 0., 0., 0., 0. ] + [ 0. ] * 18),
            (1., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 18),
            (2., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 18)
    ]
    trajectory = generate_trajectory( key_frames, 2 * n_frames )
    trajectory[ :, 0, dynamics.br_0_position[ 2 ] ] = -1.5 * cos(
            1.25 * (trajectory[ :, 0, dynamics.br_0_position[ 0 ] ] - 2) + pi
    ) - 2.5

    max_required_speed = max( norm( diff( trajectory[ :, 0, :3 ], axis=0 ), axis=1 ) ) / time_step

    if 'y' != input( f'{max_required_speed=}, continue ? (y/n) ' ):
        exit()

    proportional = eye( dynamics.state_size // 2 )[ dynamics.six_dof_actuation_mask, : ]
    integral = eye( dynamics.state_size // 2 )[ dynamics.six_dof_actuation_mask, : ]
    derivative = eye( dynamics.state_size // 2 )[ dynamics.six_dof_actuation_mask, : ]
    offset = zeros( (dynamics.actuation_size,) )
    acceleration_compensation = eye( dynamics.state_size // 2 )[ dynamics.six_dof_actuation_mask, : ]

    proportional[ dynamics.br_0_linear_actuation[ :2 ] ] *= 100.0
    proportional[ dynamics.br_0_linear_actuation[ 2 ] ] *= 100.0
    proportional[ dynamics.br_0_angular_actuation ] *= 0.1
    proportional[ dynamics.br_1_linear_actuation[ :2 ] ] *= 100.0

    proportional[ dynamics.br_1_linear_actuation[ 2 ] ] *= 100.0
    proportional[ dynamics.br_1_angular_actuation ] *= 0.1
    proportional[ dynamics.br_2_linear_actuation[ :2 ] ] *= 100.0
    proportional[ dynamics.br_2_linear_actuation[ 2 ] ] *= 100.0
    proportional[ dynamics.br_2_angular_actuation ] *= 0.1
    proportional[ dynamics.br_3_linear_actuation ] *= 100.0
    proportional[ dynamics.br_3_angular_actuation ] *= 0.3

    integral[ dynamics.br_0_linear_actuation[ :2 ] ] *= 10.0
    integral[ dynamics.br_0_linear_actuation[ 2 ] ] *= 10.0
    integral[ dynamics.br_0_angular_actuation ] *= 0.0
    integral[ dynamics.br_1_linear_actuation[ :2 ] ] *= 10.0
    integral[ dynamics.br_1_linear_actuation[ 2 ] ] *= 10.0
    integral[ dynamics.br_1_angular_actuation ] *= 0.0
    integral[ dynamics.br_2_linear_actuation[ :2 ] ] *= 10.0
    integral[ dynamics.br_2_linear_actuation[ 2 ] ] *= 10.0
    integral[ dynamics.br_2_angular_actuation ] *= 0.0
    integral[ dynamics.br_3_linear_actuation ] *= 10.0
    integral[ dynamics.br_3_angular_actuation ] *= 0.0

    derivative[ dynamics.br_0_linear_actuation[ :2 ] ] *= 6500.0
    derivative[ dynamics.br_0_linear_actuation[ 2 ] ] *= 5000.0
    derivative[ dynamics.br_0_angular_actuation ] *= .0
    derivative[ dynamics.br_1_linear_actuation[ :2 ] ] *= 6500.0
    derivative[ dynamics.br_1_linear_actuation[ 2 ] ] *= 5000.0
    derivative[ dynamics.br_1_angular_actuation ] *= .0
    derivative[ dynamics.br_2_linear_actuation[ :2 ] ] *= 6500.0
    derivative[ dynamics.br_2_linear_actuation[ 2 ] ] *= 5000.0
    derivative[ dynamics.br_2_angular_actuation ] *= .0
    derivative[ dynamics.br_3_linear_actuation ] *= 6500.0
    derivative[ dynamics.br_3_angular_actuation ] *= .0

    offset[ dynamics.br_0_linear_actuation[ 2 ] ] = 18.25
    offset[ dynamics.br_1_linear_actuation[ 2 ] ] = 18.25
    offset[ dynamics.br_2_linear_actuation[ 2 ] ] = 18.25

    br_actuation_projection_matrix = zeros( (dynamics.br_0.actuation_size, 6) )
    br_actuation_projection_matrix[:3, :3] = eye(3)
    br_actuation_projection_matrix[-1, -1] = 1.0

    br_actuation_offset = array([0.0, 0.0, -18.25, 0.0])
    
    usv_actuation_projection_matrix = zeros( (dynamics.br_3.actuation_size, 6) )
    usv_actuation_projection_matrix[0, 0] = 1.0
    usv_actuation_projection_matrix[-1, -1] = 1.0

    desired_feature_1 = array([0.9, 0.0, 0.75 ])
    desired_feature_2 = array([0.5, 0.0, 0.25 ])
    desired_feature_3 = array([0.9, 0.75, 0.75 ])

    vs_gain_01 = eye(6)
    vs_gain_01[:2, :2] *= 0.1
    vs_gain_01[2, 2] *= 3.0
    vs_gain_01[5, 5] *= 0.005

    vs_gain_12 = eye(6)
    vs_gain_12[:2, :2] *= 0.1
    vs_gain_12[2, 2] *= 3.0
    vs_gain_12[5, 5] *= 0.005

    vs_gain_23 = eye(6)
    vs_gain_23[:2, :2] *= 0.1
    vs_gain_23[2, 2] *= 3.0
    vs_gain_23[5, 5] *= 0.005

    pid_0 = PID(
            model=model,
            target=trajectory[ 0 ],
            proportional=proportional,
            integral=integral,
            derivative=derivative,
            offset=offset,
            anti_windup_limit=1e3,
            record=record,
            verbose=False
    )

    vs_01 = VS(
        leader_pose=model.state[ dynamics._br_0_pose ],
        follower_pose=model.state[ dynamics._br_1_pose ],
        target_feature=desired_feature_1,
        cable_length=dynamics.c_01.length,
        gain=vs_gain_01,
        actuation_projection_matrix=br_actuation_projection_matrix,
        actuation_offset=br_actuation_offset,
        record=True
    )

    vs_12 = VS(
        leader_pose=model.state[ dynamics._br_1_pose ],
        follower_pose=model.state[ dynamics._br_2_pose ],
        target_feature=desired_feature_2,
        cable_length=dynamics.c_01.length,
        gain=vs_gain_12,
        actuation_projection_matrix=br_actuation_projection_matrix,
        actuation_offset=br_actuation_offset,
        record=True
    )

    vs_23 = VS(
        leader_pose=model.state[ dynamics._br_2_pose ],
        follower_pose=model.state[ dynamics._br_3_pose ],
        target_feature=desired_feature_3,
        cable_length=dynamics.c_01.length,
        gain=vs_gain_23,
        actuation_projection_matrix=usv_actuation_projection_matrix,
        maximum_H = 2 * dynamics.c_01.length / 3,
        record=True
    )

    print(f'{vs_12.compute_feature()=}')

    logger = Logger()

    if record:
        previous_nfeval_record = [ 0 ]
        previous_H01_record = [ 0. ]
        previous_H12_record = [ 0. ]
        previous_H23_record = [ 0. ]

        save_rate = int( .5 / time_step ) if time_step <= .1 else 1
        count_before_save = 0

        folder = join(
                split( __file__ )[ 0 ], 'export', split( __file__ )[ 1 ].split( '.' )[ 0 ] + '_' + str( int( time() ) )
        )

        if check( folder ) + check( f'{folder}/data' ):
            exit()

        with open( f'{folder}/config.json', 'w' ) as f:
            dump( get_computer_info() 
                | { 
                    'save_rate': save_rate,
                    'target_trajectory': trajectory,
                    'pid_0': pid_0.__dict__, 
                    'vs_01': vs_01.__dict__, 
                    'vs_12': vs_12.__dict__,
                    'vs_23': vs_23.__dict__
                    }, 
                    f, 
                    default=serialize_others 
                    )

        with open( f'{folder}/config.json' ) as f:
            config = load( f )
            print_dict( config )

    if 'y' != input( 'run this simulation ? (y/n) ' ):
        exit()

    for frame in range( n_frames ):

        logger.log( f'frame {frame + 1}/{n_frames}' )
        logger.log( f'starts at t={perf_counter() - ti:.2f}' )

        pid_0.target = trajectory[ frame + 1 ]
        vs_01.leader_pose = model.state[ dynamics._br_0_pose ]
        vs_01.follower_pose = model.state[ dynamics._br_1_pose ]
        vs_12.leader_pose = model.state[ dynamics._br_1_pose ]
        vs_12.follower_pose = model.state[ dynamics._br_2_pose ]
        vs_23.leader_pose = model.state[ dynamics._br_2_pose ]
        vs_23.follower_pose = model.state[ dynamics._br_3_pose ]

        model.actuation[dynamics.br_0_actuation] = pid_0.step()[dynamics.br_0_actuation]
        model.actuation[dynamics.br_1_actuation] = vs_01.step()
        model.actuation[dynamics.br_2_actuation] = vs_12.step()
        model.actuation[dynamics.br_3_actuation] = vs_23.step()

        model.step()

        logger.lognl( f'ends at t={perf_counter() - ti:.2f}' )
        # logger.log( f'{pp.raw_result.nit} iterations' )

        logger.log( f'{model.state[ dynamics.position ]}' )
        logger.lognl( f'{model.actuation=}' )
        logger.lognl( f'{vs_01.leader_pose[:3]}|{vs_01.follower_pose[:3]}|{vs_01.compute_feature()}|{vs_01.target_feature}|{norm(model.state[ dynamics.br_1_position ] - model.state[ dynamics.br_0_position ]):.2f}|{norm(model.state[ dynamics.br_1_position[:2] ] - model.state[ dynamics.br_0_position[:2] ]):.2f}' )

        logger.lognl( f'{vs_12.leader_pose[:3]}|{vs_12.follower_pose[:3]}|{vs_12.compute_feature()}|{vs_12.target_feature}|{norm(model.state[ dynamics.br_2_position ] - model.state[ dynamics.br_1_position ]):.2f}|{norm(model.state[ dynamics.br_2_position[:2] ] - model.state[ dynamics.br_1_position[:2] ]):.2f}' )
        
        logger.lognl( f'{vs_23.leader_pose[:3]}|{vs_23.follower_pose[:3]}|{vs_23.compute_feature()}|{vs_23.target_feature}|{norm(model.state[ dynamics.br_3_position ] - model.state[ dynamics.br_2_position ]):.2f}|{norm(model.state[ dynamics.br_3_position[:2] ] - model.state[ dynamics.br_2_position[:2] ]):.2f}' )
        # logger.log( f'{model.state[ dynamics.br_3_orientation ]}' )

        logger.lognl( '' )

        if record:
            logger.save_at( folder )

            count_before_save += 1
            if count_before_save >= save_rate:
                count_before_save = 0
                print( 'saving state ...' )
                with open( f'{folder}/data/{int( frame / save_rate )}.json', 'w' ) as f:
                    dump( get_computer_info() 
                        | { 
                        'save_rate': save_rate,
                        'pid_0': pid_0.__dict__, 
                        'vs_01': vs_01.__dict__, 
                        'vs_12': vs_12.__dict__,
                        'vs_23': vs_23.__dict__
                        }, 
                        f, 
                        default=serialize_others 
                    )
