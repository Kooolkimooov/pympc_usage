from json import dump, load
from os.path import join, split
from time import perf_counter, time
from warnings import simplefilter

# noinspection PyUnresolvedReferences
from numpy import array, concatenate, cos, diff, eye, inf, max, min, pi, set_printoptions, sqrt, zeros
from numpy.linalg import norm
from scipy.optimize import Bounds, NonlinearConstraint

from pympc.controllers.pid import PID
from pympc.controllers.pp import PP
from pympc.models.dynamics.chain_of_four_with_usv import (
    ChainOf4WithUSV,
    chain_of_4_constraints_pp,
    chain_of_4_objective_pp
)
from pympc.models.model import Model
from pympc.models.seafloor import SeafloorFromFunction, seafloor_function_0
from pympc.utils import Logger, check, generate_trajectory, get_computer_info, print_dict, serialize_others

if __name__ == "__main__":
    simplefilter( 'ignore', RuntimeWarning )
    set_printoptions( precision=2, linewidth=10000, suppress=True )

    ti = perf_counter()

    record = False
    seafloor = SeafloorFromFunction( seafloor_function_0 )

    dynamics = ChainOf4WithUSV(
            water_surface_depth=0.,
            water_current=None, #array( [ sqrt( 2.0 ), 0., 0. ] ),
            seafloor=seafloor,
            cables_length=3.0,
            cables_linear_mass=0.01,
            get_cable_parameter_method='precompute',
            reference_frame='NED'
    )

    time_step = 0.1

    initial_actuation = zeros( (dynamics.actuation_size,) )

    initial_state = zeros( (dynamics.state_size,) )
    initial_state[ dynamics.br_0_position[ 0 ] ] = 2.
    initial_state[ dynamics.br_0_position[ 2 ] ] = 1.
    initial_state[ dynamics.br_1_position[ 0 ] ] = 2.5
    initial_state[ dynamics.br_1_position[ 2 ] ] = 1.
    initial_state[ dynamics.br_2_position[ 0 ] ] = 3.
    initial_state[ dynamics.br_2_position[ 2 ] ] = 1.
    initial_state[ dynamics.br_3_position[ 0 ] ] = 3.5
    initial_state[ dynamics.br_3_orientation[ 2 ] ] = pi / 2

    model = Model(
            dynamics=dynamics,
            time_step=time_step,
            initial_state=initial_state,
            initial_actuation=initial_actuation,
            record=record
    )

    horizon = 10
    time_steps_per_actuation = 10
    n_frames = 500
    tolerance = 1e-3
    max_number_of_iteration = 100

    key_frames = [
            (0., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 18),
            (.5, [ -5., 0., 0., 0., 0., 0. ] + [ 0. ] * 18),
            (1., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 18),
            (2., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 18)
    ]
    trajectory = generate_trajectory( key_frames, 2 * n_frames )
    trajectory[ :, 0, dynamics.br_0_position[ 2 ] ] = 1.5 * cos(
            1.25 * (trajectory[ :, 0, dynamics.br_0_position[ 0 ] ] - 2) + pi
    ) + 2.5

    max_required_speed = max( norm( diff( trajectory[ :, 0, :3 ], axis=0 ), axis=1 ) ) / time_step

    if 'y' != input( f'{max_required_speed=}, continue ? (y/n) ' ):
        exit()

    objective_weight = 0.01
    final_cost_weight = 0.

    pose_weight_matrix = eye( initial_state.shape[ 0 ] // 2 )

    pose_weight_matrix[ dynamics.br_0_position, dynamics.br_0_position ] *= 10.
    pose_weight_matrix[ dynamics.br_0_orientation, dynamics.br_0_orientation ] *= 1.
    pose_weight_matrix[ dynamics.br_1_position, dynamics.br_1_position ] *= 0.
    pose_weight_matrix[ dynamics.br_1_orientation, dynamics.br_1_orientation ] *= 1.
    pose_weight_matrix[ dynamics.br_2_position, dynamics.br_2_position ] *= 0.
    pose_weight_matrix[ dynamics.br_2_orientation, dynamics.br_2_orientation ] *= 1.
    pose_weight_matrix[ dynamics.br_3_position, dynamics.br_3_position ] *= 0.
    pose_weight_matrix[ dynamics.br_3_orientation, dynamics.br_3_orientation ] *= 0.

    bounds_lb_base = array( [ -1.0, -1.0, -1.0, -0.1, -0.1, -0.1 ] )
    bounds_ub_base = array( [ 1.0, 1.0, 1.0, 0.1, 0.1, 0.1 ] )

    bounds_lb_usv = array( [ -1.0, 0.0, 0.0, 0.0, 0.0, -1.0 ] )
    bounds_ub_usv = array( [ 1.0, 0.0, 0.0, 0.0, 0.0, 1.0 ] )

    bounds_lb = concatenate(
            horizon * [
                    bounds_lb_base,
                    bounds_lb_base,
                    bounds_lb_base,
                    bounds_lb_usv
            ]
    )
    bounds_ub = concatenate(
            horizon * [
                    bounds_ub_base,
                    bounds_ub_base,
                    bounds_ub_base,
                    bounds_ub_usv
            ]
    )

    assert bounds_lb.shape[
               0 ] == horizon * dynamics.state_size // 2, f"{bounds_lb.shape=}!={horizon * dynamics.state_size // 2=}"
    assert bounds_ub.shape[
               0 ] == horizon * dynamics.state_size // 2, f"{bounds_ub.shape=}!={horizon * dynamics.state_size // 2=}"

    bounds = Bounds( lb=bounds_lb, ub=bounds_ub )

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

    acceleration_compensation[ dynamics.br_0_linear_actuation ] *= 10.0
    acceleration_compensation[ dynamics.br_0_angular_actuation ] *= 0.9
    acceleration_compensation[ dynamics.br_1_linear_actuation ] *= 10.0
    acceleration_compensation[ dynamics.br_1_angular_actuation ] *= 0.9
    acceleration_compensation[ dynamics.br_2_linear_actuation ] *= 10.0
    acceleration_compensation[ dynamics.br_2_angular_actuation ] *= 0.9
    acceleration_compensation[ dynamics.br_3_linear_actuation ] *= 0.0
    acceleration_compensation[ dynamics.br_3_angular_actuation ] *= 0.0

    pp = PP(
            model=model,
            horizon=horizon,
            optimize_on='trajectory_derivative',
            target_trajectory=trajectory,
            tolerance=tolerance,
            bounds=bounds,
            max_number_of_iteration=max_number_of_iteration,
            pose_weight_matrix=pose_weight_matrix,
            objective_weight=objective_weight,
            final_weight=final_cost_weight,
            record=record
    )

    pid = PID(
            model=model,
            target=trajectory[ 0 ],
            proportional=proportional,
            integral=integral,
            derivative=derivative,
            offset=offset,
            # acceleration_compensation=acceleration_compensation,
            anti_windup_limit=1e3,
            record=record,
            verbose=False
    )

    sf_lb = 0.2
    sf_ub = inf
    dp_lb = 0.2
    dp_ub = inf
    dr_lb = -inf
    dr_ub = 2.8

    constraints_values_labels = [
            'c_01_distance_to_seafloor',
            'c_12_distance_to_seafloor',
            'c_23_distance_to_seafloor',
            'br_0_br_1_horizontal_distance',
            'br_1_br_2_horizontal_distance',
            'br_2_br_3_horizontal_distance',
            'br_0_br_1_distance',
            'br_1_br_2_distance',
            'br_2_br_3_distance'
    ]
    constraints_reason_labels = [
            'seafloor',
            'seafloor',
            'seafloor',
            'cable_length',
            'cable_length',
            'cable_length',
            'cable_length',
            'cable_length',
            'cable_length'
    ]

    constraint_lb_base = [ sf_lb, sf_lb, sf_lb, dp_lb, dp_lb, dp_lb, dr_lb, dr_lb, dr_lb ]
    constraint_ub_base = [ sf_ub, sf_ub, sf_ub, dp_ub, dp_ub, dp_ub, dr_ub, dr_ub, dr_ub ]

    assert (len( constraint_lb_base ) == len( constraints_values_labels )) and (
            len( constraint_ub_base ) == len( constraints_reason_labels )), 'bad definition of constraints'

    constraint_lb = [ constraint_lb_base ] * horizon
    constraint_ub = [ constraint_ub_base ] * horizon

    # inject constraints and objective as member functions so that they may access self
    pp.constraints_function = chain_of_4_constraints_pp.__get__( pp, PP )

    constraint = NonlinearConstraint(
            pp.constraints_function, array( constraint_lb ).flatten(), array( constraint_ub ).flatten()
    )
    constraint.value_labels = constraints_values_labels
    constraint.labels = constraints_reason_labels
    pp.constraints = (constraint,)

    pp.objective = chain_of_4_objective_pp.__get__( pp, PP )

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
            dump( pp.__dict__ | get_computer_info() | { 'save_rate': save_rate }, f, default=serialize_others )

        with open( f'{folder}/config.json' ) as f:
            config = load( f )
            print_dict( config )

    if 'y' != input( 'run this simulation ? (y/n) ' ):
        exit()

    for frame in range( n_frames ):
        pp.target_trajectory = trajectory[ frame + 1: ]

        logger.log( f'frame {frame + 1}/{n_frames}' )
        # logger.log( f'starts at t={perf_counter() - ti:.2f}' )

        pid.target = pp.step()
        # pid.target[ dynamics.br_0_pose ] = pp.target_trajectory[ 0, 0, dynamics.br_0_pose ]
        model.actuation = pid.step()
        model.step()

        # logger.log( f'ends at t={perf_counter() - ti:.2f}' )
        logger.log( f'{pp.raw_result.message == "Optimization terminated successfully"}' )
        # logger.log( f'{pp.raw_result.nit} iterations' )

        logger.log( f'{pp.target_trajectory[ 0, 0, dynamics.br_0_position ]}' )
        logger.log( f'{pid.target[ dynamics.position ]}' )
        logger.log( f'{pid.target[ dynamics.br_3_orientation ]}' )
        logger.log( f'{model.state[ dynamics.position ]}' )
        logger.log( f'{model.state[ dynamics.br_3_orientation ]}' )
        logger.log( f'{model.actuation}' )

        # try to recover if the optimization failed
        if not pp.raw_result.success and pp.tolerance < 1:
            pp.tolerance *= 10
            logger.log( f'increasing tolerance: {pp.tolerance:.0e}' )
        elif pp.raw_result.success and pp.tolerance > 2 * tolerance:
            # *2 because of floating point error
            pp.tolerance /= 10
            logger.log( f'decreasing tolerance: {pp.tolerance:.0e}' )
        else:
            logger.log( f'keeping tolerance: {pp.tolerance:.0e}' )

        objective_value = pp.get_objective()
        # logger.log( f'objective: {objective_value:.2f}' )

        constraints_values = pp.constraints_function( pp.raw_result.x )
        logger.log( f'constraints: {constraints_values[ :len( constraint_lb_base ) ]}' )

        logger.lognl( '' )

        if record:
            logger.save_at( folder )

            count_before_save += 1
            if count_before_save >= save_rate:
                count_before_save = 0
                print( 'saving state ...' )
                with open( f'{folder}/data/{int( frame / save_rate )}.json', 'w' ) as f:
                    dump( pp.__dict__, f, default=serialize_others )
