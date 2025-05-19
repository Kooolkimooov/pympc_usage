from numpy import linspace, zeros

n_steps = 100
total_time = 10
trajectory_name = "placeholder"

time = linspace( 0, total_time, n_steps )

# compute wanted trajectories with poses and velocities; zeros as a placeholder
# pose is in the world frame and velocities are in body frame
trajectory_0 = zeros( (12, n_steps) )
trajectory_1 = zeros( (12, n_steps) )

with open( f'{trajectory_name}_{n_steps=}_{total_time=}.csv', 'w' ) as f:
    f.write( 'br0_x, br0_y, br0_z, br0_phi, br0_theta, br0_psi, br0_u, br0_v, br0_w, br0_p, br0_q, br0_r,' )
    f.write( 'br1_x, br1_y, br1_z, br1_phi, br1_theta, br1_psi, br1_u, br1_v, br1_w, br1_p, br1_q, br1_r\n' )

    for state_0, state_1 in zip( trajectory_0.T, trajectory_1.T ):
        f.write( ','.join( [ f'{v}' for v in state_0 ] ) + ',' )
        f.write( ','.join( [ f'{v}' for v in state_1 ] ) + '\n' )
