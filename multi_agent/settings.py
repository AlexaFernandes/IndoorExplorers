DEFAULT_CONFIG={
    # ======== TOPOLOGY =======
    #  approach (centralized or not centralized)
    "approach": False,
    #  number of agents
    "n_agents": 4,
    #  general configuration for the topology of operational area
    "random_spawn": True, # if set to true, initial pos are ignored
    "initial": [ [0,0],
                 [0,2],
                 [3,0],
                 [3,1]
                ],
    # "initial": {0: [0,0],
    #             1: [0,20],
    #             2 :[20,0],
    #             3: [20,20]
    #             },
    "size":[16,16],
    #  configuration regarding the movements of uav
    "percentage_explored": 0.9, #goal percentage of the map to be explored

    # ======== ENVIROMENT =======
    # configuration regarding the random map generation
    # absolute number of obstacles, randomly placed in env
    "obstacles":5,
    # if rows/colums activated the obstacles will be placed in a semi random
    # spacing
    "number_rows":None,
    "number_columns":None,
    # noise activated only when row/columns activated
    # maximum noise on each axes
    "noise":[0,0],
    # margins expressed in cell if rows/columns not activated
    "margins":[1, 1],
    # obstacle size expressed in cell if rows/columns not activated
    "obstacle_size":[3,3],

    # flag to activate the verification check of an agent being stuck
    "check_stuck": True,
    # method to check if it is stuck
        #1: count the number of steps where the agent does not discover any new cell, if it reaches height*width => agent is stuck
        #2: registers the last 50 positions, and if the most common one is repeated 10 times, then it is stuck
    "stuck_method": 2,

    # max number of steps for the environment
    "max_steps":400,

    # ======== REWARDS ===========
    "movementCost": 10, #this is discounted for every time step/every movement made (don't put the minus sign!!)
    "new_cell_disc_reward": 10, #reward value for each new cell discovered
    "bonus_reward": 1000, #reward for exploring "percentage_explored"% of the map
    "stuck_reward": -1000, #penalty for getting stuck between positions
    "collision_reward":-1000, #penalty for colliding with walls
    "out_of_bounds_reward":-1000, #penalty for going out of the bounds of the map
    

    # ======== SENSORS | LIDAR =======
    "lidar_range":3, #defines the LiDAR range
    "lidar_channels":32,

    # ======== COMMUNICATION =======
    "comm_range": 3.0, #defines the communication range of each agent

    # ======== VIEWER =========
    "viewer":{"width":21*30,
              "height":21*30,
              "title":"Indoor-Explorers-V01",
              "drone_img":'/home/thedarkcurls/IndoorExplorers/img/drone.png',
              "obstacle_img":'/home/thedarkcurls/IndoorExplorers/img/stone_black2.png',
              "background_img":'/home/thedarkcurls/IndoorExplorers/img/wood_floor.jpg',
              "light_mask":"/home/thedarkcurls/IndoorExplorers/img/light_350_hard.png",
              "night_color":(20, 20, 20),
              "draw_lidar":True, #not used
              "draw_grid":True, #not used
              "draw_traceline":False, #not used
              "print_map": True, #enables the visualization of the map of each agent individualy, when rendering is active
              "print_prompts": False #enables the visualization of the map of each agent individualy, when rendering is active
             }
}
