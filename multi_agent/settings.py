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
    "movementCost":5, #it's half of what the rewards for each new cell explored

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
    # mas number of steps for the environment
    "max_steps":400,
    "bonus_reward":1000,
    "collision_reward":-1000,
    "out_of_bounds_reward":-1000,

    # ======== SENSORS | LIDAR =======
    "lidar_range":3,
    "lidar_channels":32,

    # ======== COMMUNICATION =======
    "comm_range": 3.0,

    # ======== VIEWER =========
    "viewer":{"width":21*30,
              "height":21*30,
              "title":"Indoor-Explorers-V01",
              "drone_img":'/home/thedarkcurls/IndoorExplorers/img/drone.png',
              "obstacle_img":'/home/thedarkcurls/IndoorExplorers/img/stone_black2.png',
              "background_img":'/home/thedarkcurls/IndoorExplorers/img/wood_floor.jpg',
              "light_mask":"/home/thedarkcurls/IndoorExplorers/img/light_350_hard.png",
              "night_color":(20, 20, 20),
              "draw_lidar":True,
              "draw_grid":True,
              "draw_traceline":False,
              "print_map": False,
              "print_prompts": False
             }
}
