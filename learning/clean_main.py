#fazer com base no exemplo main do DDDQN repo

from learning.dddqn import DDDQNAgent
#from utils import install_roms_in_folder

#define all the envs we want to test (dont forget to also register these env on multi_agent >>__init__.py )
GAMES = []
for game_info in [[(16, 16), 4], [(21, 21), 2], [(21, 21), 4],[(48, 48), 4]]:  # [(grid_shape, predator_n, prey_n),..]
    grid_shape, n_agents = game_info
    _game_name = 'IndoorExplorers{}x{}'.format(grid_shape[0], grid_shape[1])
    GAMES.append(_game_name)

MOVES = [0,1,2,3,4] #DOWN, LEFT, UP, RIGHT, NO OP 

if __name__ == '__main__':

    #install roms
    #install_roms_in_folder('roms/')

    #create agent
    dddqn = DDDQNAgent(GAMES[0], MOVES, epsilon_decay=0.99999, batch_size=32) #'IndoorExplorers'
    dddqn.q_eval.summary()
    dddqn.q_target.summary()
    
    #train agent
    dddqn.run(num_episodes=10000,  render=False, checkpoint=True, cp_interval=200, cp_render=True, n_intelligent_agents = 1)
    
    #load model
    dddqn.load('learning/models')# it is possible to specify: , 'DDDQN_1000_IndoorExplorers_09222023161252_QEval.h5', 'DDDQN_1000_IndoorExplorers_09222023161252_QTarget.h5') otherwise it will use the most recent ones
    
    #play game
    dddqn.play_episode(render=True, render_and_save=True, otype='GIF')