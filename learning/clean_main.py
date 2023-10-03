#fazer com base no exemplo main do DDDQN repo

from learning.dddqn import DDDQNAgent
#from utils import install_roms_in_folder

MOVES = [0,1,2,3,4] #DOWN, LEFT, UP, RIGHT, NO OP 

if __name__ == '__main__':

    #install roms
    #install_roms_in_folder('roms/')

    #create agent
    dddqn = DDDQNAgent('IndoorExplorers', MOVES, epsilon_decay=0.99999, batch_size=32)
    dddqn.q_eval.summary()
    dddqn.q_target.summary()
    
    #train agent
    dddqn.run(num_episodes=100,  render=True, checkpoint=True, cp_interval=20, cp_render=True, n_intelligent_agents = 1)
    
    #load model
    dddqn.load('models')#, 'DDDQN_1000_IndoorExplorers_09222023161252_QEval.h5', 'DDDQN_1000_IndoorExplorers_09222023161252_QTarget.h5')
    
    #play game
    dddqn.play_episode(render=True, render_and_save=True, otype='GIF')