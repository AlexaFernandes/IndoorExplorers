import logging

from gym import envs
from gym.envs.registration import register

from multi_agent.indoor_explorers import IndoorExplorers

logger = logging.getLogger(__name__)

# Register openai's environments as multi agent
# This should be done before registering new environments
# env_specs = [env_spec for env_spec in envs.registry.all() if 'gym.envs' in env_spec.entry_point]
# for spec in env_specs:
#     register(
#         id='ma_' + spec.id,
#         entry_point='ma_gym.envs.openai:MultiAgentWrapper',
#         kwargs={'name': spec.id, **spec._kwargs}
#     )

# add new environments : iterate over full observability
for game_info in [[(16, 16), 4],[(21, 21), 2], [(21, 21), 4],[(48, 48), 4]]:  # [(grid_shape, predator_n, prey_n),..]
    grid_shape, n_agents = game_info
    _game_name = 'IndoorExplorers{}x{}'.format(grid_shape[0], grid_shape[1])

    register(
            id='{}-v0'.format(_game_name),
            entry_point='multi_agent:IndoorExplorers',
            kwargs={
                #'grid_shape': grid_shape#, 'n_agents': n_agents
            }
        )

# for game_info in [[(5, 5), 2, 1], [(7, 7), 4, 2]]:  # [(grid_shape, predator_n, prey_n),..]
#     grid_shape, n_agents, n_preys = game_info
#     _game_name = 'PredatorPrey{}x{}'.format(grid_shape[0], grid_shape[1])
#     register(
#         id='{}-v0'.format(_game_name),
#         entry_point='ma_gym.envs.predator_prey:PredatorPrey',
#         kwargs={
#             'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys
#         }
#     )
    
