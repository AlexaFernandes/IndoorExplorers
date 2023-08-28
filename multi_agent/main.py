from gym.envs.registration import register

from indoor_explorers.utils.printMaps import printMap
from indoor_explorers.utils.randomMapGenerator import Generator
from indoor_explorers.utils.lidarSensor import Lidar
from indoor_explorers.render.viewer import Viewer
from ma_gym.envs.indoor_explorers.settings import DEFAULT_CONFIG as conf


from ma_gym.wrappers import Monitor



for game_info in [[(21, 21), 2], [(21, 21), 4]]:  # [(grid_shape, predator_n, prey_n),..]
    grid_shape, n_agents = game_info
    _game_name = 'IndoorExplorers{}x{}'.format(grid_shape[0], grid_shape[1])

    register(
            id='{}-v0'.format(_game_name),
            entry_point='ma_gym.envs.indoor_explorers:IndoorExplorers',
            kwargs={
                'grid_shape': grid_shape, 'n_agents': n_agents
            }
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Agent for ma-gym')
    parser.add_argument('--env', default='IndoorExplorers21x21-v0',
                        help='Name of the environment (default: %(default)s)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='episodes (default: %(default)s)')
    args = parser.parse_args()

    env = gym.make(args.env, conf=conf)
    env = Monitor(env, directory='recordings/' + args.env, force=True)
    #env = Monitor(env, directory='recordings/'+ args.env, video_callable=lambda episode_id: True, force = True) # saves all of the episodes
    #env = Monitor(env, directory='recordings' + args.env, video_callable=lambda episode_id: episode_id%10==0) #saves the 10th episode
    for ep_i in range(args.episodes):
        done_n = [False for _ in range(env.n_agents)]
        ep_reward = 0

        env.seed(ep_i)
        obs_n = env.reset()
        env.render()

        while not all(done_n):
            action_n = env.action_space.sample() #insert policy
            obs_n, reward_n, done_n, info = env.step(action_n)
            ep_reward += sum(reward_n)
            env.render()
            time.sleep(0.1)

        print('Episode #{} Reward: {}'.format(ep_i, ep_reward))
    env.close()