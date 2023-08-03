from gym.envs.registration import register

register(
    id='explorer-v01',
    entry_point='indoor_explorers.envs:Explorer',
)

register(
    id='exploConf-v01',
    entry_point='indoor_explorers.envs:ExplorerConf',
)
