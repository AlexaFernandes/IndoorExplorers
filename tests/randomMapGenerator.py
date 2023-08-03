from indoor_explorers.envs.settings.settings import DEFAULT_CONFIG as conf
from indoor_explorers.utils.randomMapGenerator import Generator
import matplotlib.pyplot as plt

g=Generator(conf)
plt.imshow(g.get_map())
plt.show()
