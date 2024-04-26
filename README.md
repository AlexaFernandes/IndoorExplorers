# IndoorExplorers
## Description
This is OpenAI Gym environemnt developed for my Master's dissertation. 
It provides a framework for testing RL algorithms in a multi-agent exploration scenario, in unknown areas.
Each agent has a limited field of vision by an emulated LiDAR sensor, and agents may or may not have a limited communication range.
There is a limit of 4 agents at the moment, that can exchange map information when in communication range.
It has an implementation of Dueling Double Deep Q Network (DDDQN), where one agent is training the model, while the rest are dummy agents. In evaluation episodes, the learnt model is loaded into all agents.

Por imagem

## How to use
1. Clone this repository
2. Create a virtual environment
  I used [miniconda3](https://docs.anaconda.com/free/miniconda/): (I think you can use others if you prefer)

  Here is a short basic tutorial: [link](https://medium.com/@aminasaeed223/a-comprehensive-tutorial-on-miniconda-creating-virtual-environments-and-setting-up-with-vs-code-f98d22fac8e2)
  ```
  conda create --name indoor_explorers_env -f environment.yml
  ```
3. Activate the env:
  ```
  conda activate indoor_explorers_env
  ```
4. Setup parameters in settings.py in multi_agents folder
5. To run (aka train a model):
   ```
   python learning/clean_main.py
   ```
   The results can be found in the learning folder, separated by:
   * models - a model is generated every _cp_interval_ of episodes (this value is defined on clean_main.py)
   * logs - a cvs file in stats is created every _cp_interval_ number of episodes -> can be used to create trendlines of the reward evolution
   * renders - gifs produced after rendering at each _cp_interval_ number of episodes (the saving of gifs can be deactivated by cp_render=False -> better explained inside the run() function)
   * for_merging - file produced in the end of training, with distance covered troughout the steps, that can be used to produce graphs comparing different models -> check non-learning/mergingData.py (different aspects can be compared) -     **make sure the .p files you want to compare are in the correct folder of your choosing**
   * Example:
   * por imagem

[DOCUMENTATION IN PROGRESS]









## Credits
This environment was based on:
* [Mars explorer](https://github.com/dimikout3/MarsExplorer/tree/main)
* [Ma-gym](https://github.com/koulanurag/ma-gym)
* [Multiagent.particle.envs](https://github.com/openai/multiagent-particle-envs/tree/master)
* [Implementation of DDDQN](https://github.com/rjalnev/DDDQN)



