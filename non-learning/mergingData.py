import numpy as np
import time
import json
import pickle as p
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


plotstyle="ggplot"
plt.style.use(f"{plotstyle}")


CASES = [
        ["/home/thedarkcurls/IndoorExplorers/learning/for_merging/50000epi_IndoorExplorers16x16_1agents_5obst_comm0.0_noStuck_14022024112434.p","1 agent","No "]# TODO
        # ["non-learning/for_merging/50000epi_IndoorExplorers16x16_1agents_movCost0.5_28102023051724.p","No stack frame", "Stuck 2" ],
        # ["non-learning/for_merging/50000epi_IndoorExplorers16x16_1agents_movCost0.5_07112023022259.p", "With stack frame", "No stuck" ],
        # ["non-learning/for_merging/50000epi_IndoorExplorers16x16_1agents_movCost0.5_07112023204151.p", "With stack frame", "Stuck 2" ]
        #   ["non-learning/for_merging/50000epi_IndoorExplorers16x16_1agents_movCost0.5_28102023051724.p","1 agent", "No comms" ],          
        #   ["non-learning/for_merging/50000epi_IndoorExplorers16x16_2agents_movCost0.5_28102023233923.p","2 agents", "No comms" ],
        #   ["non-learning/for_merging/50000epi_IndoorExplorers16x16_2agents_movCost0.5_28102023195129.p","2 agents", "Comms range 1.0" ],
        #   ["non-learning/for_merging/50000epi_IndoorExplorers16x16_2agents_movCost0.5_28102023161334.p","2 agents", "Comms range 3.0" ],
        #   ["non-learning/for_merging/50000epi_IndoorExplorers16x16_4agents_movCost0.5_29102023054705.p","4 agents", "No comms" ]

         ]


#with obstacle 5
        #   ["non-learning/for_merging/50000epi_IndoorExplorers16x16_1agents_movCost0.5_28102023051724.p","1 agent", "No comms" ],          
        #   ["non-learning/for_merging/50000epi_IndoorExplorers16x16_2agents_movCost0.5_28102023233923.p","2 agents", "No comms" ],
        #   ["non-learning/for_merging/50000epi_IndoorExplorers16x16_2agents_movCost0.5_28102023195129.p","2 agents", "Comms range 1.0" ],
        #   ["non-learning/for_merging/50000epi_IndoorExplorers16x16_2agents_movCost0.5_28102023161334.p","2 agents", "Comms range 3.0" ],
        #   ["non-learning/for_merging/50000epi_IndoorExplorers16x16_4agents_movCost0.5_29102023054705.p","4 agents", "No comms" ]


#1 agent noObst
        #  ["non-learning/for_merging/10000epi_IndoorExplorers16x16_1agents_movCost0.5_19102023211739.p","Scenario 1", "No stuck" ],
        #  ["non-learning/for_merging/10000epi_IndoorExplorers16x16_1agents_movCost0.5_19102023223753.p","Scenario 1", "Stuck 1"],
        #  ["non-learning/for_merging/10000epi_IndoorExplorers16x16_1agents_movCost0.5_19102023234550.p", "Scenario 1", "Stuck2"],

        #  ["non-learning/for_merging/10000epi_IndoorExplorers16x16_1agents_movCost1_19102023160526.p","Scenario 2", "No stuck" ],
        #  ["non-learning/for_merging/10000epi_IndoorExplorers16x16_1agents_movCost1_19102023181841.p","Scenario 2", "Stuck 1"],
        #  ["non-learning/for_merging/10000epi_IndoorExplorers16x16_1agents_movCost1_28102023022413.p", "Scenario 2", "Stuck2"], #3rd try!

        #  ["non-learning/for_merging/10000epi_IndoorExplorers16x16_1agents_movCost10_19102023133424.p","Scenario 3", "No stuck" ],
        #  ["non-learning/for_merging/10000epi_IndoorExplorers16x16_1agents_movCost10_20102023004543.p","Scenario 3", "Stuck 1"],#2nd try
        #  ["non-learning/for_merging/10000epi_IndoorExplorers16x16_1agents_movCost10_28102023030334.p", "Scenario 3", "Stuck2"] #2nd try


#1 agent snc2 Obstacles 5 - OLD
# ["non-learning/for_merging/50000epi_IndoorExplorers16x16_1agents_movCost10_20102023153038.p","Scenario 2", "No stuck" ],
# ["non-learning/for_merging/50000epi_IndoorExplorers16x16_1agents_movCost10_21102023132032.p","Scenario 2", "Stuck 1"],
# ["non-learning/for_merging/50000epi_IndoorExplorers16x16_1agents_movCost10_20102023133637.p", "Scenario 2", "Stuck2"]


if __name__ == "__main__":

    dict = {}
    #dict["Terrain"] = []
    dict["Scenario"] = []
    dict["Steps"] = []
    dict["Explored Area [%]"] = []
    dict["Exploration Policy"] = []

    for file, area, exploration_type in CASES:

        data = p.load(open(file,"rb"))

        for game in data[0]:
            for distance,explored in enumerate(game):
                dict["Scenario"].append(area)
                dict["Steps"].append(distance)
                dict["Explored Area [%]"].append(explored)
                dict["Exploration Policy"].append(exploration_type)

    df = pd.DataFrame(dict)

    sns.lineplot(data=df, x="Steps", y="Explored Area [%]",
                 hue="Exploration Policy", style="Scenario").set(title='Percentage of area explored by a single agent \n in a 16x16 map with 5 obstacles and stack frame\n with different stuck methods')
    plt.show()
