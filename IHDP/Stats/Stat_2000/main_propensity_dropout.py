# from Graphs import Graphs
from Experiments import Experiments

from Model25_10_25 import Model_25_1_25
# from DCN_PD_test import DCN_PD_Deep

if __name__ == '__main__':
    # print("Using original data")
    # Experiments().run_all_experiments(iterations=100, running_mode="original_data")
    print("Using synthetic data")
    Experiments().run_all_experiments(iterations=100, running_mode="synthetic_data")
    # Model_25_1_25().run_all_expeiments()
    # Graphs().draw_scatter_plots()
