import argparse


def get_params():
    parser = argparse.ArgumentParser(
        description="Variable parameters based on the configuration of the machine or user's choice")

    parser.add_argument("--mem_size", default=50000, type=int, help="The memory size.")
    parser.add_argument("--env_name", default="Ant-v2", type=str, help="Name of the environment.")
    parser.add_argument("--interval", default=50, type=int,
                        help="The interval specifies how often different parameters should be saved and printed,"
                             " counted by episodes.")
    parser.add_argument("--do_train", action="store_false",
                        help="The flag determines whether to train the agent or play with it.")

    parser_params = parser.parse_args()

    #  Parameters based on the TD3 paper
    # region default parameters
    default_params = {"lr": 1e-3,
                      "policy_update_period": 2,
                      "pure_explore_steps": 1000,
                      "batch_size": 100,
                      "max_steps": int(1e+6),
                      "gamma": 0.99,
                      "tau": 0.005,
                      }
    # endregion
    total_params = {**vars(parser_params), **default_params}

    return total_params
