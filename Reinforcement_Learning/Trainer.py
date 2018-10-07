import sys
sys.path.append('/home/jordanyeomans/Github/')
import argparse
import Agent
import MontyAgent
import CommandCenter

import PhoenixFileManagement.FileManager as FileManager

if __name__ == '__main__':

    Phoenix = FileManager.PhoenixFileSystem(project='MontezumaRevenge', local_homepath='/home/jordanyeomans/PhoenixFiles')

    # Arguments
    parser = argparse.ArgumentParser(description='Modify Functionality')
    parser.add_argument("--reset_trainer", type=bool, default=False)
    parser.add_argument("--version", type=float, default=1)

    args = parser.parse_args()

    reset_trainer = args.reset_trainer
    version = args.version

    CommandCenter = CommandCenter.CommandCenter(Phoenix.non_synced_folder_path, Phoenix.model_path, Phoenix.data_sim_path, worker=False, trainer=True, reset_trainer=reset_trainer, version=version)

    env_name = 'MontezumaRevenge-v0'
    verbose = False

    Monty = MontyAgent.Monty()

    Agent = Agent.Agent(env_name, Monty, CommandCenter, render=False, use_logging=True, verbose=verbose)
    Agent.run_trainer(CommandCenter)