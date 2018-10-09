import sys
sys.path.append('/home/jordanyeomans/Github/')

import argparse
import Agent
import MontyAgent
import CommandCenter
import PhoenixFileManagement.FileManager as FileManager
import time
import numpy as np

if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description='Modify Functionality')
    parser.add_argument("--start_pause", type=int, default=0)
    parser.add_argument("--reset_worker", type=bool, default=False)
    parser.add_argument("--epsilon_adjust", type=float, default=1)
    args = parser.parse_args()

    # Pause before starting
    time.sleep(args.start_pause)

    # Assign Booleans
    worker_reset = args.reset_worker
    training = True
    render = False
    verbose = False

    epsilon_adjust = args.epsilon_adjust
    epsilon_clip = [0.03, 0.3]
    env_name = 'MontezumaRevenge-v0'

    # Create Objects
    Phoenix = FileManager.PhoenixFileSystem(project='MontezumaRevenge', local_homepath='/home/jordanyeomans/PhoenixFiles')
    CommandCenter = CommandCenter.CommandCenter(Phoenix.non_synced_folder_path, Phoenix.model_path, Phoenix.data_sim_path, worker=True, reset_worker=worker_reset)
    Monty = MontyAgent.Monty(epsilon_adjust=epsilon_adjust, epsilon_clip=epsilon_clip)

    print('Worker {}, Version {}'.format(CommandCenter.worker_id, CommandCenter.model_version))

    Agent = Agent.Agent(env_name, Monty, CommandCenter, render=render, use_logging=True, verbose=verbose)
    Agent.run_worker(Monty, CommandCenter)