import sys
sys.path.append('/home/jordanyeomans/Github/')

import Agent
import MontyAgent
import CommandCenter
import PhoenixFileManagement.FileManager as FileManager
import time


if __name__ == '__main__':

    Phoenix = FileManager.PhoenixFileSystem(project='MontezumaRevenge', local_homepath='/home/jordanyeomans/PhoenixFiles')

    reset = True # When true, we are only working with Worker 1

    # version = 1.21
    version = None
    CommandCenter = CommandCenter.CommandCenter(Phoenix.non_synced_folder_path, Phoenix.model_path, Phoenix.data_sim_path, worker=True, reset_worker=reset, version=version)

    env_name = 'MontezumaRevenge-v0'
    training = True
    render = True
    verbose = False

    Monty = MontyAgent.Monty()

    print('Worker Version', CommandCenter.model_version)

    Agent = Agent.Agent(env_name, Monty, CommandCenter, render=render, use_logging=True, verbose=verbose)
    Agent.run_worker(Monty, CommandCenter)