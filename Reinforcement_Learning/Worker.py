import sys
sys.path.append('/home/jordanyeomans/Github/')

import Agent
import MontyAgent
import CommandCenter
import PhoenixFileManagement.FileManager as FileManager
import time

if __name__ == '__main__':

    Phoenix = FileManager.PhoenixFileSystem(project='MontezumaRevenge', local_homepath='/home/jordanyeomans/PhoenixFiles')

    worker_reset = False # When true, we are only working with Worker 1

    CommandCenter = CommandCenter.CommandCenter(Phoenix.non_synced_folder_path, Phoenix.model_path, Phoenix.data_sim_path, worker=True, reset_worker=worker_reset)

    env_name = 'MontezumaRevenge-v0'
    training = True
    render = False
    verbose = False

    Monty = MontyAgent.Monty()

    print('Worker Version', CommandCenter.model_version)

    Agent = Agent.Agent(env_name, Monty, CommandCenter, render=render, use_logging=True, verbose=verbose)
    Agent.run_worker(Monty, CommandCenter)