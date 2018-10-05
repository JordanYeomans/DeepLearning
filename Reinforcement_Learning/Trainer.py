import sys
sys.path.append('/home/jordanyeomans/Github/')

import Agent
import MontyAgent
import CommandCenter

import PhoenixFileManagement.FileManager as FileManager

if __name__ == '__main__':

    Phoenix = FileManager.PhoenixFileSystem(project='MontezumaRevenge', local_homepath='/home/jordanyeomans/PhoenixFiles')

    trainer = True
    reset_trainer = False

    CommandCenter = CommandCenter.CommandCenter(Phoenix.non_synced_folder_path, Phoenix.model_path, Phoenix.data_sim_path, worker=False, trainer=trainer, reset_trainer=reset_trainer)

    env_name = 'MontezumaRevenge-v0'
    verbose = False

    Monty = MontyAgent.Monty()

    Agent = Agent.Agent(env_name, Monty, CommandCenter, render=False, use_logging=True, verbose=verbose)
    Agent.run_trainer(CommandCenter)