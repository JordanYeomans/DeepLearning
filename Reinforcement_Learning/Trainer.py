import sys
sys.path.append('/home/jordanyeomans/Github/')

import Reinforcement_Learning.Agent as Agent
import Reinforcement_Learning.MontyAgent as MontyAgent
import Reinforcement_Learning.CommandCenter as CommandCenter

import PhoenixFileManagement.FileManager as FileManager

if __name__ == '__main__':

    Phoenix = FileManager.PhoenixFileSystem(project='MontezumaRevenge', local_homepath='/home/jordanyeomans/PhoenixFiles')

    trainer = True
    reset_trainer = False

    CommandCenter = CommandCenter.CommandCenter(Phoenix.non_synced_folder_path, Phoenix.model_path, Phoenix.data_sim_path, worker=worker, trainer=trainer, reset_trainer=reset_trainer)

    env_name = 'MontezumaRevenge-v0'
    verbose = False

    Monty = MontyAgent.Monty()

    Agent = Agent.Agent(env_name, Monty, CommandCenter, render=render, use_logging=True, verbose=verbose)
    Agent.run_trainer(CommandCenter)