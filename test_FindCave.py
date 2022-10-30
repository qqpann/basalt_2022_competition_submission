import logging
import pickle

import aicrowd_gym
import coloredlogs
import minerl
import torch

from agent import MineRLAgent
from config import EVAL_EPISODES, EVAL_MAX_STEPS

coloredlogs.install(logging.DEBUG)

MINERL_GYM_ENV = 'MineRLBasaltFindCave-v0'



def main():
    # NOTE: It is important that you use "aicrowd_gym" instead of regular "gym"!
    #       Otherwise, your submission will fail.
    env = aicrowd_gym.make(MINERL_GYM_ENV)

    # Load your model here
    # NOTE: The trained parameters must be inside "train" directory!
    # model = None
    agent_parameters = pickle.load(open("data/foundation-model-3x.model", "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs, device="cuda" if torch.cuda.is_available() else "cpu")
    agent.load_weights("data/foundation-model-3x.weights")

    for i in range(EVAL_EPISODES):
        obs = env.reset()
        done = False
        for step_counter in range(EVAL_MAX_STEPS):

            # Step your model here.
            # Currently, it's doing random actions
            # for 200 steps before quitting the episode
            # random_act = env.action_space.sample()
            act = agent.get_action(obs)

            obs, reward, done, info = env.step(act)

            if done:
                break
        print(f"[{i}] Episode complete")

    # Close environment and clean up any bigger memory hogs.
    # Otherwise, you might start running into memory issues
    # on the evaluation server.
    env.close()


if __name__ == "__main__":
    main()
