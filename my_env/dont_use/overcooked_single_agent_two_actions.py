from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.actions import Action

import random

import gymnasium as gym
import numpy as np

import torch
#device = torch.device('cpu')
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


mdp = OvercookedGridworld.from_layout_name("cramped_room_o_3orders")    
# "cramped_room_o_3orders" 레이아웃을 사용하여 MDP 생성
env = OvercookedEnv.from_mdp(mdp, horizon=400, info_level=0)

ACTION_MAP = {
    0: (0, -1),   # NORTH
    1: (0, 1),    # SOUTH
    2: (1, 0),    # EAST
    3: (-1, 0),   # WEST
    4: (0, 0),    # STAY
    5: "interact" # INTERACT
}
class OvercookedSingleAgentTwoActionsGymEnv(gym.Env):
    def __init__(self, overcooked_env, agent_idx=0, model=None):
        self.overcooked_env = overcooked_env
        self.agent_idx = agent_idx
        self.model = model
        self.action_space = gym.spaces.MultiDiscrete([6, 6])
        obs = self._get_obs()
        if isinstance(obs, tuple):
            obs = np.array(obs).flatten()
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=obs.shape, dtype=np.int64
        )

    def _get_obs(self):
        state = self.overcooked_env.state

        return self.overcooked_env.lossless_state_encoding_mdp(state)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.overcooked_env.reset()
        obs = self._get_obs()
        if isinstance(obs, tuple):
        # 튜플을 하나의 배열로 합치거나 첫 번째 요소만 사용
            obs = np.array(obs).flatten()
        info = {}
        return obs, info

    def step(self, action):
        obs = self._get_obs()
        if self.agent_idx == 0:
            #actions1 = action // len(Action.ALL_ACTIONS)
            #actions2 = action % len(Action.ALL_ACTIONS)
            actions = [ACTION_MAP[action[0]], ACTION_MAP[action[1]]]
        else:
            #actions1 = action // len(Action.ALL_ACTIONS)
            #actions2 = action % len(Action.ALL_ACTIONS)            
            actions = [ACTION_MAP[action[1]], ACTION_MAP[action[0]]]
        # 2. 환경에 두 에이전트의 행동을 동시에 적용해서 다음 상태, 보상, 종료 여부, 추가 정보를 얻음
        next_state, reward, done, info = self.overcooked_env.step(actions)

        # 3. 환경의 현재 상태를 관찰(observation)으로 가져옴
        obs = self._get_obs()
        # 4. 만약 obs가 튜플 형태라면 numpy 배열로 바꾸고 1차원으로 평탄화함
        if isinstance(obs, tuple):
            obs = np.array(obs).flatten()

        # 5. 강화학습 알고리즘이 요구하는 형식대로
        #    (관찰, 보상, 종료, 에피소드 중단 여부, 추가 정보)를 리턴 
        return obs, reward, done, False, info


    def render(self, mode="rgb-array"):
        print(self.overcooked_env)




env = OvercookedSingleAgentTwoActionsGymEnv(overcooked_env=env)
check_env(env)  # 에러 없으면 학습 가능

# new_model = PPO("MlpPolicy", env, verbose=1, device="cpu")
# pretrained_model = PPO.load("ppo_overcooked", env, verbose=1)
# new_model.learn(total_timesteps=1000000)
# new_model.save("new_ppo_overcooked")

PPO_model = PPO("MlpPolicy", env, verbose=1, device="cpu")
PPO_model.learn(total_timesteps=1000000)
PPO_model.save("testing")
# obs = env.reset()
# done = False
# while not done:
#     action, _ = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#     env.render()