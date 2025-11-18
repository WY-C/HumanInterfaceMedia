#모델 미리 불러오기
import gymnasium as gym
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPO
import os
import torch
import random
from ray.rllib.policy import Policy # 👈 Policy를 import 합니다.
import torch.nn.functional as F
from overcooked_ai_py.agents.agent import GreedyHumanModel
from overcooked_ai_py.planning.planners import (
    NO_COUNTERS_PARAMS,
    MediumLevelActionManager,
    MotionPlanner,
)


ACTION_MAP = {
    0: (0, -1),   # NORTH
    1: (0, 1),    # SOUTH
    2: (1, 0),    # EAST
    3: (-1, 0),   # WEST
    4: (0, 0),    # STAY
    5: "interact" # INTERACT
}

#for greedy human model
REVERSE_ACTION_MAP = {
    (0, -1): 0,   # NORTH
    (0, 1): 1,    # SOUTH
    (1, 0): 2,    # EAST
    (-1, 0): 3,   # WEST
    (0, 0): 4,    # STAY
    "interact": 5 # INTERACT
}


class Rllib_multi_agent(MultiAgentEnv):
    #agent1, agent2
    def __init__(self, config = None, reward_shaping = True):
        #이후 config에 layout name, horizon 등등을 넣어야함.
        super().__init__()
        config = config or {}
        layout_name = config.get("layout_name", "cramped_room1")
        horizon = config.get("horizon", 1000000)
        self.reward_shaping = config.get("reward_shaping", True)
        mdp = OvercookedGridworld.from_layout_name(layout_name) 
        self.overcooked_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
        self.count_delivery_soup = 0
        self.previous_trajectory = [0] * 401

        self.agents = ["agent_0", "agent_1"]
        self._agent_ids = {"agent_0", "agent_1"}
        #self._agent_ids = set(self.agents)
        sample_obs_dict = self._get_obs(0)
        flattened_shape = sample_obs_dict['agent_0'].flatten().shape
        self.observation_space = gym.spaces.Dict(
            {
                "agent_0": gym.spaces.Box(low=0, high=1, shape=flattened_shape, dtype=np.float32),
                "agent_1": gym.spaces.Box(low=0, high=1, shape=flattened_shape, dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Dict(
            {
                "agent_0": gym.spaces.Discrete(len(ACTION_MAP)),
                "agent_1": gym.spaces.Discrete(len(ACTION_MAP)),
            }
        )
        
        # self.observation_space = gym.spaces.Box(
        #     low=0, high=1, shape=sample_obs['agent_1'].shape, dtype=np.int32
        # )
        
        # self.action_space = gym.spaces.Discrete(len(ACTION_MAP))
        #print(sample_obs.shape)


    def _get_obs(self, idx = 0):
        state = self.overcooked_env.state

        #print(state.shape)
        obs_tuple = self.overcooked_env.lossless_state_encoding_mdp(state)
        observations = {
            self.agents[0]: obs_tuple[0].flatten().astype(np.float32),
            self.agents[1]: obs_tuple[1].flatten().astype(np.float32),
        }
        #obs1 = np.array(obs).flatten()
        return observations

    #obs, reward 공유됨.
    def reset(self, seed=None, options=None):
        #trajectory 저장용 변수
        self.trajectory = []
        self.timestep = 0
        self.count_delivery_soup = 0
        # print(self.previous_trajectory)

        """환경을 리셋하고 각 에이전트의 초기 관측값을 반환합니다."""
        self.overcooked_env.reset()
        #self.agents = ["agent_1", "agent_2"]
        # 각 에이전트 ID에 대한 관측값을 담은 딕셔너리를 반환합니다.
        obs = self._get_obs()
        return obs, {}

    def step(self, action_dict):

        #print("Received action_dict:", action_dict)
        # if action_dict == {}:
        #     action_dict['agent_1'] = 4
        #     action_dict['agent_2'] = 4
        #     print(1)
        #print(action_dict)
        actions = [ACTION_MAP[action_dict[agent_id]] for agent_id in self.agents]

        next_state, rewards, done, info = self.overcooked_env.step(actions)
        obs = self._get_obs()
        shaped_rewards_list = info["shaped_r_by_agent"]

        if rewards > 20:
            print(self.count_delivery_soup)

            self.count_delivery_soup += 1
        

        if self.reward_shaping == True:
            reward = {
            self.agents[0]: rewards + shaped_rewards_list[0],
            self.agents[1]: rewards + shaped_rewards_list[1],         
            }
        else:
            reward = {
            self.agents[0]: rewards,
            self.agents[1]: rewards,         
        }

        done_dict = {
            self.agents[0]: done,
            self.agents[1]: done,  
        }
        truncated_dict = {
            self.agents[0]: False,
            self.agents[1]: False,  
        }
        done_dict["__all__"] = done
        truncated_dict["__all__"] = False
        info_dict = {
            self.agents[0]: info,
            self.agents[1]: info, 
        }
        self.trajectory.append(repr(self.overcooked_env))
        self.timestep +=1
        if done_dict["__all__"]:
             self.previous_trajectory = self.trajectory.copy()
             #print(self.previous_trajectory)
        return obs, reward, done_dict, truncated_dict, info_dict

    
    def render(self, mode="rgb-array"):
        print(self.overcooked_env)

def set_partner(path):
    checkpoint_path = os.path.abspath(path)
    restored_trainer = PPO.from_checkpoint(checkpoint_path)
    module = restored_trainer.get_module("shared_policy")
    return module

def get_partner_action(module, obs):
    agent_ids = sorted(obs.keys())
    obs_list = [obs[agent_id] for agent_id in agent_ids]
    
    # RLModule을 사용해 행동 추론
    module_input = {
        "obs": torch.from_numpy(np.stack(obs_list))
    }
    action_tensors = module.forward_inference(module_input)
    
    # 로짓(logits)에서 가장 가능성 높은 행동을 선택 (argmax)
    logits = action_tensors['action_dist_inputs']
    probs = F.softmax(logits, dim=1)

    actions_tensor = torch.multinomial(probs, num_samples=1).squeeze(1)
    actions_np = actions_tensor.numpy()

    # 추론 결과를 action_dict 형태로 변환
    action_dict = {agent_id: action for agent_id, action in zip(agent_ids, actions_np)}
    return action_dict

class FCP_Rllib_for_visualization(gym.Env):
    #학습할 모델이 0번, 학습된 모델은 1번
    def __init__(self, env_config=None):
        super().__init__()
        # ⭐️⭐️⭐️ 디버깅을 위한 핵심 print 문 ⭐️⭐️⭐️
        # RLlib으로부터 받은 env_config를 내부 multi-agent 환경에 전달합니다.
        self.multi_agent_env = Rllib_multi_agent(env_config)
        #self.horizon = env_config.get("horizon", 400)
        self.active_agent_id = "agent_0"
        self.partner_agent_id = "agent_1"
        # 파트너 모델 로드는 클래스 내부에서 처리합니다.
        self.partner_paths ={
            1: "FCP_partner_agent/reward_595_24",
            }
        # 💡 [개선 2] __init__에서 모든 파트너 모듈을 미리 로드하여 딕셔너리에 저장
        self.partner_modules = {}
        print("="*30)
        print("파트너 모델들을 미리 불러오는 중입니다... (시간이 걸릴 수 있습니다)")
        
        print("모든 파트너 모델 로딩 완료!")
        print("="*30)
        self.observation_space = self.multi_agent_env.observation_space[self.active_agent_id]
        self.action_space = self.multi_agent_env.action_space[self.active_agent_id]
        self.total_reward = 0
        #random, greedy, mid, pro가 뽑힌 횟수
        self.count = [0] * 4
        self.iteration = 0
        

        self.num_of_dish = 0
        
        self.episode_reward = 0


    def _get_obs(self, idx = 0):
        return self.multi_agent_env._get_obs()
    #obs, reward 공유됨.
    def reset(self, seed=None, options=None):
        obs_dict, info_dict = self.multi_agent_env.reset()
        return obs_dict[self.active_agent_id], {}


    def step(self, action):

        if isinstance(action, dict):
            action_dict_to_step = action
        else:
            action_dict_to_step = {
                self.active_agent_id: action,
                self.partner_agent_id: 2,
            }
        #print(action_dict_to_step)
        #print(action_dict_to_step)
        obs_dict, reward_dict, done_dict, trunc_dict, info_dict = self.multi_agent_env.step(action_dict_to_step)
        #print(info_dict)
        # 4. 단일 에이전트 환경의 결과 형식에 맞게 값을 추출합니다.
        observation = obs_dict[self.active_agent_id]
        reward = reward_dict[self.active_agent_id]
        #print(reward)
        if reward > 0:
            print(self.num_of_dish)

            self.num_of_dish += 1
        #print(self.multi_agent_env.overcooked_env.mdp.state_string(self.multi_agent_env.overcooked_env.state))
        

        self.total_reward += reward
        terminated = done_dict["__all__"] # 에피소드 성공/실패 종료 여부
        truncated = trunc_dict["__all__"] # 시간 초과 종료 여부
        # ⭐️⭐️⭐️ 핵심 수정 부분 ⭐️⭐️⭐️
        if terminated or truncated:
            self.iteration +=1
        
        info = info_dict.get(self.active_agent_id, {})


        return observation, reward, terminated, truncated, info
    
    def get_num_of_dish(self):
        return self.num_of_dish


    def update_agent_pool(self, num):
        self.random_num = num
