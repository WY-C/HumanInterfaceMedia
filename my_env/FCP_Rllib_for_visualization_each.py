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

mdp = OvercookedGridworld.from_layout_name("cramped_room") 
mlam = MediumLevelActionManager.from_pickle_or_compute(mdp, mlam_params=NO_COUNTERS_PARAMS, force_compute=False)
humanmodel = GreedyHumanModel(mlam)

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
        layout_name = config.get("layout_name", "cramped_room")
        horizon = config.get("horizon", 540)
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

        if rewards > 0:
            #print(rewards)
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
    def __init__(self, env_config=None, random_num = None):
        super().__init__()
        # ⭐️⭐️⭐️ 디버깅을 위한 핵심 print 문 ⭐️⭐️⭐️
        # RLlib으로부터 받은 env_config를 내부 multi-agent 환경에 전달합니다.
        self.multi_agent_env = Rllib_multi_agent(env_config)
        #self.horizon = env_config.get("horizon", 400)
        self.active_agent_id = "agent_0"
        self.partner_agent_id = "agent_1"
        self.num_of_dish = 0
        # 파트너 모델 로드는 클래스 내부에서 처리합니다.
        self.partner_paths ={
            0: "user_input",
            1: "SP_agent_for_FCP/testing/0_best_checkpoint",
            2: "SP_agent_for_FCP/testing/110_middle_checkpoint",
            3: "SP_agent_for_FCP/testing/0_final_checkpoint",
            4: "SP_agent_for_FCP/testing/110_random_initalized_checkpoint",
            5: "SP_agent_for_FCP/testing/0_middle_checkpoint",
            6: "SP_agent_for_FCP/testing/111_best_checkpoint",
            7: "SP_agent_for_FCP/testing/0_random_initalized_checkpoint",
            8: "SP_agent_for_FCP/testing/111_final_checkpoint",
            9: "SP_agent_for_FCP/testing/100_best_checkpoint",
            10: "SP_agent_for_FCP/testing/111_middle_checkpoint",
            11: "SP_agent_for_FCP/testing/100_final_checkpoint",
            12: "SP_agent_for_FCP/testing/111_random_initalized_checkpoint",
            13: "SP_agent_for_FCP/testing/100_middle_checkpoint",
            14: "SP_agent_for_FCP/testing/112_random_initalized_checkpoint",
            15: "SP_agent_for_FCP/testing/100_random_initalized_checkpoint",
            16: "SP_agent_for_FCP/testing/11_best_checkpoint",
            17: "SP_agent_for_FCP/testing/101_best_checkpoint",
            18: "SP_agent_for_FCP/testing/11_final_checkpoint",
            19: "SP_agent_for_FCP/testing/101_final_checkpoint",
            20: "SP_agent_for_FCP/testing/11_middle_checkpoint",
            21: "SP_agent_for_FCP/testing/101_middle_checkpoint",
            22: "SP_agent_for_FCP/testing/11_random_initalized_checkpoint",
            23: "SP_agent_for_FCP/testing/101_random_initalized_checkpoint",
            24: "SP_agent_for_FCP/testing/12_best_checkpoint",
            25: "SP_agent_for_FCP/testing/102_best_checkpoint",
            26: "SP_agent_for_FCP/testing/12_middle_checkpoint",
            27: "SP_agent_for_FCP/testing/102_final_checkpoint",
            28: "SP_agent_for_FCP/testing/12_random_initalized_checkpoint",
            29: "SP_agent_for_FCP/testing/102_middle_checkpoint",
            30: "SP_agent_for_FCP/testing/1_best_checkpoint",
            31: "SP_agent_for_FCP/testing/102_random_initalized_checkpoint",
            32: "SP_agent_for_FCP/testing/1_final_checkpoint",
            33: "SP_agent_for_FCP/testing/103_best_checkpoint",
            34: "SP_agent_for_FCP/testing/1_middle_checkpoint",
            35: "SP_agent_for_FCP/testing/103_final_checkpoint",
            36: "SP_agent_for_FCP/testing/1_random_initalized_checkpoint",
            37: "SP_agent_for_FCP/testing/103_middle_checkpoint",
            38: "SP_agent_for_FCP/testing/2_best_checkpoint",
            39: "SP_agent_for_FCP/testing/103_random_initalized_checkpoint",
            40: "SP_agent_for_FCP/testing/2_final_checkpoint",
            41: "SP_agent_for_FCP/testing/104_best_checkpoint",
            42: "SP_agent_for_FCP/testing/2_middle_checkpoint",
            43: "SP_agent_for_FCP/testing/104_final_checkpoint",
            44: "SP_agent_for_FCP/testing/2_random_initalized_checkpoint",
            45: "SP_agent_for_FCP/testing/104_middle_checkpoint",
            46: "SP_agent_for_FCP/testing/3_best_checkpoint",
            47: "SP_agent_for_FCP/testing/104_random_initalized_checkpoint",
            48: "SP_agent_for_FCP/testing/3_final_checkpoint",
            49: "SP_agent_for_FCP/testing/105_best_checkpoint",
            50: "SP_agent_for_FCP/testing/3_middle_checkpoint",
            51: "SP_agent_for_FCP/testing/105_final_checkpoint",
            52: "SP_agent_for_FCP/testing/3_random_initalized_checkpoint",
            53: "SP_agent_for_FCP/testing/105_middle_checkpoint",
            54: "SP_agent_for_FCP/testing/4_best_checkpoint",
            55: "SP_agent_for_FCP/testing/105_random_initalized_checkpoint",
            56: "SP_agent_for_FCP/testing/4_final_checkpoint",
            57: "SP_agent_for_FCP/testing/106_best_checkpoint",
            58: "SP_agent_for_FCP/testing/4_middle_checkpoint",
            59: "SP_agent_for_FCP/testing/106_final_checkpoint",
            60: "SP_agent_for_FCP/testing/5_best_checkpoint",
            61: "SP_agent_for_FCP/testing/106_middle_checkpoint",
            62: "SP_agent_for_FCP/testing/5_final_checkpoint",
            63: "SP_agent_for_FCP/testing/107_best_checkpoint",
            64: "SP_agent_for_FCP/testing/5_middle_checkpoint",
            65: "SP_agent_for_FCP/testing/107_final_checkpoint",
            66: "SP_agent_for_FCP/testing/5_random_initalized_checkpoint",
            67: "SP_agent_for_FCP/testing/107_middle_checkpoint",
            68: "SP_agent_for_FCP/testing/6_best_checkpoint",
            69: "SP_agent_for_FCP/testing/107_random_initalized_checkpoint",
            70: "SP_agent_for_FCP/testing/6_final_checkpoint",
            71: "SP_agent_for_FCP/testing/108_best_checkpoint",
            72: "SP_agent_for_FCP/testing/6_middle_checkpoint",
            73: "SP_agent_for_FCP/testing/108_final_checkpoint",
            74: "SP_agent_for_FCP/testing/6_random_initalized_checkpoint",
            75: "SP_agent_for_FCP/testing/108_middle_checkpoint",
            76: "SP_agent_for_FCP/testing/7_best_checkpoint",
            77: "SP_agent_for_FCP/testing/108_random_initalized_checkpoint",
            78: "SP_agent_for_FCP/testing/7_final_checkpoint",
            79: "SP_agent_for_FCP/testing/109_best_checkpoint",
            80: "SP_agent_for_FCP/testing/7_middle_checkpoint",
            81: "SP_agent_for_FCP/testing/109_final_checkpoint",
            82: "SP_agent_for_FCP/testing/7_random_initalized_checkpoint",
            83: "SP_agent_for_FCP/testing/109_middle_checkpoint",
            84: "SP_agent_for_FCP/testing/8_best_checkpoint",
            85: "SP_agent_for_FCP/testing/109_random_initalized_checkpoint",
            86: "SP_agent_for_FCP/testing/8_final_checkpoint",
            87: "SP_agent_for_FCP/testing/10_best_checkpoint",
            88: "SP_agent_for_FCP/testing/8_middle_checkpoint",
            89: "SP_agent_for_FCP/testing/10_final_checkpoint",
            90: "SP_agent_for_FCP/testing/8_random_initalized_checkpoint",
            91: "SP_agent_for_FCP/testing/10_middle_checkpoint",
            92: "SP_agent_for_FCP/testing/9_best_checkpoint",
            93: "SP_agent_for_FCP/testing/10_random_initalized_checkpoint",
            94: "SP_agent_for_FCP/testing/9_final_checkpoint",
            95: "SP_agent_for_FCP/testing/110_best_checkpoint",
            96: "SP_agent_for_FCP/testing/9_middle_checkpoint",
            97: "SP_agent_for_FCP/testing/110_final_checkpoint",
            98: "SP_agent_for_FCP/testing/9_random_initalized_checkpoint",
            99: "SP_agent_for_FCP/testing/111_best_checkpoint",
            100: "SP_agent_for_FCP/testing/111_final_checkpoint"
        }
            
        # 💡 [개선 2] __init__에서 모든 파트너 모듈을 미리 로드하여 딕셔너리에 저장
        self.partner_modules = {}
        print("="*30)
        print("파트너 모델들을 미리 불러오는 중입니다... (시간이 걸릴 수 있습니다)")
        
        # 'humanmodel'이 전역적으로 정의되어 있다고 가정
        # from overcooked_ai_py.agents.agent import GreedyHumanModel
        # mdp = OvercookedGridworld.from_layout_name("cramped_room")
        # mlam = MediumLevelActionManager.from_pickle_or_compute(mdp, ...)
        # humanmodel = GreedyHumanModel(mlam)
        self.random_num = random_num
        #print(self.random_num)

        for key, path in self.partner_paths.items():
            print(f"  - 로딩 중: 파트너 {key} ({path})")

            if key != self.random_num:
                continue
            if path == "random_agent":
                self.partner_modules[key] = "random_agent"
            elif path == "greedy_human_agent":
                self.partner_modules[key] = humanmodel
            elif path == "user_input":
                self.partner_modules[key] = "user_input"
            else:
                # set_partner 함수를 여기서 단 한 번만 호출합니다.
                self.partner_modules[key] = set_partner(path)
        
        print("모든 파트너 모델 로딩 완료!")
        print("="*30)
        self.observation_space = self.multi_agent_env.observation_space[self.active_agent_id]
        self.action_space = self.multi_agent_env.action_space[self.active_agent_id]
        self.total_reward = 0
        #random, greedy, mid, pro가 뽑힌 횟수
        self.count = [0] * 4
        self.iteration = 0
        
        #random, mid, pro의 reward각각의 리워드 총합
        self.each_reward = [0] * 4
        #0 random, 1 mid, 2 pro
        #reward는 count로 나눠서 측정.
        #dish는 eval용으로 사용함.


        
        self.episode_reward = 0


    def _get_obs(self, idx = 0):
        return self.multi_agent_env._get_obs()
    #obs, reward 공유됨.
    def reset(self, seed=None, options=None):
        self.episode_reward = 0
        self.num_of_dish = 0
        self.total_reward = 0

        # 💡 [개선 3] 매번 로드하는 대신, 미리 로드된 모듈에서 무작위로 선택
        random_key = self.random_num#random.randint(1, 12)
        # 딕셔너리에서 바로 모듈을 가져오므로 매우 빠릅니다.
        self.current_partner_module = self.partner_modules[self.random_num]
        
        # 로깅을 위한 파트너 타입 결정
        path = self.partner_paths[random_key]
        if path == "random_agent":
            self.count[0] +=1
            self.current_partner_type = "random"
        elif path == "greedy_human_agent":
            self.count[1] +=1
            self.current_partner_type = "greedy_human"
        elif path == "user_input":
            self.current_partner_type = "user_input"
        elif random_key <= 4 :
            self.count[3] +=1
            self.current_partner_type = "pro"
        else:
            self.count[2] +=1
            self.current_partner_type = "mid"
        
        obs_dict, info_dict = self.multi_agent_env.reset()
        return obs_dict[self.active_agent_id], {}


    def step(self, action):
        #print(self.current_partner_type)
        # 💡 [개선 4] 파트너 행동 결정 로직이 더 깔끔해짐
        if self.current_partner_type == "random":
            partner_action = self.action_space.sample()
        elif self.current_partner_type == "greedy_human":
            state = self.multi_agent_env.overcooked_env.state

            partner_joint_action = self.current_partner_module.action(state)
            #print(partner_joint_action)
            partner_action = REVERSE_ACTION_MAP[partner_joint_action[0]]
        elif self.current_partner_type == "user_input":
            pass   
        else: # pro or mid
            obs_for_partner = self.multi_agent_env._get_obs()
            partner_action_dict = get_partner_action(self.current_partner_module, obs_for_partner)
            partner_action = partner_action_dict[self.partner_agent_id]

        
        if isinstance(action, dict):
            action_dict_to_step = action
        else:
            action_dict_to_step = {
                self.active_agent_id: action,
                self.partner_agent_id: partner_action,
            }
        #print(action_dict_to_step)
        #print(action_dict_to_step)
        obs_dict, reward_dict, done_dict, trunc_dict, info_dict = self.multi_agent_env.step(action_dict_to_step)
        #print(info_dict)
        # 4. 단일 에이전트 환경의 결과 형식에 맞게 값을 추출합니다.
        observation = obs_dict[self.active_agent_id]
        reward = reward_dict[self.active_agent_id]
                #print(reward)
        if reward >= 20:
            self.num_of_dish += 1
        reward -= 0.2
        self.episode_reward += reward
        if self.current_partner_type == "random":
            self.each_reward[0] += reward
        elif self.current_partner_type == "greedy_human":
            self.each_reward[1] += reward
        elif self.current_partner_type =="pro":
            self.each_reward[3] += reward
        elif self.current_partner_type == "mid":
            self.each_reward[2] += reward

        print(self.multi_agent_env.overcooked_env.mdp.state_string(self.multi_agent_env.overcooked_env.state))
        

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
