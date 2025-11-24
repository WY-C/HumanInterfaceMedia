#evaluate 모드 설정하기
#info에 pro, 







import gymnasium as gym
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPO
import os
import torch
ACTION_MAP = {
    0: (0, -1),   # NORTH
    1: (0, 1),    # SOUTH
    2: (1, 0),    # EAST
    3: (-1, 0),   # WEST
    4: (0, 0),    # STAY
    5: "interact" # INTERACT
}
class Rllib_multi_agent(MultiAgentEnv):
    #agent1, agent2
    def __init__(self, config = None):
        #이후 config에 layout name, horizon 등등을 넣어야함.
        super().__init__()
        config = config or {}
        layout_name = config.get("layout_name", "cramped_room")
        horizon = config.get("horizon", 400)
        self.reward_shaping = config.get("reward_shaping", "True")
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


def set_partner(path = "checkpoint/rewardshaping_checkpoints/reward_599_56"):
    #path = 'checkpoint/rewardshaping_checkpoints/reward_548_72'
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
    actions_tensor = torch.argmax(logits, dim=1)
    actions_np = actions_tensor.numpy()
    
    # 추론 결과를 action_dict 형태로 변환
    action_dict = {agent_id: action for agent_id, action in zip(agent_ids, actions_np)}
    return action_dict

class Rllib_single_agent(gym.Env):
    #학습할 모델이 0번, 학습된 모델은 1번
    def __init__(self, env_config=None):
        super().__init__()
        
        # RLlib으로부터 받은 env_config를 내부 multi-agent 환경에 전달합니다.
        self.multi_agent_env = Rllib_multi_agent(env_config)
        
        self.active_agent_id = "agent_0"
        self.partner_agent_id = "agent_1"
        
        # 파트너 모델 로드는 클래스 내부에서 처리합니다.
        self.partner_module = set_partner("FCP_partner_agent/reward_610_64")
        
        self.observation_space = self.multi_agent_env.observation_space[self.active_agent_id]
        self.action_space = self.multi_agent_env.action_space[self.active_agent_id]

    def _get_obs(self, idx = 0):
        return self.multi_agent_env._get_obs()
    
    def _set_partner(self, env_config):
        # env_config에서 체크포인트 경로를 가져옵니다.
        path = env_config.get("partner_checkpoint_path", 'checkpoint/rewardshaping_checkpoints/reward_599_56')
        checkpoint_path = os.path.abspath(path)
        
        # 참고: PPO.from_checkpoint는 불안정할 수 있으므로, 이전 대화에서 논의한
        # config.build() -> algo.restore() 방식이 더 안정적일 수 있습니다.
        # 여기서는 기존 로직을 유지합니다.
        restored_trainer = PPO.from_checkpoint(checkpoint_path)
        module = restored_trainer.get_module("shared_policy")
        return module
    #obs, reward 공유됨.
    def reset(self, seed=None, options=None):
        obs_dict, info_dict = self.multi_agent_env.reset()
        active_agent_obs = obs_dict[self.active_agent_id]
        return active_agent_obs, {}

    def step(self, action):

        partner_action_dict = get_partner_action(self.partner_module, self._get_obs())
        partner_action = partner_action_dict[self.partner_agent_id]
        
        #action = ACTION_MAP[action]
        action_dict_to_step = {
            self.active_agent_id: action,
            self.partner_agent_id: partner_action,
        }
        #print(action_dict_to_step)
        obs_dict, reward_dict, done_dict, trunc_dict, info_dict = self.multi_agent_env.step(action_dict_to_step)

        # 4. 단일 에이전트 환경의 결과 형식에 맞게 값을 추출합니다.
        observation = obs_dict[self.active_agent_id]
        reward = reward_dict[self.active_agent_id]
        terminated = done_dict["__all__"] # 에피소드 성공/실패 종료 여부
        truncated = trunc_dict["__all__"] # 시간 초과 종료 여부
        info = info_dict.get(self.active_agent_id, {}) # 추가 정보

        # Gymnasium API 표준에 따라 5개의 값을 튜플로 반환합니다.
        return observation, reward, terminated, truncated, info
    #     action_dict = {
    #         self.active_agent_id: action,
    #         self.partner_agent_id: partner_action,
    #     }

    #     # 합쳐진 행동 딕셔너리로 원본 환경의 스텝을 진행합니다.
    #     obs_dict, reward_dict, done_dict, info_dict = self.multi_agent_env.step(action_dict)

    #     # 파트너의 다음 observation을 업데이트합니다.
    #     self.partner_obs = obs_dict[self.partner_agent_id]

    #     # 주인공 에이전트의 결과만 반환합니다.
    #     obs = obs_dict[self.active_agent_id]
    #     reward = reward_dict[self.active_agent_id]
    #     done = done_dict["__all__"] # 전체 게임 종료 여부
    #     info = info_dict.get(self.active_agent_id, {})

    #     return obs, reward, done, info

    
    # def render(self, mode="rgb-array"):
    #     print(self.overcooked_env)




