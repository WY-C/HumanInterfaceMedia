#07 02 reward shaping 으로 학습 140? 180? reward 확인환료
#todo
#PPO모델과 humanmodel을 사용하여 Overcooked 환경에서 단일 에이전트가 상호작용하는 Gym 환경 구현.
import gymnasium as gym
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv

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
    def __init__(self, config = None, reward_shaping = False):
        #이후 config에 layout name, horizon 등등을 넣어야함.
        super().__init__()
        config = config or {}
        layout_name = config.get("layout_name", "asymmetric_advantages")
        horizon = config.get("horizon", 400)
        self.reward_shaping = config.get("reward_shaping", "False")
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
        #     action_dict['agent_0'] = 4
        #     action_dict['agent_1'] = 4
        #     print(1)
            
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

    def get_map(self):
        return self.overcooked_env.mdp.state_string(self.overcooked_env.state)


