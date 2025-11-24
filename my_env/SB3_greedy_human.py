#07 02 reward shaping 으로 학습 140? 180? reward 확인환료
#todo
#PPO모델과 humanmodel을 사용하여 Overcooked 환경에서 단일 에이전트가 상호작용하는 Gym 환경 구현.



import gymnasium as gym
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import GreedyHumanModel
#device = torch.device('cpu')

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

mdp = OvercookedGridworld.from_layout_name("cramped_room") 
mlam = MediumLevelActionManager.from_pickle_or_compute(mdp, mlam_params=NO_COUNTERS_PARAMS, force_compute=False)
humanmodel = GreedyHumanModel(mlam)


class SB3_greedy_human(gym.Env):
    def __init__(self, overcooked_env, agent_idx=0, model=None):
        self.overcooked_env = overcooked_env
        self.agent_idx = agent_idx
        self.model = model
        self.action_space = gym.spaces.Discrete(len(ACTION_MAP))
        obs = self._get_obs()
        if isinstance(obs, tuple):
            obs = np.array(obs).flatten()
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=obs.shape, dtype=np.int64
        )

    def _get_obs(self):
        state = self.overcooked_env.state
        #print(state.type)
        #print(self.overcooked_env.lossless_state_encoding_mdp(state))
        #self.overcooked_env.display_states(state)
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
        state = self.overcooked_env.state
        #print(action)
        action = ACTION_MAP[action]
        action2 = humanmodel.action(state)[0]
        #action2 = stay_agent.action(self.overcooked_env.state)[0]
        if self.agent_idx == 0:
            actions = [action2, action]  # 첫 번째 에이전트의 행동과 두 번째 에이전트는 상호작용
            #print(action, action2)
        else:     
            actions = [action, action2]  # 두 번째 에이전트의 행동과 첫 번째 에이전트는 상호작용
            #print(action, action2)
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




