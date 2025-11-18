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

from human_aware_rl.imitation.behavior_cloning_tf2 import load_bc_model
from human_aware_rl.imitation.behavior_cloning_tf2 import _get_base_ae, BehaviorCloningPolicy
from human_aware_rl.static import CLEAN_2019_HUMAN_DATA_TRAIN
from human_aware_rl.imitation.behavior_cloning_tf2 import get_bc_params, train_bc_model
from human_aware_rl.rllib.rllib import RlLibAgent
params_to_override = {
    
    # this is the layouts where the training will happen
    "layouts": ["cramped_room"], 
    # this is the layout that the agents will be evaluated on
    # Most of the time they should be the same, but because of refactoring some old layouts have more than one name and they need to be adjusted accordingly
    "layout_name": "cramped_room", 
    "data_path": CLEAN_2019_HUMAN_DATA_TRAIN,
    "epochs": 10,
    "old_dynamics": True,
}

class OvercookedSingleAgentOneActionsGymEnv(gym.Env):
    def __init__(self, overcooked_env, agent_idx=0, model=None, mode=None):
        self.overcooked_env = overcooked_env
        self.agent_idx = agent_idx
        self.model = model
        self.mode = mode #bc, human model 구분
        if(self.mode == "bc"):
            self.train_bc_model()
            self.bc_model = self.load_bc_model()
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
        #print(self.overcooked_env.state.type)
        action2 = self.bc_model.action(state)[0]
        #action2 = stay_agent.action(self.overcooked_env.state)[0]
        if self.agent_idx == 0:         
            actions = [action2, action]  # 첫 번째 에이전트의 행동과 두 번째 에이전트는 상호작용
        else:     
            actions = [action, action2]  # 두 번째 에이전트의 행동과 첫 번째 에이전트는 상호작용
        # 2. 환경에 두 에이전트의 행동을 동시에 적용해서 다음 상태, 보상, 종료 여부, 추가 정보를 얻음
        next_state, reward, done, info = self.overcooked_env.step(actions)
        
        # 3. 환경의 현재 상태를 관찰(observation)으로 가져옴
        obs = self._get_obs()
        # 4. 만약 obs가 튜플 형태라면 numpy 배열로 바꾸고 1차원으로 평탄화함
        if isinstance(obs, tuple):
            obs = np.array(obs).flatten()

        # 5. 강화학습 알고리즘이 요구하는 형식대로
        #    (관찰, 보상, 종료, 에피소드 중단 여부, 추가 정보)를 리턴
        #print(1)
        return obs, reward, done, False, info


    def render(self, mode="rgb-array"):
        print(self.overcooked_env)

    def train_bc_model(self):
        bc_params = get_bc_params(**params_to_override)
        train_bc_model("tutorial_notebook_results/BC", bc_params, verbose = True)

    def load_bc_model(self):

        #this is the same path you used when training the BC agent
        bc_model_path = "tutorial_notebook_results/BC"
        bc_model, bc_params = load_bc_model(bc_model_path)
        
        bc_policy = BehaviorCloningPolicy.from_model(bc_model, bc_params, stochastic=True)
        # We need the featurization function that is specifically defined for BC agent
        # The easiest way to do it is to create a base environment from the configuration and extract the featurization function
        # The environment is also needed to do evaluation
        base_ae = _get_base_ae(bc_params)
        base_env = base_ae.env        
        bc_agent0 = RlLibAgent(bc_policy, 0, base_env.featurize_state_mdp)
        bc_agent0.reset()

        return bc_agent0


