import imageio
from humaninterfacemedia.env import FCP_Rllib_for_visualization
from my_env.Rllib_multi_agent import Rllib_multi_agent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import pygame
import sys
import os
from dotenv import load_dotenv
import numpy as np
import math  # ⭐️ 시간 계산(올림)을 위해 추가

from humaninterfacemedia.grid_util import load_layout_grid_from_name
from humaninterfacemedia.grid_util import sync_custom_layouts

from ray.tune.registry import register_env
import ray
ray.init()
load_dotenv()

# 맵 동기화
sync_custom_layouts()

# ... (환경 생성 및 등록 코드는 동일) ...
def single_env_creator(env_config):
    return FCP_Rllib_for_visualization(env_config, random_num=1)
register_env("FCP_Rllib", single_env_creator)

def env_creator(config):
    return Rllib_multi_agent(config)
register_env("Rllib_multi_agent", env_creator)
# ... (맵 레이아웃 정의 코드는 동일) ...
COUNTER = 'X'
POT = 'P'
ONION_DISPENSER = 'O'
DISH_DISPENSER = 'D'
SERVING_LOC = 'S'
EMPTY = ' '

# LAYOUT_GRID = [
#     [COUNTER, COUNTER,         POT,             COUNTER,         COUNTER],
#     [ONION_DISPENSER, EMPTY,   EMPTY,           EMPTY,           ONION_DISPENSER],
#     [COUNTER,         EMPTY,   EMPTY,           EMPTY,           COUNTER],
#     [COUNTER, DISH_DISPENSER,  COUNTER,         SERVING_LOC,     COUNTER]
# ]
LAYOUT_NAME = os.getenv("LAYOUT_NAME", "easy-2")
LAYOUT_GRID = load_layout_grid_from_name(LAYOUT_NAME)
print("map loaded: " + LAYOUT_NAME)

# --- 1. 초기화 (루프 시작 전) ---
pygame.init()
pygame.font.init() # ⭐️ 텍스트 렌더링을 위해 폰트 모듈 초기화

# ⭐️ 텍스트를 표시할 폰트 설정 (기본 폰트, 크기 50)
try:
    font = pygame.font.Font(None, 20)
except: # 폰트 로드 실패 시 기본값 사용
    font = pygame.font.SysFont('arial', 20)


visualizer = StateVisualizer()

mode = "user_input"
my_env = FCP_Rllib_for_visualization({"layout_name": LAYOUT_NAME})

initial_surface = visualizer.render_state(my_env.multi_agent_env.overcooked_env.state, grid=LAYOUT_GRID)
screen_width, screen_height = initial_surface.get_size()

clock = pygame.time.Clock()

frames = []
screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
pygame.display.set_caption("Overcooked AI Live")

running = True
obs, info = my_env.reset()

# --- ⭐️ 2. 타이머 설정 ---
game_duration_seconds = 5
game_duration_ms = game_duration_seconds * 1000  # 밀리초 단위로 변환

flag = True
flag1 = False

#todo
number = input("사용자 번호를 입력하세요")
#엔터 누르고 게임 시작.
while True:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:        # key press event?
            if event.key == pygame.K_SPACE:     # space bar?
                flag1 = True
                break

    if flag1:
        break

start_time = pygame.time.get_ticks()  # 게임 시작 시간 기록
while running:
    player_action = 4
    
    # --- 이벤트 처리 ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.VIDEORESIZE:
            screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
        if event.type == pygame.KEYDOWN:

            # 경과시간
            elapsed_ms = pygame.time.get_ticks() - start_time
            if event.key == pygame.K_LEFT:
                player_action = 3
                #print(f"[{elapsed_ms:5d} ms] K_move : LEFT")
            elif event.key == pygame.K_RIGHT:
                player_action = 2
                #print(f"[{elapsed_ms:5d} ms] K_move : RIGHT")
            elif event.key == pygame.K_UP:
                player_action = 0
                #print(f"[{elapsed_ms:5d} ms] K_move : UP")
            elif event.key == pygame.K_DOWN:
                player_action = 1
                #print(f"[{elapsed_ms:5d} ms] K_move : DOWN")
            elif event.key == pygame.K_SPACE:
                player_action = 5
                #print(f"[{elapsed_ms:5d} ms] K_act  : SPACE (interact)")
            # else: player_action = 4 (기본값)
    if flag:
        flag = False
        action_dict = {
            "agent_0": 1,
            "agent_1": player_action
        }
    else:
        flag = True
        action_dict = {
            "agent_0": 1,
            "agent_1": player_action
        }

    
    # ⭐️ [수정] step()의 반환값을 모두 받습니다.
    obs, reward, terminated, truncated, info = my_env.step(action_dict)

    # --- ⭐️ 3. 시간 계산 ---
    elapsed_time_ms = pygame.time.get_ticks() - start_time
    remaining_ms = game_duration_ms - elapsed_time_ms
    
    # 남은 시간을 초 단위로 계산 (올림)
    remaining_seconds = max(0, math.ceil(remaining_ms / 1000))

    # --- 화면 렌더링 ---
    # 1. 게임 상태 그리기
    state_surface = visualizer.render_state(my_env.multi_agent_env.overcooked_env.state, grid=LAYOUT_GRID)
    screen.blit(state_surface, (0, 0))

    # ⭐️ 4. 텍스트 렌더링 (남은 시간)
    # 텍스트 Surface 생성 (내용, 안티앨리어싱, 색상)
    text_surface = font.render(f"Time Left: {remaining_seconds}", True, (255, 255, 255))
    
    # 텍스트를 화면에 그리기 (위치: 좌상단 10, 10)
    screen.blit(text_surface, (10, 10))

    # 3. 화면 전체 업데이트
    pygame.display.flip()

    # --- 프레임 캡처 ---
    frame_data = pygame.surfarray.array3d(screen)
    frame_data = np.rot90(frame_data)
    frame_data = np.flipud(frame_data)
    frames.append(frame_data)

    # --- FPS 제어 ---
    clock.tick(50)

    # ⭐️ 5. 종료 조건 확인
    # 시간이 다 되었거나, 게임이 종료(terminated)되거나, 시간이 초과(truncated)되면 루프 종료
    if remaining_ms <= 0 or terminated or truncated:
        running = False
        
print("Number : ", number, " Layout : ", LAYOUT_NAME, " Score :", 20 * my_env.get_num_of_dish())
text = f"Number : {number}, Layout : {LAYOUT_NAME}, Score : {20 * my_env.get_num_of_dish()}\n"

with open("result.txt", "a", encoding="utf-8") as f:
    f.write(text)

if frames:
    print("Saving GIF...")
    
    # 🚀 핵심: 프레임을 솎아냅니다 (Slicing)
    # frames[::3] -> 3장 중 1장만 저장 (3배속 효과)
    # frames[::5] -> 5장 중 1장만 저장 (5배속 효과 -> 더 빠름)
    #fast_frames = frames[::5] 

    # duration 대신 fps=60을 쓰면 가장 부드럽고 빠른 속도로 맞춰줍니다.
    imageio.mimsave(f'GIF/{number}_{LAYOUT_NAME}.gif', frames, fps=50, loop=0)
    
    print("GIF saved successfully!")
else:
    print("No frames were recorded.")


pygame.quit()
sys.exit()
