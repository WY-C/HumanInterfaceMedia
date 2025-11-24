import imageio
from humaninterfacemedia.env import FCP_Rllib_for_visualization
from humaninterfacemedia.env import Rllib_multi_agent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import pygame
import sys
import os
from dotenv import load_dotenv
import numpy as np
import math

from humaninterfacemedia.grid_util import load_layout_grid_from_name
from humaninterfacemedia.grid_util import sync_custom_layouts

from ray.tune.registry import register_env
import ray
ray.init()
load_dotenv()

# 맵 동기화
sync_custom_layouts()

def single_env_creator(env_config):
    return FCP_Rllib_for_visualization(env_config, random_num=1)
register_env("FCP_Rllib", single_env_creator)

def env_creator(config):
    return Rllib_multi_agent(config)
register_env("Rllib_multi_agent", env_creator)

COUNTER = 'X'
POT = 'P'
ONION_DISPENSER = 'O'
DISH_DISPENSER = 'D'
SERVING_LOC = 'S'
EMPTY = ' '

#실험시 수정할 변수들
LAYOUT_NAME = os.getenv("LAYOUT_NAME", "easy2_1")
tick = 60
LAYOUT_GRID = load_layout_grid_from_name(LAYOUT_NAME)
print("map loaded: " + LAYOUT_NAME)

# --- 1. 초기화 (루프 시작 전) ---
pygame.init()
pygame.font.init()

try:
    font = pygame.font.Font(None, 20)
    symbol_font = pygame.font.Font(None, 20) 
except:
    font = pygame.font.SysFont('arial', 20)
    symbol_font = pygame.font.SysFont('arial', 50)

visualizer = StateVisualizer()

mode = "user_input"
my_env = FCP_Rllib_for_visualization({"layout_name": LAYOUT_NAME})

initial_surface = visualizer.render_state(my_env.multi_agent_env.overcooked_env.state, grid=LAYOUT_GRID)
screen_width, screen_height = initial_surface.get_size()

clock = pygame.time.Clock()

frames = []
user_actions = []

screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
pygame.display.set_caption("Overcooked AI Live")

running = True
obs, info = my_env.reset()

#시간
game_duration_seconds = 60
game_duration_ms = game_duration_seconds * 1000

flag = True
flag1 = False

# 접시 제출 감지 및 효과 변수
prev_dish_count = 0        
last_served_time = 0       
effect_duration = 1000     

#todo
number = input("사용자 번호를 입력하세요")

#엔터 누르고 게임 시작.
while True:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
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
            elapsed_ms = pygame.time.get_ticks() - start_time
            logged_key = None 

            if event.key == pygame.K_LEFT:
                player_action = 3
                logged_key = "LEFT"
            elif event.key == pygame.K_RIGHT:
                player_action = 2
                logged_key = "RIGHT"
            elif event.key == pygame.K_UP:
                player_action = 0
                logged_key = "UP"
            elif event.key == pygame.K_DOWN:
                player_action = 1
                logged_key = "DOWN"
            elif event.key == pygame.K_SPACE:
                player_action = 5
                logged_key = "SPACE"
            
            if logged_key:
                user_actions.append(f"{elapsed_ms},{logged_key}")

    if flag:
        flag = False
        action_dict = {"agent_0": 1, "agent_1": player_action}
    else:
        flag = True
        action_dict = {"agent_0": 2, "agent_1": player_action}

    obs, reward, terminated, truncated, info = my_env.step(action_dict)

    # --- 접시 제출 감지 로직 ---
    current_dish_count = my_env.get_num_of_dish()
    if current_dish_count > prev_dish_count:
        # 1. 시각 효과용 시간 기록
        last_served_time = pygame.time.get_ticks()
        
        # 2. ⭐️ [추가] 로그 파일용 기록 (현재 경과 시간 + 이벤트명)
        log_time = pygame.time.get_ticks() - start_time
        user_actions.append(f"{log_time},SERVED")
        
        #print(f"Dish Served! Total: {current_dish_count+1}")
    
    prev_dish_count = current_dish_count 

    # --- 3. 시간 계산 ---
    elapsed_time_ms = pygame.time.get_ticks() - start_time
    remaining_ms = game_duration_ms - elapsed_time_ms
    remaining_seconds = max(0, math.ceil(remaining_ms / 1000))

    # --- 화면 렌더링 ---
    state_surface = visualizer.render_state(my_env.multi_agent_env.overcooked_env.state, grid=LAYOUT_GRID)
    screen.blit(state_surface, (0, 0))

    text_surface = font.render(f"Time Left: {remaining_seconds}", True, (255, 255, 255))
    screen.blit(text_surface, (10, 10))

    pygame.display.flip()

    # --- 프레임 캡처 ---
    frame_data = pygame.surfarray.array3d(screen)
    frame_data = np.rot90(frame_data)
    frame_data = np.flipud(frame_data)
    frames.append(frame_data)

    clock.tick(tick)

    # 5. 종료 조건 확인
    if remaining_ms <= 0 or terminated or truncated:
        running = False
        
print("Number : ", number, " Layout : ", LAYOUT_NAME, " Score :", 20 * my_env.get_num_of_dish())
text = f"Number : {number}, Layout : {LAYOUT_NAME}, Score : {20 * my_env.get_num_of_dish()}\n"

os.makedirs("GIF", exist_ok=True)
os.makedirs("input", exist_ok=True)

with open("result.txt", "a", encoding="utf-8") as f:
    f.write(text)

# 로그 파일 저장
if user_actions:
    input_filename = f"input/{number}_{LAYOUT_NAME}_input.txt"
    with open(input_filename, "w", encoding="utf-8") as f:
        f.write("Timestamp(ms),Event\n") # 헤더를 Key -> Event로 살짝 변경 (키 뿐만 아니라 이벤트도 들어가므로)
        for action in user_actions:
            f.write(action + "\n")
    print("Input logs saved successfully!")

if frames:
    print("Saving GIF...")
    save_fps = tick
    if tick == 60:
        frames = frames[::3]
        save_fps = tick // 3 

    imageio.mimsave(f'GIF/{number}_{LAYOUT_NAME}.gif', frames, fps=save_fps, loop=0)
    print(f"GIF saved successfully! (FPS: {save_fps})")
else:
    print("No frames were recorded.")

pygame.quit()
sys.exit()