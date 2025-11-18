실행방법(python 3.10)
1. requirements.txt 설치 (pip install -r requirements.txt)
2. python -m humaninterfacemedia.single_play_game
하면 실행됨

5. 본인 layout위치
(anaconda3\envs\name\lib\site-packages\overcooked_ai_py\data\layouts)
 찾아서 cramped_room
1XPXX
                O  2O
                X   X
                XDXSX
로 바꾸기

6. 그러면 에러가 하나 뜸.
(anaconda3\envs\name\lib\site-packages\overcooked_ai_py\mdp\overcooked_mdp.py) 들어가서
1596~1601 주석처리하기
   
