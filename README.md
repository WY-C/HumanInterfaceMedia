## 실행방법(python 3.10)
1. requirements.txt 설치 (pip install -r requirements.txt)
2. python -m humaninterfacemedia.single_play_game
하면 실행됨

3. 본인 layout위치
(anaconda3\envs\name\lib\site-packages\overcooked_ai_py\data\layouts)
 찾아서 cramped_room
1XPXX
                O  2O
                X   X
                XDXSX
로 바꾸기

4. 그러면 에러가 하나 뜸.
(anaconda3\envs\name\lib\site-packages\overcooked_ai_py\mdp\overcooked_mdp.py) 들어가서
1596~1601 주석처리하기
   
5. overcooked_mdp.py begin_cooking함수를
    def begin_cooking(self):
        if not self.is_idle:
            raise ValueError("Cannot begin cooking this soup at this time")
        if len(self.ingredients) == 0:
            raise ValueError("Must add at least one ingredient to soup before you can begin cooking")
        #수정사항
        if len(self.ingredients) == 3:
            self._cooking_tick = 0
로 수정 (모든 재료가 준비되어야만 요리 시작)
---
## 2. 맵 별 KLM 설정
### Keystroke-Level Model(KLM) for overcooked

KLM(Keystroke-Level Model)은
<br>숙련된 사용자가 오류 없이 수행하는 작업 시간을 예측하기 위해
<br>작업을 일련의 원자적 행동(operator)으로 분해하고,
<br>각 operator에 평균 시간을 곱해 총 수행 시간을 계산하는 모델이다.

본 게임(오버쿡드 유사 플레이)의 조작 방식은
<br>키보드(WASD 방향키 + 스페이스바 인터랙션) 중심으로 이루어지므로
<br>전통 KLM의 모든 operator 중 K(입력), M(정신적 준비) 를 중심으로 사용하되
<br>마우스 기반 P, D 등은 제외하였다.

또한 본 게임 특성상
<br>“어디로 갈지 판단하는 과정”, “상호작용 순서 판단” 같은
<br>계획(Mental Preparation) 단계가 작업 흐름 중 필수적으로 나타나므로
<br>KLM 계산에 반드시 포함하였다.


### 본 게임에서 사용하는 KLM Operators 정의

전통 KLM의 K, M을 기반으로 하되
게임 조작의 실제 특성에 맞게 서브-오퍼레이터를 정의했다.

| **Operator** | **의미** | **본 게임에서의 예시** | **적용 이유**               |
| --- | --- | --- |-------------------------|
| **K_move** | 방향키 1회 입력 (전진/후진/좌우 이동) | W/A/S/D 1번 | 이동 동작의 최소 단위            |
| **K_turn** | 방향만 바꾸기 위한 입력 1회 | 방향만 바꿀 때 W/A/S/D | 맵 구조상 회전이 자주 발생         |
| **K_act** | 상호작용(스페이스바 1회) | 양파 집기, 냄비에 넣기, 접시 집기, 서빙 | 모든 인터랙션 동일 키 사용(스페이스바)  |
| **M** | 정신적 준비, 계획 (Mental preparation) | 재료 선택, 경로 변경 판단, 다음 행동 결정 | 오버쿡드류는 subtask 사이 사고가 필수 |



▶️ Key input(K) 시간

원본 표:
- Best: 0.08
- Good: 0.12
- Skilled: 0.20
- Average: 0.28
- Non-secretary: 0.40
- Random letters: 0.50
- Code typing: 0.75
- Worst: 1.20

게임 입력 특성은 “단순 방향 전환·이동 반복”이므로
Skilled 수준인 0.20초를 사용하는 것이 가장 적절하다.

▶️ Mental(M) 시간

원본 표: 1.35초
이 값을 그대로 사용한다.