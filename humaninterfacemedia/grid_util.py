import textwrap
import importlib.resources as pkg_resources
import overcooked_ai_py.data.layouts as layouts_pkg
import ast
from pathlib import Path
import shutil


def load_layout_grid_from_name(layout_name: str):
    """
    layout_name (예: "cramped_room")만 넣으면
    overcooked_ai_py/data/layouts/<layout_name>.layout 파일을 찾아
    grid를 LAYOUT_GRID 형태로 파싱하여 반환.
    """
    filename = f"{layout_name}.layout"

    with pkg_resources.open_text(layouts_pkg, filename) as f:
        layout = ast.literal_eval(f.read())

    grid_str = layout["grid"]
    grid_str = textwrap.dedent(grid_str)

    raw_lines = [line for line in grid_str.splitlines() if line.strip()]
    if not raw_lines:
        return []

    # 첫 줄(예: "XXPXX") 기준으로 실제 width 계산
    first_core = raw_lines[0].rstrip("\n")
    width = len(first_core.strip())   # "XXPXX" -> 5

    rows = []
    for line in raw_lines:
        core = line.rstrip("\n")
        # 오른쪽에서 width개만 자름 (공백 포함)
        core = core[-width:]

        # 에이전트 표시 '1','2'는 시각화용 grid에서는 빈칸으로 처리
        core = core.replace('1', ' ').replace('2', ' ')

        rows.append(list(core))

    return rows

def sync_custom_layouts(custom_dir: str | Path | None = None):
    """
    HumanInterfaceMedia/map/*.layout 파일들을
    overcooked_ai_py/data/layouts/ 로 복사한다.

    """

    # 1) 소스 폴더 결정
    if custom_dir is None:
        # grid_util.py가 있는 디렉터리 기준 ./map 폴더 사용
        here = Path(__file__).resolve().parent
        src_dir = here / "map"
    else:
        src_dir = Path(custom_dir)

    if not src_dir.exists():
        print(f"[sync_custom_layouts] source dir not found: {src_dir}")
        return

    # 2) overcooked_ai_py의 layouts 디렉토리 찾기
    pkg_paths = list(layouts_pkg.__path__)
    if not pkg_paths:
        print("[sync_custom_layouts] could not resolve layouts package path")
        return
    dest_dir = Path(pkg_paths[0]).resolve()

    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0

    # 3) *.layout 파일 순회하며 복사 (항상 덮어쓰기)
    for src_path in src_dir.glob("*.layout"):
        dest_path = dest_dir / src_path.name

        shutil.copy2(src_path, dest_path)
        print(f"[copy/overwrite] {src_path.name} -> {dest_path}")
        copied += 1

    print(
        f"[sync_custom_layouts] done. "
        f"copied={copied}, src={src_dir}, dest={dest_dir}"
    )