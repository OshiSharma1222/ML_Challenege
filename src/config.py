import os

DATA_DIR = os.environ.get(
    "ER_DATA_DIR", "D:/6ab10eb3b23ba_student_resource/student_resource/dataset")
WORK_DIR = os.environ.get("ER_WORK_DIR", os.path.join(os.path.dirname(__file__), "..", "work"))
OUT_DIR = os.environ.get("ER_OUT_DIR", os.path.join(os.path.dirname(__file__), "..", "output"))
N_JOBS = int(os.environ.get("ER_JOBS", max(1, (os.cpu_count() or 4) - 2)))

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def raw_path(split, src):
    return os.path.join(DATA_DIR, split, f"{split}_source{src}.tsv")


def work(name):
    return os.path.join(WORK_DIR, name)
