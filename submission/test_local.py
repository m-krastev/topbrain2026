"""Run the submission without Docker, through the template's unchanged main.py,
on test/input/interf0 (MR) and interf1 (CT); outputs go to test/output_local/.
Set TOPBRAIN_MODEL_DIR to the model folder (default ./model).
    python test_local.py
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("TOPBRAIN_MODEL_DIR", str(HERE / "model"))
os.chdir(HERE)
sys.path.insert(0, str(HERE / "build"))  # predict_tta.py, postprocess.py (prepare.sh)

import main  # noqa: E402

for interf in ("interf0", "interf1"):
    main.INPUT_PATH = HERE / "test" / "input" / interf
    main.OUTPUT_PATH = HERE / "test" / "output_local" / interf
    print(f"=== {interf}")
    main.run()
