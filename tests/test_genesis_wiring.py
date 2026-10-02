"""genesis_flight.py의 배선 시험: Genesis 대역으로 돌린 결과가 numpy 6자유도와 0.5 % 안에서 같아야 한다.

실행: python tests/test_genesis_wiring.py
"""
import sys, tempfile, shutil
from pathlib import Path
here = Path(__file__).resolve().parent
sys.path[:0] = [str(here), str(here.parent / "analysis")]
import fake_genesis, genesis_flight  # noqa: E402

out = genesis_flight.main(["--wind", "2"], gs_module=fake_genesis)
root = here.parent
(root / "results_fake_genesis.json").unlink(missing_ok=True)
(root / "figures" / "fake_genesis_vs_numpy.png").unlink(missing_ok=True)
assert abs(out["apogee_diff_pct"]) < 0.5, out
assert abs(out["genesis"]["tilt_at_burnout_deg"] - out["numpy_6dof"]["tilt_at_burnout_deg"]) < 0.5, out
print("PASS: 대역 엔진과 numpy 6자유도 최고고도 차이 %.3f %%" % out["apogee_diff_pct"])
