"""AJR-2 비행을 Genesis 물리 엔진으로 돌리는 스크립트 (2026.10 작성).

Genesis가 강체의 병진·회전 운동(적분)을 맡고, 매 스텝 힘과 모멘트는 numpy 6자유도
시뮬레이터(sixdof.py)와 똑같은 함수 `loads()`로 계산해 넣는다.
→ 같은 공력 모델을 두 개의 서로 다른 적분기(Genesis vs 직접 구현)로 돌려 결과를 비교한다.

  설치 (CPU로도 동작):  pip install genesis-world torch
  실행:                 python analysis/genesis_flight.py --wind 2
  결과:                 results_genesis.json, figures/genesis_vs_numpy.png

구현 메모
  - Genesis 원통(반지름 65.5 mm, 길이 1.05 m)을 로켓 몸체로 쓴다. 원통 축(local z) = 노즈 방향.
  - Genesis 중력은 끄고, 중력을 포함한 모든 힘을 loads()에서 계산해 넣는다.
  - 로켓은 연소하며 질량·관성이 변하지만 Genesis 원통은 고정 질량이다. 그래서 힘은 m_genesis/m(t),
    모멘트는 축별 I_genesis/I(t) 배로 바꿔 넣는다 → Genesis 안에서 가속도·각가속도가 실제 로켓과 같아진다.
  - 레일(1 m) 위에서는 동체축 방향 힘만 넣고 모멘트는 넣지 않는다.

이 스크립트는 작성 환경에 Genesis를 설치할 수 없어, Genesis와 같은 함수 이름을 가진 대역 모듈
(tests/fake_genesis.py)로 배선만 시험했다. 실제 Genesis 버전에 따라 외력 함수 이름이 다를 수 있어
apply_external_wrench(신버전)와 rigid_solver.apply_links_external_force(구버전)를 둘 다 시도한다.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rocket_analysis import read_eng, read_sim, ROOT, FIG          # noqa: E402
from sixdof import (Rocket, _Wind, loads, simulate, mass_props_from_openrocket,  # noqa: E402
                    R, L_TOTAL, RAIL)


def quat_to_R(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def to_np(x):
    try:
        return x.detach().cpu().numpy().reshape(-1)
    except AttributeError:
        return np.asarray(x, float).reshape(-1)


class Wrench:
    """Genesis 버전에 따라 다른 외력 API를 감싼다."""
    def __init__(self, scene, link):
        self.scene, self.link = scene, link
        self.mode = "link" if hasattr(link, "apply_external_wrench") else "solver"

    def apply(self, force_w, torque_w):
        if self.mode == "link":
            self.link.apply_external_wrench(force=force_w, torque=torque_w, local=False)
        else:
            s = self.scene.sim.rigid_solver
            s.apply_links_external_force(force=np.array([force_w]), links_idx=[self.link.idx])
            s.apply_links_external_torque(torque=np.array([torque_w]), links_idx=[self.link.idx])


def run_genesis(gs, rocket, wind_mean=2.0, seed=0, dt=0.002, t_max=40.0):
    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=dt, gravity=(0.0, 0.0, 0.0)), show_viewer=False)
    body = scene.add_entity(gs.morphs.Cylinder(radius=R, height=L_TOTAL, pos=(0.0, 0.0, L_TOTAL / 2)))
    scene.build()
    link = body.links[0]
    wrench = Wrench(scene, link)

    m_g = float(body.get_mass())
    I_g_long = m_g * (3 * R**2 + L_TOTAL**2) / 12      # 균일 원통의 가로축 관성
    I_g_axis = 0.5 * m_g * R**2                        # 길이축 관성
    z0 = to_np(body.get_pos())[2]

    wnd = _Wind(wind_mean, 0.10, seed)
    t = 0.0; on_rail = True; rail_v = np.nan; log = []
    while t < t_max:
        mp = rocket.props(t)
        m, x_cg, I_l, I_r = mp["mass"], mp["cg"], mp["I_long"], mp["I_rot"]
        pos = to_np(body.get_pos()) - np.array([0, 0, z0]); vel = to_np(body.get_vel())
        Rg = quat_to_R(to_np(body.get_quat()))
        Rm = np.column_stack([Rg[:, 2], Rg[:, 0], Rg[:, 1]])      # 동체 x=노즈(=Genesis local z)
        w_body = Rm.T @ to_np(body.get_ang())
        F, M_body, alpha = loads(rocket, t, pos, vel, Rm, w_body, wnd.step(dt), m, x_cg)
        xb = Rm[:, 0]
        if on_rail:
            fa = F @ xb
            if pos[2] <= 1e-4 and fa <= 0:
                fa = -m * (vel @ xb) / dt                              # 패드 위: 움직이지 않게
            F_apply = fa * xb * (m_g / m); T_apply = np.zeros(3)
        else:
            F_apply = F * (m_g / m)
            scale = np.array([I_g_axis / I_r, I_g_long / I_l, I_g_long / I_l])
            T_apply = Rm @ (M_body * scale)
        wrench.apply(F_apply, T_apply)
        scene.step()
        t += dt
        if on_rail and pos[2] >= RAIL:
            on_rail = False; rail_v = float(np.linalg.norm(vel))
        tilt = np.degrees(np.arccos(np.clip(xb[2], -1, 1)))
        log.append((t, *pos, *vel, tilt, -np.degrees(np.arctan2(xb[0], xb[2])), np.degrees(alpha)))
        if not on_rail and t > rocket.t_th[-1] and vel[2] < 0:
            break
    L = np.array(log); i = int(np.argmax(L[:, 3]))
    return L, {"apogee_m": float(L[i, 3]), "t_apogee_s": float(L[i, 0]),
               "drift_at_apogee_m": float(np.hypot(L[i, 1], L[i, 2])),
               "v_max_ms": float(np.linalg.norm(L[:, 4:7], axis=1).max()), "rail_exit_ms": rail_v,
               "tilt_at_burnout_deg": float(np.interp(rocket.t_th[-1], L[:, 0], L[:, 7]))}


def main(argv=None, gs_module=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wind", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if gs_module is None:
        import genesis as gs_module                       # pip install genesis-world
    t_th, F_th = read_eng(ROOT / "motors" / "TMS.eng")[1:]
    d, ev = read_sim(ROOT / "openrocket" / "sim_tms.csv", extra=True)
    rk = Rocket(t_th, F_th, mass_props_from_openrocket(d, ev))

    Lg, rg = run_genesis(gs_module, rk, args.wind, args.seed)
    Ln, rn = simulate(rk, wind_mean=args.wind, seed=args.seed)
    out = {"engine": getattr(gs_module, "__name__", "genesis"), "wind_ms": args.wind,
           "genesis": rg, "numpy_6dof": {k: rn[k] for k in rg},
           "openrocket_apogee_m": float(np.nanmax(d["h"]))}
    out["apogee_diff_pct"] = 100 * (rg["apogee_m"] / rn["apogee_m"] - 1)
    name = "results_genesis.json" if out["engine"] == "genesis" else f"results_{out['engine']}.json"
    (ROOT / name).write_text(json.dumps(out, ensure_ascii=False, indent=2))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for f in ("NanumGothic", "Noto Sans CJK KR", "Malgun Gothic", "AppleGothic"):
        try:
            matplotlib.font_manager.findfont(f, fallback_to_default=False); plt.rcParams["font.family"] = f; break
        except Exception:
            pass
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for L, lab, st in ((Ln, "numpy 6자유도", "-"), (Lg, out["engine"], "--")):
        m = L[:, 0] <= max(rg["t_apogee_s"], rn["t_apogee_s"])
        axs[0].plot(L[m, 0], L[m, 3], ls=st, lw=1.8, label=lab)
        axs[1].plot(L[m, 0], L[m, 8], ls=st, lw=1.8, label=lab)
    axs[0].set_title("고도 (m)", loc="left"); axs[1].set_title("바람이 불어오는 쪽으로 기울어진 각 (°)", loc="left")
    for ax in axs: ax.set_xlabel("시간 (s)"); ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / ("genesis_vs_numpy.png" if out["engine"] == "genesis" else f"{out['engine']}_vs_numpy.png"), dpi=160)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


if __name__ == "__main__":
    main()
