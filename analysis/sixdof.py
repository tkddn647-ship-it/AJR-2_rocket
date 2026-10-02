"""AJR-2 6자유도 비행 시뮬레이터 (2026.10 작성).

OpenRocket과 독립적으로 직접 구현한 강체 6자유도 모델로, 로드셀 실측 TMS 추력곡선과
바람을 넣어 자세(pitch·yaw)와 궤적을 계산한다.

  - 공력: Barrowman 방법으로 노즈콘·핀의 법선력 기울기 CNα와 압력중심(CP)을 설계 치수에서 직접 계산
  - 질량: 이륙·연소 종료 시점의 질량·무게중심·관성모멘트(OpenRocket 설계값)를 실측 누적 역량 비율로 보간
  - 힘: 추력(동체축), 중력, 축방향 항력(Cd), 받음각에 비례하는 법선력(CP에 작용), 피치·요 감쇠
  - 레일: 1 m 레일을 벗어날 때까지 수직으로 구속
  - 바람: 평균 풍속 + 1차 저역통과 난류 (난류 강도 10 %)

실행: python analysis/sixdof.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rocket_analysis import read_eng, read_sim, isa_rho, ROOT, FIG, G0

# ---------- 설계 치수 (openrocket/AJR2.ork) ----------
D = 0.131                 # 동체 지름 [m]
R = D / 2
A_REF = np.pi * R ** 2
L_NOSE = 0.200            # 타원형 노즈콘 길이
L_TOTAL = 0.200 + 0.480 + 0.370
FIN = dict(n=4, root=0.110, tip=0.049, span=0.180, sweep=0.061)
FIN_LE = L_TOTAL - 0.020 - FIN["root"]   # 핀 앞전 위치 (노즈 끝 기준)
RAIL = 1.0
CD = 0.55                 # OpenRocket TMS 시뮬레이션 상승 구간 평균 항력계수


def barrowman():
    """노즈콘과 핀의 CNα [1/rad]와 CP 위치 [m, 노즈 끝 기준]."""
    cn_n, x_n = 2.0, 0.333 * L_NOSE                     # 타원형 노즈콘
    a, b, s, m, n = FIN["root"], FIN["tip"], FIN["span"], FIN["sweep"], FIN["n"]
    lf = np.hypot(s, m + b / 2 - a / 2)                 # 중간 시위선 길이
    cn_f = (1 + R / (s + R)) * (4 * n * (s / D) ** 2) / (1 + np.sqrt(1 + (2 * lf / (a + b)) ** 2))
    x_f = FIN_LE + m * (a + 2 * b) / (3 * (a + b)) + (a + b - a * b / (a + b)) / 6
    cn = cn_n + cn_f
    return {"cn_nose": cn_n, "x_nose": x_n, "cn_fins": cn_f, "x_fins": x_f,
            "cn_total": cn, "x_cp": (cn_n * x_n + cn_f * x_f) / cn}


def quat_mul(q, r):
    w0, x0, y0, z0 = q; w1, x1, y1, z1 = r
    return np.array([w0*w1 - x0*x1 - y0*y1 - z0*z1, w0*x1 + x0*w1 + y0*z1 - z0*y1,
                     w0*y1 - x0*z1 + y0*w1 + z0*x1, w0*z1 + x0*y1 - y0*x1 + z0*w1])

def rotmat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])   # body → world


class Rocket:
    def __init__(self, t_th, F_th, mass_props):
        self.t_th, self.F_th = t_th, F_th
        self.I_cum = np.concatenate([[0], np.cumsum(np.diff(t_th) * (F_th[1:] + F_th[:-1]) / 2)])
        self.mp = mass_props
        self.aero = barrowman()

    def props(self, t):
        f = min(np.interp(t, self.t_th, self.I_cum, right=self.I_cum[-1]) / self.I_cum[-1], 1.0)
        p0, p1 = self.mp["start"], self.mp["end"]
        return {k: p0[k] + f * (p1[k] - p0[k]) for k in p0}

    def thrust(self, t):
        return float(np.interp(t, self.t_th, self.F_th, right=0.0))


def simulate(rocket, wind_mean=2.0, turb=0.10, seed=0, dt=0.002, t_max=40.0):
    rng = np.random.default_rng(seed)
    pos = np.zeros(3); vel = np.zeros(3)
    q = np.array([np.cos(np.pi / 4), 0, -np.sin(np.pi / 4), 0])   # 동체 x축을 세계 +z(위)로
    w = np.zeros(3)                                                  # 동체 좌표 각속도
    gust = np.zeros(2); tau_g = 1.0
    on_rail = True; t = 0.0; rail_v = np.nan
    log = []
    a = rocket.aero
    while t < t_max:
        mp = rocket.props(t)
        m, x_cg, I_l, I_r = mp["mass"], mp["cg"], mp["I_long"], mp["I_rot"]
        Rm = rotmat(q); xb = Rm[:, 0]
        # 바람 (서쪽에서 동쪽 +x로 불고, 난류는 1차 저역통과 백색잡음)
        gust += dt / tau_g * (-gust) + turb * wind_mean * np.sqrt(2 * dt / tau_g) * rng.standard_normal(2)
        wind = np.array([wind_mean + gust[0], gust[1], 0.0])
        v_air = vel - wind
        V = np.linalg.norm(v_air); rho = isa_rho(max(pos[2], 0.0)); qd = 0.5 * rho * V * V
        F = rocket.thrust(t) * xb + np.array([0, 0, -m * G0])
        M_body = np.zeros(3); alpha = 0.0
        if V > 1e-3:
            u = v_air / V
            F += -qd * A_REF * CD * u                                  # 축방향 항력 (상대풍 반대)
            v_b = Rm.T @ v_air
            perp = np.array([0.0, v_b[1], v_b[2]])                     # 동체에 수직인 상대풍 성분
            sin_a = np.linalg.norm(perp) / V
            alpha = np.arcsin(min(sin_a, 1.0))
            if sin_a > 1e-9:
                N_b = -qd * A_REF * a["cn_total"] * sin_a * perp / np.linalg.norm(perp)
                F += Rm @ N_b
                r_cp = np.array([-(a["x_cp"] - x_cg), 0, 0])         # CG → CP (동체 뒤쪽이 -x)
                M_body += np.cross(r_cp, N_b)
            # 피치·요 감쇠 (Barrowman 감쇠 모멘트)
            c_damp = 0.5 * rho * V * A_REF * (a["cn_nose"] * (a["x_nose"] - x_cg) ** 2 + a["cn_fins"] * (a["x_fins"] - x_cg) ** 2)
            M_body += -c_damp * np.array([0, w[1], w[2]])
        if on_rail:
            f_axis = F @ xb
            if pos[2] <= 0 and f_axis <= 0:     # 아직 추력이 무게를 못 넘음
                f_axis = 0.0; vel = np.zeros(3)
            acc = f_axis / m * xb
            w_dot = np.zeros(3)
        else:
            acc = F / m
            I = np.array([I_r, I_l, I_l])
            w_dot = (M_body - np.cross(w, I * w)) / I
        vel = vel + acc * dt; pos = pos + vel * dt
        w = w + w_dot * dt
        q = q + 0.5 * quat_mul(q, np.array([0, *w])) * dt; q /= np.linalg.norm(q)
        if on_rail and pos[2] >= RAIL:
            on_rail = False; rail_v = np.linalg.norm(vel)
        if not on_rail and pos[2] < 0:
            break
        tilt = np.degrees(np.arccos(np.clip(xb[2], -1, 1)))
        upwind = -np.degrees(np.arctan2(xb[0], xb[2]))              # +: 바람이 불어오는 쪽(서쪽)으로 기울어짐
        log.append((t, *pos, *vel, tilt, upwind, np.degrees(alpha), *np.degrees(w)))
        t += dt
        if not on_rail and t > rocket.t_th[-1] and vel[2] < 0:
            break
    L = np.array(log)
    i = int(np.argmax(L[:, 3]))
    return L, {"apogee_m": float(L[i, 3]), "t_apogee_s": float(L[i, 0]),
               "drift_at_apogee_m": float(np.hypot(L[i, 1], L[i, 2])),
               "v_max_ms": float(np.linalg.norm(L[:, 4:7], axis=1).max()),
               "rail_exit_ms": float(rail_v), "tilt_at_burnout_deg": float(np.interp(rocket.t_th[-1], L[:, 0], L[:, 7])),
               "tilt_at_apogee_deg": float(L[i, 7]), "max_aoa_after_rail_deg": float(L[L[:, 3] > RAIL + 0.5, 9].max())}


def mass_props_from_openrocket(d, ev):
    s = d["t"] <= ev["BURNOUT"]
    def at(idx):
        return {"mass": d["mass_g"][idx] / 1000, "cg": d["cg_cm"][idx] / 100,
                "I_long": d["I_long"][idx], "I_rot": d["I_rot"][idx]}
    return {"start": at(0), "end": at(np.where(s)[0][-1])}


def main():
    t_th, F_th = read_eng(ROOT / "motors" / "TMS.eng")[1:]
    d, ev = read_sim(ROOT / "openrocket" / "sim_tms.csv", extra=True)
    rk = Rocket(t_th, F_th, mass_props_from_openrocket(d, ev))
    aero = rk.aero
    or_cp = float(np.nanmedian(d["cp_cm"])) / 100

    L, res = simulate(rk, wind_mean=2.0, seed=0)
    summary = {"barrowman": {**{k: round(float(v), 4) for k, v in aero.items()},
                             "openrocket_cp_m": round(or_cp, 4),
                             "cp_difference_cm": round(100 * (aero["x_cp"] - or_cp), 2)},
               "nominal_wind_2ms": res,
               "openrocket": {"apogee_m": float(np.nanmax(d["h"]))}}
    summary["nominal_wind_2ms"]["apogee_vs_openrocket_pct"] = 100 * (res["apogee_m"] / summary["openrocket"]["apogee_m"] - 1)

    # 풍속별 민감도 (풍속마다 난류 시드 5개)
    sweep = []
    for wv in [0, 1, 2, 3, 4, 5, 6]:
        rs = [simulate(rk, wind_mean=wv, seed=s)[1] for s in range(5)]
        sweep.append({"wind_ms": wv,
                      "apogee_mean_m": float(np.mean([r["apogee_m"] for r in rs])),
                      "apogee_std_m": float(np.std([r["apogee_m"] for r in rs])),
                      "tilt_burnout_mean_deg": float(np.mean([r["tilt_at_burnout_deg"] for r in rs])),
                      "drift_mean_m": float(np.mean([r["drift_at_apogee_m"] for r in rs]))})
    summary["wind_sweep"] = sweep
    (ROOT / "results_6dof.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    # --- 그림 4: 자세와 받음각 ---
    COL = "#9A6F33"; GR = "#8A90A3"
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    asc = L[:, 0] <= res["t_apogee_s"]
    axs[0].plot(d["t"][d["t"] <= res["t_apogee_s"] + 1], d["h"][d["t"] <= res["t_apogee_s"] + 1], color=GR, lw=2, label="OpenRocket")
    axs[0].plot(L[asc, 0], L[asc, 3], color=COL, lw=1.6, ls="--", label="6자유도 (직접 구현)")
    axs[0].set_title("고도 (m)", loc="left", fontsize=10); axs[0].legend(frameon=False, fontsize=8.5)
    axs[1].plot(L[asc, 0], L[asc, 8], color=COL, lw=1.8)
    axs[1].set_title("바람이 불어오는 쪽으로 기울어진 각 (°)", loc="left", fontsize=10)
    axs[1].axhline(0, color="#999", lw=0.8)
    m = asc & (L[:, 3] > RAIL + 0.5)
    or_m = (d["t"] > ev["LAUNCHROD"]) & (d["t"] <= res["t_apogee_s"])
    axs[2].plot(d["t"][or_m], d["aoa"][or_m], color=GR, lw=1.6, label="OpenRocket")
    axs[2].plot(L[m, 0], L[m, 9], color=COL, lw=1.2, ls="--", label="6자유도")
    axs[2].set_title("받음각 (°)", loc="left", fontsize=10); axs[2].set_ylim(0, 15); axs[2].legend(frameon=False, fontsize=8.5)
    for ax in axs: ax.set_xlabel("시간 (s)")
    fig.suptitle("TMS 실측 추력, 평균풍 2 m/s: 레일을 벗어나면 바람이 불어오는 쪽으로 머리를 돌린다 (weathercocking)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(); fig.savefig(FIG / "sixdof_attitude.png", dpi=180); plt.close(fig)

    # --- 그림 5: 풍속 민감도 ---
    fig, ax1 = plt.subplots(figsize=(8, 3.5))
    ws = [s["wind_ms"] for s in sweep]
    ax1.errorbar(ws, [s["apogee_mean_m"] for s in sweep], yerr=[s["apogee_std_m"] for s in sweep], color=COL, marker="o", lw=2, capsize=3)
    ax1.set_xlabel("평균 풍속 (m/s)"); ax1.set_ylabel("최고고도 (m)", color=COL)
    ax2 = ax1.twinx(); ax2.spines["right"].set_visible(True)
    ax2.plot(ws, [s["tilt_burnout_mean_deg"] for s in sweep], color=GR, marker="s", lw=1.6)
    ax2.set_ylabel("연소 종료 시 기울기 (°)", color=GR)
    ax1.set_title("풍속이 커질수록 연소 중에 바람을 향해 더 눕고 최고고도가 낮아진다 (난류 시드 5개)", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "sixdof_wind_sweep.png", dpi=180); plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#E6E8EE"})
    for f in ("NanumGothic", "Noto Sans CJK KR", "Malgun Gothic", "AppleGothic"):
        try:
            matplotlib.font_manager.findfont(f, fallback_to_default=False); plt.rcParams["font.family"] = f; break
        except Exception:
            pass
    main()
