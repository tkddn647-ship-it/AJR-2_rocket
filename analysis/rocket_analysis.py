"""AJR-2 로켓 OpenRocket 결과 분석과 독립 검증.

원본(openrocket/, motors/)은 수정하지 않고 읽기만 한다.
  1) .ork 설계 파일에서 부품·질량 목록(BOM)을 뽑는다.
  2) 두 추력곡선(.eng)의 총역량·연소시간·평균/최대 추력을 계산한다.
  3) OpenRocket 시뮬레이션 CSV에서 최고고도·최대속도·안정여유·레일 이탈 속도를 뽑는다.
  4) 같은 추력곡선·질량·항력계수로 1자유도 수직 비행을 직접 적분해 OpenRocket 최고고도와 비교한다.

실행: python analysis/rocket_analysis.py   (numpy, matplotlib 필요)
"""
from __future__ import annotations
import json, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ORK = ROOT / "openrocket" / "AJR2.ork"
MOTORS = {"METEOR": ROOT / "motors" / "METEOR.eng", "TMS": ROOT / "motors" / "TMS.eng"}
SIMS = {"METEOR": ROOT / "openrocket" / "sim_meteor.csv", "TMS": ROOT / "openrocket" / "sim_tms.csv"}
FIG = ROOT / "figures"
COL = {"METEOR": "#8A90A3", "TMS": "#9A6F33"}
G0 = 9.80665


# ---------- 1) 설계 파일 ----------
def read_bom(path: Path):
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read(z.namelist()[0]))
    rows = []
    def walk(node, depth):
        for c in node:
            if c.tag == "subcomponents":
                walk(c, depth)
                continue
            name = c.findtext("name")
            if name is None:
                continue
            mass = c.findtext("overridemass")
            rows.append({"depth": depth, "type": c.tag, "name": name,
                         "length_m": float(c.findtext("length")) if c.findtext("length") else None,
                         "mass_kg": float(mass) if mass else None,
                         "material": c.findtext("material")})
            sub = c.find("subcomponents")
            if sub is not None:
                walk(sub, depth + 1)
    walk(root.find("rocket").find("subcomponents"), 0)
    fins = root.find(".//trapezoidfinset")
    fin = {k: float(fins.findtext(k)) for k in ("fincount", "rootchord", "tipchord", "height", "sweeplength")}
    body_r = float(root.find(".//bodytube").findtext("radius"))
    return rows, fin, body_r


# ---------- 2) 추력곡선 ----------
def read_eng(path: Path):
    lines = [l.split() for l in path.read_text().splitlines() if l.strip() and not l.startswith(";")]
    head = lines[0]
    d = np.array([[float(a), float(b)] for a, b, *_ in lines[1:]])
    return head, d[:, 0], d[:, 1]

def thrust_stats(t, F):
    I = float(np.trapezoid(F, t))
    return {"total_impulse_Ns": I, "burn_time_s": float(t[-1]), "avg_thrust_N": I / t[-1], "peak_thrust_N": float(F.max())}


# ---------- 3) OpenRocket 결과 ----------
def read_sim(path: Path, extra: bool = False):
    rows, events = [], {}
    for line in path.read_text(encoding="latin-1").splitlines():
        if line.startswith("#"):
            if "Event" in line:
                txt = line.strip("# ,")
                name = txt.split()[1]; events[name] = float(txt.split("t=")[1].split()[0])
            continue
        parts = line.split(",")[:25]
        try:
            rows.append([float(x) if x not in ("NaN", "") else np.nan for x in parts])
        except ValueError:
            pass
    a = np.array(rows)
    cols = dict(t=0, h=1, vz=2, v=4, acc=5, aoa=8, mass_g=12, I_long=14, I_rot=15, cp_cm=16, cg_cm=17, thrust=18, drag=19, cd=20)
    return {k: a[:, i] for k, i in cols.items()}, events


# ---------- 4) 독립 1자유도 적분 ----------
def isa_rho(h):
    T = 288.15 - 0.0065 * h
    return 1.225 * (T / 288.15) ** 4.2559

def sim_1dof(t_th, F_th, m0, m_prop, cd, area, rail=1.0, dt=0.001):
    """수직 1자유도. 질량은 누적 역량에 비례해 추진제가 줄어든다고 가정."""
    I_cum = np.concatenate([[0], np.cumsum(np.diff(t_th) * (F_th[1:] + F_th[:-1]) / 2)])
    I_tot = I_cum[-1]
    t = h = v = 0.0; out = [(0, 0, 0)]; rail_v = None
    while True:
        F = np.interp(t, t_th, F_th, right=0.0)
        m = m0 - m_prop * min(np.interp(t, t_th, I_cum, right=I_tot) / I_tot, 1.0)
        D = 0.5 * isa_rho(h) * v * abs(v) * cd * area
        a = (F - D) / m - G0
        if h <= 0 and a < 0 and v <= 0:  # 패드 위에서 아직 못 뜸
            a, v = 0.0, 0.0
        v += a * dt; h += v * dt; t += dt
        if rail_v is None and h >= rail:
            rail_v = v
        out.append((t, h, v))
        if t > t_th[-1] and v < 0:
            break
    o = np.array(out)
    i = int(np.argmax(o[:, 1]))
    return {"apogee_m": float(o[i, 1]), "t_apogee_s": float(o[i, 0]), "v_max_ms": float(o[:, 2].max()), "rail_exit_ms": rail_v}, o


def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E6E8EE"})
    for f in ("NanumGothic", "Noto Sans CJK KR", "Malgun Gothic", "AppleGothic"):
        try:
            matplotlib.font_manager.findfont(f, fallback_to_default=False); plt.rcParams["font.family"] = f; break
        except Exception:
            pass

    bom, fin, body_r = read_bom(ORK)
    area = np.pi * body_r ** 2
    summary = {"design": {"body_diameter_m": 2 * body_r, "fins": fin,
                          "component_mass_sum_kg": round(sum(r["mass_kg"] or 0 for r in bom), 3)},
               "motors": {}, "openrocket": {}, "crosscheck_1dof": {}}

    curves = {}
    for k, p in MOTORS.items():
        head, t, F = read_eng(p); curves[k] = (t, F)
        summary["motors"][k] = {"header": " ".join(head), **thrust_stats(t, F)}

    sims = {}
    for k, p in SIMS.items():
        d, ev = read_sim(p); sims[k] = d
        i = int(np.nanargmax(d["h"]))
        boost = d["t"] <= ev["BURNOUT"]
        stab = (d["cp_cm"] - d["cg_cm"]) / (2 * body_r * 100)
        coast = (d["t"] > ev["LAUNCHROD"]) & (d["t"] < d["t"][i])
        rail_v = float(np.interp(ev["LAUNCHROD"], d["t"], d["v"]))
        summary["openrocket"][k] = {
            "apogee_m": float(d["h"][i]), "t_apogee_s": float(d["t"][i]), "v_max_ms": float(np.nanmax(d["v"])),
            "acc_max_ms2": float(np.nanmax(d["acc"])), "liftoff_mass_g": float(d["mass_g"][0]),
            "rail_exit_s": ev["LAUNCHROD"], "rail_exit_ms": rail_v, "burnout_s": ev["BURNOUT"],
            "stability_min_cal": float(np.nanmin(stab[coast])),
            "cd_mean_ascent": float(np.nanmean(d["cd"][(d["t"] > ev["LAUNCHROD"]) & (d["t"] < d["t"][i])])),
        }
        prop = summary["openrocket"][k]["liftoff_mass_g"] - float(d["mass_g"][boost][-1])
        res, traj = sim_1dof(*curves[k], m0=d["mass_g"][0] / 1000, m_prop=prop / 1000,
                             cd=summary["openrocket"][k]["cd_mean_ascent"], area=area)
        res["apogee_error_pct"] = 100 * (res["apogee_m"] - summary["openrocket"][k]["apogee_m"]) / summary["openrocket"][k]["apogee_m"]
        summary["crosscheck_1dof"][k] = res
        sims[k + "_1dof"] = traj

    # --- 그림 1: 추력곡선 ---
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for k, (t, F) in curves.items():
        s = summary["motors"][k]
        ax.plot(t, F, color=COL[k], lw=2, label=f"{k}  {s['total_impulse_Ns']:.0f} N·s, {s['burn_time_s']:.2f} s")
    ax.set_xlabel("시간 (s)"); ax.set_ylabel("추력 (N)"); ax.legend(frameon=False)
    ax.set_title("같은 I247 모터, 두 추력곡선", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "thrust_curves.png", dpi=180); plt.close(fig)

    # --- 그림 2: 고도·속도 + 1자유도 검증 ---
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for k in ("METEOR", "TMS"):
        d = sims[k]; o = sims[k + "_1dof"]; asc = d["t"] <= summary["openrocket"][k]["t_apogee_s"] + 2
        axs[0].plot(d["t"][asc], d["h"][asc], color=COL[k], lw=2, label=f"{k} OpenRocket")
        axs[0].plot(o[:, 0], o[:, 1], color=COL[k], lw=1.2, ls="--", label=f"{k} 1자유도 검증")
        axs[1].plot(d["t"][asc], d["v"][asc], color=COL[k], lw=2)
        axs[1].plot(o[:, 0], o[:, 2], color=COL[k], lw=1.2, ls="--")
    axs[0].set_xlabel("시간 (s)"); axs[0].set_ylabel("고도 (m)"); axs[0].legend(frameon=False, fontsize=8.5)
    axs[1].set_xlabel("시간 (s)"); axs[1].set_ylabel("속도 (m/s)")
    axs[0].set_title("고도: OpenRocket(실선) vs 직접 적분(점선)", loc="left", fontsize=10)
    axs[1].set_title("속도", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "trajectory_crosscheck.png", dpi=180); plt.close(fig)

    # --- 그림 3: 안정여유 ---
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for k in ("METEOR", "TMS"):
        d = sims[k]; s = summary["openrocket"][k]
        m = (d["t"] >= s["rail_exit_s"]) & (d["t"] <= s["t_apogee_s"])
        ax.plot(d["t"][m], ((d["cp_cm"] - d["cg_cm"]) / (2 * body_r * 100))[m], color=COL[k], lw=2, label=k)
    ax.axhline(1.0, color="#B42318", lw=1, ls=":"); ax.text(0.2, 1.03, "통상 최소 기준 1 cal", color="#B42318", fontsize=8.5)
    ax.set_xlabel("시간 (s)"); ax.set_ylabel("안정여유 (cal)"); ax.legend(frameon=False)
    ax.set_title("레일 이탈부터 최고고도까지 안정여유 (CP−CG)/D", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "stability_margin.png", dpi=180); plt.close(fig)

    (ROOT / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with open(ROOT / "bom.csv", "w", encoding="utf-8") as f:
        f.write("depth,type,name,length_m,mass_kg,material\n")
        for r in bom:
            f.write(f"{r['depth']},{r['type']},{r['name']},{r['length_m'] or ''},{r['mass_kg'] or ''},{r['material'] or ''}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
