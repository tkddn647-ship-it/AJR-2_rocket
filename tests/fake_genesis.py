"""Genesis 대역 모듈 — genesis_flight.py의 배선(좌표 변환, 질량·관성 보정, 외력 적용)을 시험하기 위한 것.

실제 Genesis가 아니다. Genesis와 같은 함수 이름(init, Scene, morphs.Cylinder, get_pos/get_quat/get_vel/get_ang,
apply_external_wrench)만 흉내 내고, 균일 원통 강체를 직접 적분한다.
이 대역으로 돌린 결과가 numpy 6자유도와 같으면, 스크립트의 좌표·보정 로직이 맞다는 뜻이다.
"""
import types
import numpy as np

__name__ = "fake_genesis"
cpu = "cpu"


def init(**kw):
    pass


class _SimOptions:
    def __init__(self, dt=0.01, gravity=(0, 0, -9.81)):
        self.dt, self.gravity = dt, np.array(gravity, float)

options = types.SimpleNamespace(SimOptions=_SimOptions)


class _Cylinder:
    def __init__(self, radius, height, pos=(0, 0, 0)):
        self.r, self.h, self.pos = radius, height, np.array(pos, float)

morphs = types.SimpleNamespace(Cylinder=_Cylinder)


def _qmul(a, b):
    w0, x0, y0, z0 = a; w1, x1, y1, z1 = b
    return np.array([w0*w1-x0*x1-y0*y1-z0*z1, w0*x1+x0*w1+y0*z1-z0*y1, w0*y1-x0*z1+y0*w1+z0*x1, w0*z1+x0*y1-y0*x1+z0*w1])

def _R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


class _Link:
    def __init__(self, ent):
        self.ent, self.idx = ent, 0
    def apply_external_wrench(self, force=None, torque=None, local=False):
        assert not local
        self.ent.F = np.asarray(force, float); self.ent.T = np.asarray(torque, float)


class _Entity:
    def __init__(self, morph, rho=1000.0):
        self.m = rho * np.pi * morph.r**2 * morph.h
        self.I = np.array([self.m*(3*morph.r**2+morph.h**2)/12]*2 + [0.5*self.m*morph.r**2])
        self.p = morph.pos.copy(); self.v = np.zeros(3); self.q = np.array([1.0, 0, 0, 0]); self.w = np.zeros(3)
        self.F = np.zeros(3); self.T = np.zeros(3); self.links = [_Link(self)]
    def get_mass(self): return self.m
    def get_pos(self): return self.p.copy()
    def get_vel(self): return self.v.copy()
    def get_quat(self): return self.q.copy()
    def get_ang(self): return _R(self.q) @ self.w           # 세계 좌표 각속도
    def step(self, dt, g):
        self.v += (self.F / self.m + g) * dt; self.p += self.v * dt
        Rm = _R(self.q); Tb = Rm.T @ self.T
        self.w += (Tb - np.cross(self.w, self.I * self.w)) / self.I * dt
        self.q = self.q + 0.5 * _qmul(self.q, np.array([0, *self.w])) * dt; self.q /= np.linalg.norm(self.q)
        self.F[:] = 0; self.T[:] = 0


class Scene:
    def __init__(self, sim_options=None, show_viewer=False):
        self.o = sim_options; self.ents = []
    def add_entity(self, morph):
        e = _Entity(morph); self.ents.append(e); return e
    def build(self): pass
    def step(self):
        for e in self.ents: e.step(self.o.dt, self.o.gravity)
