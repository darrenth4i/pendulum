"""
Double Pendulum — Control System (standalone; kept separate from v2)
=======================================================================
Actuation : horizontal pivot acceleration (cart-pole style)
a ∈ [−800, +800] px/s² → converted to m/s² internally
Cart mass 2 kg (not exposed).

State machine
─────────────
OFF → DAMPING → IDLE
└──────────→ SWING-UP → BALANCING
↑←── (fall-back) ──┘

Buttons
DAMP : any → DAMPING → IDLE
BALANCE : any → DAMPING → SWING-UP → BALANCING

Controllers
DAMPING : PD on omega1, omega2 + cart re-centering
SWING-UP : Åström-Furuta energy pump on angle1 (collocated)
BALANCING : full-state LQR [x_pivot, ẋ_pivot, angle1, omega1, angle2, omega2]
or 4-D in single-pendulum mode
Numerical Jacobian + scipy solve_continuous_are
Recomputes automatically whenever a physics slider changes.

HUD (on-canvas overlay)
• Mode label (colour-coded)
• Pivot force arrow (direction + magnitude)
• Coloured pivot crosshair
• Live energy readout
• angle1, angle2, omega1, omega2
• LQR K-vector

Note
────
Single-pendulum balancing should be reasonably stable.
Double-pendulum balancing has not been implemented.
"""

import sys
import math
import numpy as np
from collections import deque
from enum import Enum

from scipy.linalg import solve_continuous_are

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QSlider, QLabel, QFrame, QPushButton, QMessageBox,
)
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QLinearGradient, QFont, QPolygonF,
)
from PyQt5.QtCore import Qt, QTimer, QPointF

# ══════════════════════════════════════════════════════════════════════════════
# Free-swinging dynamics
# ══════════════════════════════════════════════════════════════════════════════

def derivatives(state, arm1_m, arm2_m, m1, m2, g):
    angle1, omega1, angle2, omega2 = state
    delta_angle = angle1 - angle2
    denominator = 2*m1 + m2 - m2*math.cos(2*delta_angle)

    accel1 = (
        -g*(2*m1 + m2)*math.sin(angle1)
        - m2*g*math.sin(angle1 - 2*angle2)
        - 2*m2*math.sin(delta_angle)*(omega2**2*arm2_m + omega1**2*arm1_m*math.cos(delta_angle))
    ) / (arm1_m * denominator)

    accel2 = (
        2*math.sin(delta_angle) * (
            omega1**2*arm1_m*(m1 + m2)
            + g*(m1 + m2)*math.cos(angle1)
            + omega2**2*arm2_m*m2*math.cos(delta_angle)
        )
    ) / (arm2_m * denominator)

    return (omega1, accel1, omega2, accel2)


def derivatives_single(state2, arm1_m, g):
    angle1, omega1 = state2
    return (omega1, -g/arm1_m * math.sin(angle1))


# ══════════════════════════════════════════════════════════════════════════════
# Physics — moving-pivot control dynamics
# ══════════════════════════════════════════════════════════════════════════════

def derivatives_ctrl(state, arm1_m, arm2_m, m1, m2, g, a, k_air=0.0):
    angle1, omega1, angle2, omega2 = state
    delta_angle = angle1 - angle2
    denominator = 2*m1 + m2 - m2*math.cos(2*delta_angle)

    force1 = (
        -(m1 + m2) * g * math.sin(angle1)
        - m2 * arm2_m * omega2**2 * math.sin(delta_angle)
        - (m1 + m2) * a * math.cos(angle1)
    )
    force2_from_m2 = (
        arm1_m * omega1**2 * math.sin(delta_angle)
        - g * math.sin(angle2)
        - a * math.cos(angle2)
    )

    accel1 = 2 * (force1 - m2 * math.cos(delta_angle) * force2_from_m2) / (arm1_m * denominator) - k_air * omega1
    accel2 = 2 * ((m1 + m2) * force2_from_m2 - math.cos(delta_angle) * force1) / (arm2_m * denominator) - k_air * omega2

    return (omega1, accel1, omega2, accel2)


def derivatives_ctrl_single(state2, arm1_m, g, a, k_air=0.0):
    angle1, omega1 = state2
    return (omega1, -g/arm1_m*math.sin(angle1) - a/arm1_m*math.cos(angle1) - k_air * omega1)


# ══════════════════════════════════════════════════════════════════════════════
# RK4 integrators (numerical stepping)
# ══════════════════════════════════════════════════════════════════════════════

def _rk4(f, state, dt, n):
    k1 = f(state)
    k2 = f(tuple(state[i] + dt/2 * k1[i] for i in range(n)))
    k3 = f(tuple(state[i] + dt/2 * k2[i] for i in range(n)))
    k4 = f(tuple(state[i] + dt * k3[i] for i in range(n)))
    return tuple(state[i] + dt/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(n))


def rk4_free(state, dt, arm1_m, arm2_m, m1, m2, g):
    return _rk4(lambda s: derivatives(s, arm1_m, arm2_m, m1, m2, g), state, dt, 4)


def rk4_free_single(state2, dt, arm1_m, g):
    return _rk4(lambda s: derivatives_single(s, arm1_m, g), state2, dt, 2)


def rk4_ctrl(state, dt, arm1_m, arm2_m, m1, m2, g, a, k_air=0.0):
    return _rk4(lambda s: derivatives_ctrl(s, arm1_m, arm2_m, m1, m2, g, a, k_air), state, dt, 4)


def rk4_ctrl_single(state2, dt, arm1_m, g, a, k_air=0.0):
    return _rk4(lambda s: derivatives_ctrl_single(s, arm1_m, g, a, k_air), state2, dt, 2)


# ══════════════════════════════════════════════════════════════════════════════
# Energy helpers
# ══════════════════════════════════════════════════════════════════════════════

def energy_double(state, arm1_m, arm2_m, m1, m2, g):
    angle1, omega1, angle2, omega2 = state
    delta_angle = angle1 - angle2
    kinetic_energy = (
        0.5 * m1 * arm1_m**2 * omega1**2
        + 0.5 * m2 * (
            arm1_m**2 * omega1**2 + arm2_m**2 * omega2**2
            + 2 * arm1_m * arm2_m * omega1 * omega2 * math.cos(delta_angle)
        )
    )
    potential_energy = -m1*g*arm1_m*math.cos(angle1) - m2*g*(arm1_m*math.cos(angle1) + arm2_m*math.cos(angle2))
    return kinetic_energy + potential_energy


def ke_double(state, arm1_m, arm2_m, m1, m2):
    angle1, omega1, angle2, omega2 = state
    delta_angle = angle1 - angle2
    return (
        0.5 * m1 * arm1_m**2 * omega1**2
        + 0.5 * m2 * (
            arm1_m**2 * omega1**2 + arm2_m**2 * omega2**2
            + 2 * arm1_m * arm2_m * omega1 * omega2 * math.cos(delta_angle)
        )
    )


def energy_ref_double(arm1_m, arm2_m, m1, m2, g):
    return m1*g*arm1_m + m2*g*(arm1_m + arm2_m)


def energy_single(state2, arm1_m, m1, g):
    angle1, omega1 = state2
    return 0.5*m1*arm1_m**2*omega1**2 - m1*g*arm1_m*math.cos(angle1)


def ke_single(state2, arm1_m, m1):
    _, omega1 = state2
    return 0.5*m1*arm1_m**2*omega1**2


def energy_ref_single(arm1_m, m1, g):
    return m1*g*arm1_m


# ══════════════════════════════════════════════════════════════════════════════
# LQR
# ══════════════════════════════════════════════════════════════════════════════

CART_MASS = 2.0
A_MAX_M = 8.0
PX_PER_M = 100.0


def compute_lqr(arm1_m, arm2_m, m1, m2, g, aggressiveness, single_mode):
    if single_mode:
        equilibrium_state = np.array([0.0, 0.0, math.pi, 0.0])

        def linearized_dynamics(x, control_signal):
            xp, vp, th, om = x
            d = derivatives_ctrl_single((th, om), arm1_m, g, control_signal)
            return np.array([vp, control_signal, d[0], d[1]])

        Q = np.diag([2.0, 1.0, 200.0, 20.0])
    else:
        equilibrium_state = np.array([0.0, 0.0, math.pi, 0.0, math.pi, 0.0])

        def linearized_dynamics(x, control_signal):
            xp, vp, th1, om1, th2, om2 = x
            d = derivatives_ctrl((th1, om1, th2, om2), arm1_m, arm2_m, m1, m2, g, control_signal)
            return np.array([vp, control_signal, d[0], d[1], d[2], d[3]])

        Q = np.diag([2.0, 1.0, 400.0, 40.0, 400.0, 40.0])

    n = len(equilibrium_state)
    eps = 1e-6

    A = np.zeros((n, n))
    for j in range(n):
        xp = equilibrium_state.copy()
        xm = equilibrium_state.copy()
        xp[j] += eps
        xm[j] -= eps
        A[:, j] = (linearized_dynamics(xp, 0.0) - linearized_dynamics(xm, 0.0)) / (2 * eps)

    B = ((linearized_dynamics(equilibrium_state, eps) - linearized_dynamics(equilibrium_state, -eps)) / (2 * eps)).reshape(n, 1)

    r_base = 20.0 if single_mode else 10.0
    R_val = max(1e-4, r_base / float(aggressiveness))
    R = np.array([[R_val]])

    try:
        P = solve_continuous_are(A, B, Q, R)
        K = (1.0 / R_val) * B.T @ P
        return K[0]
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Control modes
# ══════════════════════════════════════════════════════════════════════════════

class Mode(Enum):
    OFF = "OFF"
    DAMPING = "DAMPING"
    IDLE = "IDLE"
    SWING_UP = "SWING-UP"
    BALANCING = "BALANCING"


_MCOL = {
    Mode.OFF: QColor(100, 100, 140),
    Mode.DAMPING: QColor(80, 130, 255),
    Mode.IDLE: QColor(120, 120, 160),
    Mode.SWING_UP: QColor(255, 210, 40),
    Mode.BALANCING: QColor(50, 220, 100),
}

GATE_ANGLE_RAD = math.radians(25)
GATE_KE_J = 0.35
FALLBACK_ANGLE = math.radians(37)

TRAIL_LEN = 300
GRAB_RADIUS = 30
_HALF_CROSS = 14


def _wrap(a, b):
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


class PendulumCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(420, 420)
        self.setCursor(Qt.ArrowCursor)

        self.state = [math.pi/2, 0.0, math.pi/4, 0.0]
        self.length1 = 120.0
        self.length2 = 100.0
        self.gravity = 9.81
        self.m1 = 1.0
        self.m2 = 1.0
        self.single_mode = False
        self.air_resistance = False
        self._pending_air_off = False

        self.cart_x = 0.0
        self.cart_v = 0.0
        self.pivot_x_manual = 0

        self.ctrl_mode = Mode.OFF
        self._after_damp = Mode.IDLE
        self._aggressiveness = 50
        self.K = None
        self._lqr_dirty = True
        self._ctrl_u = 0.0

        self.trail: deque[QPointF] = deque(maxlen=TRAIL_LEN)

        self._dragging = None
        self._prev_angle = None

    def _pivot(self):
        ox = self.width()/2 + self.cart_x*PX_PER_M + self.pivot_x_manual
        oy = self.height()/3
        return ox, oy

    def _positions(self):
        ox, oy = self._pivot()
        angle1, _, angle2, _ = self.state
        x1 = ox + self.length1*math.sin(angle1)
        y1 = oy + self.length1*math.cos(angle1)
        x2 = x1 + self.length2*math.sin(angle2)
        y2 = y1 + self.length2*math.cos(angle2)
        return ox, oy, x1, y1, x2, y2

    def advance(self, dt: float):
        if self._dragging is not None:
            self._ctrl_u = 0.0
            return

        pivot_accel_m = self._step_control(dt)
        self._ctrl_u = pivot_accel_m

        self.cart_v += pivot_accel_m * dt
        self.cart_x += self.cart_v * dt

        wall_m = self.width() * 0.40 / PX_PER_M
        if abs(self.cart_x) > wall_m:
            self.cart_x = math.copysign(wall_m, self.cart_x)
            self.cart_v *= -0.15

        arm1_m = self.length1 / PX_PER_M
        arm2_m = self.length2 / PX_PER_M
        k_air = 0.5 if self.air_resistance else 0.0

        if self.single_mode:
            s2 = rk4_ctrl_single((self.state[0], self.state[1]), dt, arm1_m, self.gravity, pivot_accel_m, k_air)
            self.state[0], self.state[1] = s2
            _, _, x1, y1, _, _ = self._positions()
            self.trail.append(QPointF(x1, y1))
        else:
            ns = rk4_ctrl(self.state, dt, arm1_m, arm2_m, self.m1, self.m2, self.gravity, pivot_accel_m, k_air)
            self.state = list(ns)
            _, _, _, _, x2, y2 = self._positions()
            self.trail.append(QPointF(x2, y2))

    def _step_control(self, dt: float) -> float:
        mode = self.ctrl_mode
        if mode in (Mode.OFF, Mode.IDLE):
            return 0.0

        angle1, omega1, angle2, omega2 = self.state
        arm1_m = self.length1 / PX_PER_M
        arm2_m = self.length2 / PX_PER_M
        g = self.gravity
        m1, m2 = self.m1, self.m2
        cart_x_m, cart_v_m = self.cart_x, self.cart_v

        control_signal = 0.0

        if mode == Mode.DAMPING:
            control_signal = (+18.0*omega1 + (0.0 if self.single_mode else 6.0*omega2) - 4.0*cart_v_m - 1.8*cart_x_m)

            still = (
                abs(omega1) < 0.08
                and (self.single_mode or abs(omega2) < 0.08)
                and abs(cart_v_m) < 0.08
            )
            if still:
                self.ctrl_mode = self._after_damp

                # Automatically disable air resistance only for the balance flow
                if self._pending_air_off and self._after_damp == Mode.SWING_UP:
                    self.air_resistance = False
                    self._pending_air_off = False

        elif mode == Mode.SWING_UP:
            if self.single_mode:
                E = energy_single((angle1, omega1), arm1_m, m1, g)
                reference_energy = energy_ref_single(arm1_m, m1, g)
                kinetic_energy = ke_single((angle1, omega1), arm1_m, m1)
            else:
                E = energy_double(self.state, arm1_m, arm2_m, m1, m2, g)
                reference_energy = energy_ref_double(arm1_m, arm2_m, m1, m2, g)
                kinetic_energy = ke_double(self.state, arm1_m, arm2_m, m1, m2)

            dE = E - reference_energy
            pump_control = 3.0 * omega1 * math.cos(angle1) * dE
            if not self.single_mode:
                pump_control += 1.5 * omega2 * math.cos(angle2) * dE
            stabilizer_control = -0.8*cart_x_m - 0.4*cart_v_m
            control_signal = pump_control + stabilizer_control

            δθ1 = _wrap(angle1, math.pi)
            δθ2 = _wrap(angle2, math.pi) if not self.single_mode else 0.0
            ke_gate = GATE_KE_J if self.single_mode else GATE_KE_J * 2.5
            if (abs(δθ1) < GATE_ANGLE_RAD and abs(δθ2) < GATE_ANGLE_RAD and kinetic_energy < ke_gate):
                self.ctrl_mode = Mode.BALANCING
                if self._lqr_dirty or self.K is None:
                    self.K = compute_lqr(arm1_m, arm2_m, m1, m2, g, self._aggressiveness, self.single_mode)
                    self._lqr_dirty = False

        elif mode == Mode.BALANCING:
            if self._lqr_dirty or self.K is None:
                self.K = compute_lqr(arm1_m, arm2_m, m1, m2, g, self._aggressiveness, self.single_mode)
                self._lqr_dirty = False

            if self.K is not None:
                δθ1 = _wrap(angle1, math.pi)
                δθ2 = _wrap(angle2, math.pi)
                if self.single_mode:
                    xv = np.array([cart_x_m, cart_v_m, δθ1, omega1])
                else:
                    xv = np.array([cart_x_m, cart_v_m, δθ1, omega1, δθ2, omega2])
                control_signal = float(-self.K @ xv)

                fallback = FALLBACK_ANGLE if self.single_mode else math.radians(50)
                if (abs(δθ1) > fallback or (not self.single_mode and abs(δθ2) > fallback)):
                    self.ctrl_mode = Mode.SWING_UP
            else:
                self._lqr_dirty = True

        return max(-A_MAX_M, min(A_MAX_M, control_signal))

    def reset(self):
        self.state = [math.pi/2, 0.0, math.pi/4, 0.0]
        self.trail.clear()
        self._dragging = None

    def full_reset(self):
        self.state = [0.0, 0.0, 0.0, 0.0]
        self.trail.clear()
        self._dragging = None
        self.cart_x = 0.0
        self.cart_v = 0.0
        self.ctrl_mode = Mode.OFF

    def reset_velocity(self):
        self.state[1] = 0.0
        self.state[3] = 0.0
        self.trail.clear()

    def _reset_cart(self):
        self.cart_x = 0.0
        self.cart_v = 0.0

    def toggle_single_mode(self):
        self.single_mode = not self.single_mode
        self.trail.clear()
        self.ctrl_mode = Mode.OFF
        self._reset_cart()
        self._lqr_dirty = True

    @staticmethod
    def _dist(ax, ay, bx, by):
        return math.hypot(ax - bx, ay - by)

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            return
        ox, oy, x1, y1, x2, y2 = self._positions()
        panel_x, panel_y = ev.x(), ev.y()
        if not self.single_mode and self._dist(panel_x, panel_y, x2, y2) <= GRAB_RADIUS:
            self._dragging = 2
            self._prev_angle = math.atan2(x2 - x1, y2 - y1)
            self.state[3] = 0.0
            self.setCursor(Qt.ClosedHandCursor)
        elif self._dist(panel_x, panel_y, x1, y1) <= GRAB_RADIUS:
            self._dragging = 1
            self._prev_angle = math.atan2(x1 - ox, y1 - oy)
            self.state[1] = 0.0
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, ev):
        if self._dragging is None:
            ox, oy, x1, y1, x2, y2 = self._positions()
            panel_x, panel_y = ev.x(), ev.y()
            near = self._dist(panel_x, panel_y, x1, y1) <= GRAB_RADIUS
            if not self.single_mode:
                near = near or self._dist(panel_x, panel_y, x2, y2) <= GRAB_RADIUS
            self.setCursor(Qt.OpenHandCursor if near else Qt.ArrowCursor)
            return

        ox, oy, x1, y1, x2, y2 = self._positions()
        panel_x, panel_y = ev.x(), ev.y()
        if self._dragging == 1:
            new_a = math.atan2(panel_x - ox, panel_y - oy)
            if self._prev_angle is not None:
                self.state[1] = _wrap(new_a, self._prev_angle) * 60
            self.state[0] = new_a
            self._prev_angle = new_a
        elif self._dragging == 2:
            _, _, x1, y1, _, _ = self._positions()
            new_a = math.atan2(panel_x - x1, panel_y - y1)
            if self._prev_angle is not None:
                self.state[3] = _wrap(new_a, self._prev_angle) * 60
            self.state[2] = new_a
            self._prev_angle = new_a

        self.trail.clear()
        self.update()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton and self._dragging is not None:
            self._dragging = None
            self._prev_angle = None
            self.setCursor(Qt.ArrowCursor)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0, QColor(10, 10, 28))
        grad.setColorAt(1, QColor(20, 10, 40))
        p.fillRect(self.rect(), grad)

        ox, oy, x1, y1, x2, y2 = self._positions()
        mode = self.ctrl_mode
        mode_color = _MCOL.get(mode, _MCOL[Mode.OFF])

        trail = list(self.trail)
        for i in range(1, len(trail)):
            t = i / TRAIL_LEN
            a = int(220 * t)
            r = int(80 + 175 * t)
            gc = int(40 + 60 * t)
            b = int(200 - 100 * t)
            p.setPen(QPen(QColor(r, gc, b, a), max(1, t * 2.5)))
            p.drawLine(trail[i - 1], trail[i])

        active = mode not in (Mode.OFF, Mode.IDLE)
        cw = 2 if active else 1
        p.setPen(QPen(mode_color, cw, Qt.DotLine))
        p.drawLine(QPointF(ox - _HALF_CROSS, oy), QPointF(ox + _HALF_CROSS, oy))
        p.drawLine(QPointF(ox, oy - _HALF_CROSS), QPointF(ox, oy + _HALF_CROSS))
        p.setPen(QPen(mode_color, cw))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(ox, oy), 6, 6)

        if active and abs(self._ctrl_u) > 0.02:
            self._draw_force_arrow(p, ox, oy, self._ctrl_u, mode_color)

        p.setPen(QPen(QColor(200, 200, 230), 2))
        p.drawLine(QPointF(ox, oy), QPointF(x1, y1))
        if not self.single_mode:
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        if self._dragging == 1:
            p.setPen(QPen(QColor(255, 255, 100), 2))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(x1, y1), 15, 15)
        g1 = QLinearGradient(x1 - 10, y1 - 10, x1 + 10, y1 + 10)
        g1.setColorAt(0, QColor(120, 200, 255))
        g1.setColorAt(1, QColor(40, 100, 200))
        p.setBrush(QBrush(g1))
        p.setPen(QPen(QColor(160, 220, 255), 1))
        p.drawEllipse(QPointF(x1, y1), 11, 11)

        if not self.single_mode:
            if self._dragging == 2:
                p.setPen(QPen(QColor(255, 255, 100), 2))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QPointF(x2, y2), 15, 15)
            g2 = QLinearGradient(x2 - 10, y2 - 10, x2 + 10, y2 + 10)
            g2.setColorAt(0, QColor(255, 140, 100))
            g2.setColorAt(1, QColor(200, 40, 40))
            p.setBrush(QBrush(g2))
            p.setPen(QPen(QColor(255, 180, 140), 1))
            p.drawEllipse(QPointF(x2, y2), 11, 11)

        self._draw_hud(p)

    def _draw_force_arrow(self, p: QPainter, ox, oy, u_m, line_color: QColor):
        length = (u_m / A_MAX_M) * 60.0
        tx = ox + length
        sign = 1 if length >= 0 else -1
        head = 9

        p.setPen(QPen(line_color, 2))
        p.drawLine(QPointF(ox, oy), QPointF(tx, oy))

        pts = QPolygonF([
            QPointF(tx, oy),
            QPointF(tx - sign * head, oy - 5),
            QPointF(tx - sign * head, oy + 5),
        ])
        p.setBrush(QBrush(line_color))
        p.setPen(Qt.NoPen)
        p.drawPolygon(pts)

    def _draw_hud(self, p: QPainter):
        angle1, omega1, angle2, omega2 = self.state
        mode = self.ctrl_mode
        mode_color = _MCOL.get(mode, _MCOL[Mode.OFF])
        arm1_m = self.length1 / PX_PER_M
        arm2_m = self.length2 / PX_PER_M
        g = self.gravity

        lines = []
        lines.append((f"● {mode.value}", mode_color))

        F_N = CART_MASS * self._ctrl_u
        arrow_sym = "→" if self._ctrl_u >= 0 else "←"
        lines.append((f"F {F_N:+6.1f} N {arrow_sym}", mode_color))

        cx_px = self.cart_x * PX_PER_M
        cv_px = self.cart_v * PX_PER_M
        lines.append((f"cart {cx_px:+6.1f}px {cv_px:+6.1f}px/s", None))

        lines.append((f"angle1 {math.degrees(angle1):+7.1f}° omega1 {omega1:+5.2f}", None))

        if self.single_mode:
            lines.append(("angle2 — omega2 —", QColor(80, 80, 100)))
        else:
            lines.append((f"angle2 {math.degrees(angle2):+7.1f}° omega2 {omega2:+5.2f}", None))

        if self.single_mode:
            E = energy_single((angle1, omega1), arm1_m, self.m1, g)
            reference_energy = energy_ref_single(arm1_m, self.m1, g)
        else:
            E = energy_double(self.state, arm1_m, arm2_m, self.m1, self.m2, g)
            reference_energy = energy_ref_double(arm1_m, arm2_m, self.m1, self.m2, g)
        energy_color = QColor(100, 255, 150) if abs(E - reference_energy) < 1.0 else None
        lines.append((f"E {E:+7.2f}J ref {reference_energy:.2f}J", energy_color))

        if self.K is not None:
            K = self.K
            if self.single_mode:
                lines.append((f"K[x,v] {K[0]:+6.1f} {K[1]:+6.1f}", QColor(140, 255, 180)))
                lines.append((f"K[θ,ω] {K[2]:+6.0f} {K[3]:+6.1f}", QColor(140, 255, 180)))
            else:
                lines.append((f"K[x,v] {K[0]:+5.1f} {K[1]:+5.1f}", QColor(140, 255, 180)))
                lines.append((f"K[angle1,omega1] {K[2]:+5.0f} {K[3]:+5.1f}", QColor(140, 255, 180)))
                if len(K) >= 6:
                    lines.append((f"K[angle2,omega2] {K[4]:+5.0f} {K[5]:+5.1f}", QColor(140, 255, 180)))
                else:
                    lines.append(("K — (not computed)", QColor(160, 80, 80)))

        font = QFont("Courier New", 9)
        p.setFont(font)
        line_height = 15
        pad = 6
        w_box = 222
        h_box = len(lines) * line_height + pad * 2
        panel_x, panel_y = 10, 10

        p.setBrush(QBrush(QColor(0, 0, 0, 140)))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(panel_x - pad, panel_y - pad, w_box, h_box, 7, 7)

        default_col = QColor(185, 190, 210)
        for i, (line_text, line_color) in enumerate(lines):
            p.setPen(line_color if line_color is not None else default_col)
            p.drawText(panel_x, panel_y + i * line_height + line_height - 2, line_text)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_slider(lo, hi, val, width, step=1):
    s = QSlider(Qt.Horizontal)
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    s.setFixedHeight(36)
    s.setFixedWidth(width)
    return s


def section_label(text):
    lbl = QLabel(text)
    lbl.setStyleSheet("color:#8888bb;font-size:26px;letter-spacing:1px;margin-top:8px;")
    return lbl


def make_button(text, accent="#aaaacc"):
    btn = QPushButton(text)
    btn.setFixedHeight(36)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setStyleSheet(f"""
QPushButton {{
    background:#1e1e3a; color:{accent};
    border:1px solid #3a3a60; border-radius:4px;
    font-family:'Courier New',monospace;
    font-size:26px; letter-spacing:1px; padding:0 6px;
}}
QPushButton:hover {{
    background:#2a2a50; color:#ccccff;
    border-color:#5566cc;
}}
QPushButton:pressed {{
    background:#4455aa; color:#ffffff;
}}
""")
    return btn


def _sep():
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet("color:#333355;margin:6px 0;")
    return f


# ══════════════════════════════════════════════════════════════════════════════
# Main window and controls
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Double Pendulum — Control System")
        self.setGeometry(100, 100, 830, 560)
        self.setStyleSheet("""
QMainWindow,QWidget{background:#0d0d1e;color:#ccccdd;}
QLabel{font-family:'Courier New',monospace;font-size:24px;}
QSlider::groove:horizontal{
    height:4px;background:#2a2a4a;border-radius:2px;}
QSlider::handle:horizontal{
    background:#5566cc;border:1px solid #8899ee;
    width:14px;height:14px;margin:-5px 0;border-radius:7px;}
QSlider::sub-page:horizontal{background:#4455aa;border-radius:2px;}
""")

        self.canvas = PendulumCanvas()
        iw = max(80, int(self.height() * 0.50))

        self.sl_l1 = make_slider(40, 200, 120, iw)
        self.sl_l2 = make_slider(40, 200, 100, iw)
        self.sl_g = make_slider(1, 300, 98, iw)
        self.sl_pivot_x = make_slider(-150, 150, 0, iw)
        self.sl_aggr = make_slider(1, 100, 50, iw)

        self.lbl_l1 = QLabel()
        self.lbl_l2 = QLabel()
        self.lbl_g = QLabel()
        self.lbl_pivot_x = QLabel()
        self.lbl_aggr = QLabel()
        self._refresh_labels()

        self.sl_l1.valueChanged.connect(self._on_l1)
        self.sl_l2.valueChanged.connect(self._on_l2)
        self.sl_g.valueChanged.connect(self._on_g)
        self.sl_pivot_x.valueChanged.connect(self._on_pivot_x)
        self.sl_aggr.valueChanged.connect(self._on_aggr)

        self.btn_damp = make_button("DAMP", "#6699ff")
        self.btn_balance = make_button("BALANCE - DISABLED", "#888888")
        self.btn_balance.setEnabled(False)
        self.btn_zero_v = make_button("ZERO VELOCITY")
        self.btn_freset = make_button("FULL RESET")
        self.btn_toggle = make_button("→ SINGLE")
        self.btn_air = make_button("AIR RES: OFF", "#ff9966")

        self.btn_damp.clicked.connect(self._on_damp)
        self.btn_balance.clicked.connect(self._on_balance)
        self.btn_zero_v.clicked.connect(lambda: self.canvas.reset_velocity())
        self.btn_freset.clicked.connect(self._on_full_reset)
        self.btn_toggle.clicked.connect(self._on_toggle_mode)
        self.btn_air.clicked.connect(self._on_toggle_air)

        sb = QVBoxLayout()
        sb.setSpacing(4)

        sb.addWidget(section_label("ARM LENGTH 1"))
        sb.addWidget(self.lbl_l1)
        sb.addWidget(self.sl_l1)

        sb.addWidget(section_label("ARM LENGTH 2"))
        sb.addWidget(self.lbl_l2)
        sb.addWidget(self.sl_l2)

        sb.addWidget(_sep())
        sb.addWidget(section_label("GRAVITY (m/s²)"))
        sb.addWidget(self.lbl_g)
        sb.addWidget(self.sl_g)

        sb.addWidget(_sep())
        sb.addWidget(section_label("PIVOT X (disturbance)"))
        sb.addWidget(self.lbl_pivot_x)
        sb.addWidget(self.sl_pivot_x)

        sb.addWidget(_sep())
        sb.addWidget(section_label("AGGRESSIVENESS"))
        sb.addWidget(self.lbl_aggr)
        sb.addWidget(self.sl_aggr)

        sb.addWidget(_sep())
        sb.addWidget(section_label("CONTROL"))
        sb.addWidget(self.btn_damp)
        sb.addSpacing(3)
        sb.addWidget(self.btn_balance)

        sb.addWidget(_sep())
        sb.addWidget(section_label("UTIL"))
        sb.addWidget(self.btn_zero_v)
        sb.addSpacing(3)
        sb.addWidget(self.btn_freset)
        sb.addSpacing(3)
        sb.addWidget(self.btn_toggle)
        sb.addWidget(self.btn_air)
        sb.addStretch()

        hint = QLabel(
            "Drag the bobs to move\n"
            "the pendulums around\n\n"
            "DAMP → calm → idle\n"
            "BALANCE → damp →\n"
            " swing-up → balance\n\n"
            "Pivot-X slider: live\n"
            "disturbance while\n"
            "controller is active"
        )
        hint.setStyleSheet("color:#404060;font-size:20px;margin-top:10px;")
        sb.addWidget(hint)

        self.sb_widget = QWidget()
        self.sb_widget.setLayout(sb)
        self.sb_widget.setContentsMargins(8, 12, 8, 12)

        layout = QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        layout.addWidget(self.sb_widget)

        c = QWidget()
        c.setLayout(layout)
        self.setCentralWidget(c)
        self._update_slider_widths()

        self._dt = 1/240
        self.timer = QTimer()
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.step)
        self.timer.start()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._update_slider_widths()

    def _update_slider_widths(self):
        w = max(80, int(self.height() * 0.50))
        for sl in (self.sl_l1, self.sl_l2, self.sl_g, self.sl_pivot_x, self.sl_aggr):
            sl.setFixedWidth(w)
        self.sb_widget.setFixedWidth(w + 32)

    def step(self):
        for _ in range(4):
            self.canvas.advance(self._dt)

        # Sync buttons with state
        self._sync_air_button()
        if self.canvas.ctrl_mode == Mode.DAMPING:
            if self.canvas._after_damp == Mode.SWING_UP:
                self._sync_control_buttons("balance")
            else:
                self._sync_control_buttons("damp")
        else:
            self._sync_control_buttons()

        self.canvas.update()

    def _physics_changed(self):
        self.canvas._lqr_dirty = True
        self.canvas.trail.clear()
        self._refresh_labels()

    def _on_l1(self, v):
        self.canvas.length1 = float(v)
        self._physics_changed()

    def _on_l2(self, v):
        self.canvas.length2 = float(v)
        self._physics_changed()

    def _on_g(self, v):
        self.canvas.gravity = v / 10.0
        self._physics_changed()

    def _on_pivot_x(self, v):
        self._move_pivot(dx=v - self.canvas.pivot_x_manual, dy=0)
        self.canvas.pivot_x_manual = v
        self.canvas.trail.clear()
        self._refresh_labels()

    def _on_aggr(self, v):
        self.canvas._aggressiveness = v
        self.canvas._lqr_dirty = True
        self._refresh_labels()

    def _move_pivot(self, dx: float, dy: float):
        if dx == 0 and dy == 0:
            return
        ox, oy, x1, y1, x2, y2 = self.canvas._positions()
        new_ox = ox + dx
        new_oy = oy + dy
        self.canvas.state[0] = math.atan2(x1 - new_ox, y1 - new_oy)
        nx1 = new_ox + self.canvas.length1 * math.sin(self.canvas.state[0])
        ny1 = new_oy + self.canvas.length1 * math.cos(self.canvas.state[0])
        self.canvas.state[2] = math.atan2(x2 - nx1, y2 - ny1)

    def _refresh_labels(self):
        c = self.canvas
        self.lbl_l1.setText(f"{c.length1:.0f} px")
        self.lbl_l2.setText(f"{c.length2:.0f} px")
        self.lbl_g.setText(f"{c.gravity:.2f} m/s²")
        sx = "+" if c.pivot_x_manual >= 0 else ""
        self.lbl_pivot_x.setText(f"{sx}{c.pivot_x_manual} px")
        self.lbl_aggr.setText(f"{c._aggressiveness}")

    def _sync_air_button(self):
        if self.canvas.air_resistance:
            self.btn_air.setText("AIR RES: ON")
            self.btn_air.setStyleSheet(self.btn_air.styleSheet().replace("color:#ff9966", "color:#44ee88"))
        else:
            self.btn_air.setText("AIR RES: OFF")
            self.btn_air.setStyleSheet(self.btn_air.styleSheet().replace("color:#44ee88", "color:#ff9966"))

    def _sync_control_buttons(self, active=None):
        # active: None, "damp", or "balance"
        if not self.canvas.single_mode:
            self.btn_damp.setText("DAMP - ENABLED" if active == "damp" else "DAMP")
            self.btn_balance.setText("BALANCE - DISABLED")
            self.btn_balance.setEnabled(False)
            return
        if active == "damp":
            self.btn_damp.setText("DAMP - ENABLED")
            self.btn_balance.setText("BALANCE")
        elif active == "balance":
            self.btn_damp.setText("DAMP")
            self.btn_balance.setText("BALANCE - ENABLED")
        else:
            self.btn_damp.setText("DAMP")
            self.btn_balance.setText("BALANCE")

    def _on_damp(self):
        c = self.canvas
        c.air_resistance = True
        c._pending_air_off = False
        self._sync_air_button()
        self._sync_control_buttons("damp")
        c._after_damp = Mode.IDLE
        c.ctrl_mode = Mode.DAMPING

    def _on_balance(self):
        c = self.canvas
        if not c.single_mode:
            return

        c.air_resistance = True
        c._pending_air_off = False
        self._sync_air_button()
        self._sync_control_buttons("balance")

        # Balance always starts by damping first, then swings up, then balances.
        arm1_m = c.length1 / PX_PER_M
        arm2_m = c.length2 / PX_PER_M
        c.K = compute_lqr(arm1_m, arm2_m, c.m1, c.m2, c.gravity, c._aggressiveness, c.single_mode)
        c._lqr_dirty = False
        c._after_damp = Mode.SWING_UP
        c.ctrl_mode = Mode.DAMPING

    def _on_full_reset(self):
        self.canvas.full_reset()
        self._sync_air_button()
        self._sync_control_buttons()

    def _on_toggle_mode(self):
        self.canvas.toggle_single_mode()
        self._sync_control_buttons()
        if self.canvas.single_mode:
            self.btn_toggle.setText("→ DOUBLE")
            self.lbl_l2.setEnabled(False)
            self.sl_l2.setEnabled(False)
            self.btn_balance.setEnabled(True)
            self.btn_balance.setText("BALANCE")
        else:
            self.btn_toggle.setText("→ SINGLE")
            self.lbl_l2.setEnabled(True)
            self.sl_l2.setEnabled(True)
            self.btn_balance.setEnabled(False)
            self.btn_balance.setText("BALANCE - DISABLED")

    def _on_toggle_air(self):
        c = self.canvas
        turning_off = c.air_resistance

        if turning_off and c.ctrl_mode == Mode.DAMPING:
            # Only balance flow is allowed to defer the air-off until damping completes.
            if c._after_damp == Mode.SWING_UP:
                c._pending_air_off = True
                QMessageBox.warning(
                    self,
                    "Air Resistance Required",
                    "Damping cannot occur without air resistance."
                    "Air resistance will turn off automatically after damping is complete."
                )
            else:
                c._pending_air_off = False
                QMessageBox.warning(
                    self,
                    "Air Resistance Required",
                    "Damping cannot occur without air resistance."
                )

            c.air_resistance = True
            self._sync_air_button()
            return

        c.air_resistance = not c.air_resistance
        c._pending_air_off = False
        self._sync_air_button()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Courier New", 22))
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())