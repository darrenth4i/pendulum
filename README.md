# Pendulum

Physics simulation of single and double pendulums, with a state-machine controller for autonomous swing-up and upright balancing.

## Controller

The single-pendulum balancing controller is a state machine with three modes:

| Mode | Method | Role |
|---|---|---|
| Damping | PD control | Dissipates energy when the pendulum overshoots the target |
| Swing-up | Åström–Furuta energy pump | Injects energy to drive the pendulum toward the upright position |
| Balancing | Full-state LQR | Stabilizes the pendulum once it reaches the upright equilibrium |

The controller transitions automatically between modes based on the pendulum's current energy and angle.

## Requirements

- Python 3.x
- [SciPy](https://scipy.org/)
- [PyQt5](https://pypi.org/project/PyQt5/)

Install dependencies with:

```bash
pip install scipy PyQt5
```

## Usage

```bash
python pendulum.py
```

## References

K. J. Åström and K. Furuta, "Swinging up a pendulum by energy control," *Automatica*, 2000.
https://web.ece.ucsb.edu/~hespanha/ece229/references/AstromFurutaAUTOM00.pdf
