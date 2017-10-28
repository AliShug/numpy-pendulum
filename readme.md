# Pendulum Simulation
Here I've implemented a 3D multi-link, rigid-body pendulum simulation from the Newton-Euler equations of motion, using explicit Euler integration, rotational damping, and Baumgarte stabilization of the hinge constraints. 1-8 links are combined into a single matrix representing the linear system of motion and constraints, which is solved using Numpy's `linalg.solve` function. The link's 3D rotations are stored and manipulated using quaternions.

## Instructions
Requires:
* Python (I use 3.5)
* Numpy
* Pyquaternion
* PyOpenGL
* PyOpenGL-accelerate
* Matplotlib

Run:
`python sim.py [-t|--track]`

Options:
* -t/--track: Enable energy tracking

Controls:
* 'r' : reset the simulation
* '+' : add a link - up to 8
* '-' : remove a link - minimum 1
* 's' : plot energy over time (**requires tracking**)
* 'q/ESC' : quit
