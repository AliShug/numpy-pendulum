from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys
import argparse

import matplotlib.pyplot as plt

from pyquaternion import Quaternion
import numpy as np


def funkify(v):
    """Returns a skew-symmetric matrix M for input vector v such that cross(v, k) = M @ k"""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class Link(object):
    def __init__(self):
        self.color = [0, 0, 0]
        self.size = [1, 1, 1]
        self.mass = 1.0
        self.inertia = np.identity(3)
        self.q_rot = Quaternion()
        self.omega = np.array([0.0, 0.0, 0.0])
        self.pos = np.array([0.0, 0.0, 0.0])
        self.vel = np.array([0.0, 0.0, 0.0])

        self.display_force = np.array([0.0, 0.0, 0.0])

    def draw(self):
        """Render the link with OpenGL"""
        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])
        glMultMatrixf(self.q_rot.transformation_matrix.T)
        glScale(self.size[0], self.size[1], self.size[2])
        glColor3f(self.color[0], self.color[1], self.color[2])
        self.draw_cube()
        glPopMatrix()
        glBegin(GL_LINES)
        glVertex3fv(self.pos + self.get_r())
        glVertex3fv(self.pos + self.get_r() + self.display_force)
        glEnd()

    def set_cuboid(self, mass, w, h, d):
        """Initializes link to a cuboid of the specified mass width, depth and height (x, y, z respectively)"""
        self.mass = mass
        self.inertia = mass/12 * np.array([[h**2+d**2, 0, 0], [0, w**2+d**2, 0], [0, 0, w**2+h**2]])
        self.size = np.array([w, h, d])

    def get_r(self):
        """Return the world-space vector from the link's center to its upper hinge joint"""
        # return (self.size[1]/2)*np.array([-np.sin(self.theta), np.cos(self.theta), 0])
        return self.q_rot.rotation_matrix @ np.array([0, self.size[1]/2, 0])

    @staticmethod
    def draw_cube():
        glScalef(0.5, 0.5, 0.5)  # dimensions below are for a 2x2x2 cube, so scale it down by a half first
        glBegin(GL_QUADS)  # Start Drawing The Cube

        glVertex3f(1.0, 1.0, -1.0)  # Top Right Of The Quad (Top)
        glVertex3f(-1.0, 1.0, -1.0)  # Top Left Of The Quad (Top)
        glVertex3f(-1.0, 1.0, 1.0)  # Bottom Left Of The Quad (Top)
        glVertex3f(1.0, 1.0, 1.0)  # Bottom Right Of The Quad (Top)

        glVertex3f(1.0, -1.0, 1.0)  # Top Right Of The Quad (Bottom)
        glVertex3f(-1.0, -1.0, 1.0)  # Top Left Of The Quad (Bottom)
        glVertex3f(-1.0, -1.0, -1.0)  # Bottom Left Of The Quad (Bottom)
        glVertex3f(1.0, -1.0, -1.0)  # Bottom Right Of The Quad (Bottom)

        glVertex3f(1.0, 1.0, 1.0)  # Top Right Of The Quad (Front)
        glVertex3f(-1.0, 1.0, 1.0)  # Top Left Of The Quad (Front)
        glVertex3f(-1.0, -1.0, 1.0)  # Bottom Left Of The Quad (Front)
        glVertex3f(1.0, -1.0, 1.0)  # Bottom Right Of The Quad (Front)

        glVertex3f(1.0, -1.0, -1.0)  # Bottom Left Of The Quad (Back)
        glVertex3f(-1.0, -1.0, -1.0)  # Bottom Right Of The Quad (Back)
        glVertex3f(-1.0, 1.0, -1.0)  # Top Right Of The Quad (Back)
        glVertex3f(1.0, 1.0, -1.0)  # Top Left Of The Quad (Back)

        glVertex3f(-1.0, 1.0, 1.0)  # Top Right Of The Quad (Left)
        glVertex3f(-1.0, 1.0, -1.0)  # Top Left Of The Quad (Left)
        glVertex3f(-1.0, -1.0, -1.0)  # Bottom Left Of The Quad (Left)
        glVertex3f(-1.0, -1.0, 1.0)  # Bottom Right Of The Quad (Left)

        glVertex3f(1.0, 1.0, -1.0)  # Top Right Of The Quad (Right)
        glVertex3f(1.0, 1.0, 1.0)  # Top Left Of The Quad (Right)
        glVertex3f(1.0, -1.0, 1.0)  # Bottom Left Of The Quad (Right)
        glVertex3f(1.0, -1.0, -1.0)  # Bottom Right Of The Quad (Right)
        glEnd()  # Done Drawing The Quad

        # Draw the wireframe edges
        glColor3f(0.0, 0.0, 0.0)
        glLineWidth(1.0)

        glBegin(GL_LINE_LOOP)
        glVertex3f(1.0, 1.0, -1.0)  # Top Right Of The Quad (Top)
        glVertex3f(-1.0, 1.0, -1.0)  # Top Left Of The Quad (Top)
        glVertex3f(-1.0, 1.0, 1.0)  # Bottom Left Of The Quad (Top)
        glVertex3f(1.0, 1.0, 1.0)  # Bottom Right Of The Quad (Top)
        glEnd()  # Done Drawing The Quad

        glBegin(GL_LINE_LOOP)
        glVertex3f(1.0, -1.0, 1.0)  # Top Right Of The Quad (Bottom)
        glVertex3f(-1.0, -1.0, 1.0)  # Top Left Of The Quad (Bottom)
        glVertex3f(-1.0, -1.0, -1.0)  # Bottom Left Of The Quad (Bottom)
        glVertex3f(1.0, -1.0, -1.0)  # Bottom Right Of The Quad (Bottom)
        glEnd()  # Done Drawing The Quad

        glBegin(GL_LINE_LOOP)
        glVertex3f(1.0, 1.0, 1.0)  # Top Right Of The Quad (Front)
        glVertex3f(-1.0, 1.0, 1.0)  # Top Left Of The Quad (Front)
        glVertex3f(-1.0, -1.0, 1.0)  # Bottom Left Of The Quad (Front)
        glVertex3f(1.0, -1.0, 1.0)  # Bottom Right Of The Quad (Front)
        glEnd()  # Done Drawing The Quad

        glBegin(GL_LINE_LOOP)
        glVertex3f(1.0, -1.0, -1.0)  # Bottom Left Of The Quad (Back)
        glVertex3f(-1.0, -1.0, -1.0)  # Bottom Right Of The Quad (Back)
        glVertex3f(-1.0, 1.0, -1.0)  # Top Right Of The Quad (Back)
        glVertex3f(1.0, 1.0, -1.0)  # Top Left Of The Quad (Back)
        glEnd()  # Done Drawing The Quad

        glBegin(GL_LINE_LOOP)
        glVertex3f(-1.0, 1.0, 1.0)  # Top Right Of The Quad (Left)
        glVertex3f(-1.0, 1.0, -1.0)  # Top Left Of The Quad (Left)
        glVertex3f(-1.0, -1.0, -1.0)  # Bottom Left Of The Quad (Left)
        glVertex3f(-1.0, -1.0, 1.0)  # Bottom Right Of The Quad (Left)
        glEnd()  # Done Drawing The Quad

        glBegin(GL_LINE_LOOP)
        glVertex3f(1.0, 1.0, -1.0)  # Top Right Of The Quad (Right)
        glVertex3f(1.0, 1.0, 1.0)  # Top Left Of The Quad (Right)
        glVertex3f(1.0, -1.0, 1.0)  # Bottom Left Of The Quad (Right)
        glVertex3f(1.0, -1.0, -1.0)  # Bottom Right Of The Quad (Right)
        glEnd()  # Done Drawing The Quad


class Sim(object):
    def __init__(self):
        self.window = 0
        self.theta = 0.0
        self.sim_time = 0
        self.dT = 0.003
        self.sim_running = True
        self.RAD_TO_DEG = 180.0 / np.pi
        self.GRAVITY = -9.81
        self.anchor = np.array([0.0, 1.0, 0.0])
        self.link_length = 0.5
        self.link_thickness = 0.04
        self.link_mass = 1.0
        self.kp = 20.0
        self.kd = 1.0
        self.cp = 4000.0
        self.cd = 50.0
        self.damp = 0.08
        self.plane = True
        self.plane_height = 0.0
        self.links = []
        self.sim_running = False
        self.sim_time = 0.0

        self.args = None
        self.energies = []
        self.potentials = []
        self.times = []

    def plot_energies(self):
        plt.plot(self.times, self.energies)
        plt.plot(self.times, self.potentials)
        totals = np.array(self.energies) + self.potentials
        plt.plot(self.times, totals)
        plt.legend(('E', 'PE', 'Total'), loc='best')
        plt.ylabel('energy')
        plt.xlabel('time (s)')
        plt.show()

    def main(self):
        # parse args
        parser = argparse.ArgumentParser()
        parser.add_argument("-t", "--track", action="store_true")
        self.args = parser.parse_args()
        # set up the GLUT window
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)  # display mode
        glutInitWindowSize(640, 480)
        glutInitWindowPosition(0, 0)
        self.window = glutCreateWindow(b"CPSC 526 Simulation Template")
        glutDisplayFunc(self.draw_world)
        glutIdleFunc(self.simulate_world)
        glutReshapeFunc(self.resize_gl_scene)
        glutKeyboardFunc(self.key_pressed)
        self.init_gl(640, 480)
        # initialize the simulation
        self.reset_sim(2)
        # event processing loop
        glutMainLoop()

    def reset_sim(self, num_links):
        print("Simulation reset")
        self.sim_running = True
        self.sim_time = 0

        colors = ([0.5, 0.5, 0.5], [0.9, 0.9, 0.9])
        angle = np.pi/2
        axis = np.array([1, 0, 1])
        axis = axis / np.linalg.norm(axis)

        # clear existing links
        self.links = []
        # clear stored energies
        self.energies = []
        self.potentials = []
        self.times = []

        # first link
        # links MUST start at rest
        link = Link()
        link.set_cuboid(self.link_mass, self.link_thickness, self.link_length, self.link_thickness)
        link.color = colors[0]
        link.q_rot = Quaternion(axis=axis, angle=angle)
        link.pos = self.anchor - link.get_r()
        self.links.append(link)
        print("pos 0", link.pos)

        for i in range(1, num_links):
            link = Link()
            prev_link = self.links[i-1]
            link.set_cuboid(self.link_mass, self.link_thickness, self.link_length, self.link_thickness)
            link.color = colors[i % 2]
            link.q_rot = Quaternion(axis=axis, angle=angle)
            link.pos = prev_link.pos - prev_link.get_r() - link.get_r()
            self.links.append(link)
            print("pos ", i, link.pos)
        print("anchor ", self.anchor)

    def key_pressed(self, key, x, y):
        ch = key.decode("utf-8")
        if ch == ' ':  # toggle the simulation
            if self.sim_running:
                self.sim_running = False
            else:
                self.sim_running = True
        elif ch == chr(27):  # ESC key
            sys.exit()
        elif ch == 'q':  # quit
            sys.exit()
        elif ch == 'r':  # reset simulation
            self.reset_sim(len(self.links))
        elif ch == '+':
            n = len(self.links)
            n2 = min(8, n+1)
            if n2 != n:
                self.reset_sim(n2)
        elif ch == '-':
            n = len(self.links)
            n2 = max(1, n-1)
            if n2 != n:
                self.reset_sim(n2)
        elif ch == 's' and self.args.track:
            # display tracked energies over time
            self.plot_energies()
        elif ch == 'p':
            self.plane = not self.plane

    def simulate_world(self):
        if not self.sim_running:  # is simulation stopped?
            return

        # solve for the equations of motion (simple in this case!)
        grav = np.array([0, self.GRAVITY, 0])  # linear acceleration = [0, -G, 0]

        # pre-allocate big matrix
        dim = len(self.links) * 9
        mat = np.zeros((dim, dim))
        rhs = np.zeros(dim)
        offs = len(self.links) * 6

        # simulate several physics steps for each drawing update
        for step in range(10):
            # iterate the links
            for i in range(len(self.links)):
                link = self.links[i]
                if i == 0:
                    anchor = None
                else:
                    anchor = self.links[i-1]
                w = link.omega
                r = link.get_r()
                # penalty force (ground plane)
                if self.plane:
                    pt = link.pos - r
                    vel = link.vel + np.cross(w, -r)
                    fp = max(0, self.cp*(self.plane_height - pt[1]) - self.cd*vel[1])
                else:
                    fp = 0.0
                m = link.mass * np.identity(3)
                mat[i*6:i*6+3, i*6:i*6+3] = m
                rhs[i*6:i*6+3] = m @ grav + [0, fp, 0]
                rot = link.q_rot.rotation_matrix
                ir = rot @ link.inertia @ rot.T
                mat[i*6+3:i*6+6, i*6+3:i*6+6] = ir
                r_s = funkify(r)
                rhs[i*6+3:i*6+6] = -np.cross(w, ir@w) - self.damp*w - np.cross(r, [0, fp, 0])
                if i == 0:
                    # first link has space-anchored constraint
                    pos_drift = (link.pos + r) - self.anchor
                    vel_drift = (link.vel + np.cross(w, r))
                    mat[0:3, offs+i*3:offs+i*3+3] = -np.identity(3)
                    mat[3:6, offs+i*3:offs+i*3+3] = -r_s
                    # first constraint
                    mat[offs+i*3:offs+i*3+3, 0:3] = -np.identity(3)
                    mat[offs+i*3:offs+i*3+3, 3:6] = r_s
                    rhs[offs+i*3:offs+i*3+3] = \
                        np.cross(w, np.cross(w, r)) + \
                        self.kp*pos_drift + self.kd*vel_drift
                else:
                    # subsequent links anchor to the previous link in the chain
                    ra = -anchor.get_r()
                    ra_s = funkify(ra)
                    wa = anchor.omega
                    pos_drift = (link.pos + r) - (anchor.pos + ra)
                    vel_drift = (link.vel + np.cross(w, r)) - (anchor.vel + np.cross(wa, ra))
                    mat[i*6-6:i*6-3, offs+i*3:offs+i*3+3] = np.identity(3)
                    mat[i*6-3:i*6, offs+i*3:offs+i*3+3] = ra_s
                    mat[i*6:i*6+3, offs+i*3:offs+i*3+3] = -np.identity(3)
                    mat[i*6+3:i*6+6, offs+i*3:offs+i*3+3] = -r_s
                    # intermediate constraints
                    mat[offs+i*3:offs+i*3+3, i*6-6:i*6-3] = -np.identity(3)
                    mat[offs+i*3:offs+i*3+3, i*6-3:i*6] = ra_s
                    mat[offs+i*3:offs+i*3+3, i*6:i*6+3] = np.identity(3)
                    mat[offs+i*3:offs+i*3+3, i*6+3:i*6+6] = -r_s
                    rhs[offs+i*3:offs+i*3+3] = \
                        np.cross(wa, np.cross(wa, ra)) - np.cross(w, np.cross(w, r)) - \
                        self.kp*pos_drift - self.kd*vel_drift

            # solve
            results = np.linalg.solve(mat, rhs)
            # update links
            for i in range(len(self.links)):
                link = self.links[i]
                acc = results[i*6:i*6+3]
                w_dot = results[i*6+3:i*6+6]
                # explicit euler integration
                link.pos += link.vel * self.dT
                link.vel += acc * self.dT
                w_mag = np.linalg.norm(link.omega)
                if w_mag != 0.0 and w_mag < 1000000 and not np.isnan(w_mag):
                    axis = link.omega / w_mag
                    link.q_rot *= Quaternion(axis=axis, angle=w_mag*self.dT)  # link.omega[2] * self.dT
                link.omega += w_dot * self.dT

            # track values over time
            if self.args.track:
                energy = self.sum_energy()
                self.energies.append(energy[0])
                self.potentials.append(energy[1])
                self.times.append(self.sim_time)

            self.sim_time += self.dT


        # draw the updated state
        self.draw_world()
        energy = self.sum_energy()
        print("t=%.2f E=%.2f, total energy=%.2f" % (self.sim_time, energy[0], energy[1]+energy[0]))

        # print("Diff from anchor: ", self.links[0].pos + self.links[0].get_r() - self.anchor)

    def sum_energy(self):
        energy = 0.0
        pe = 0.0
        # track minimum reachable height for each link, including ground plane
        min_height = self.anchor[1] - 0.5*self.link_length
        if self.plane:
            min_height = max(self.plane_height, min_height)
        for link in self.links:
            energy += 0.5*link.mass*np.linalg.norm(link.vel)**2
            # energy += 0.5*link.inertia[2, 2]*link.omega[2]**2
            w = link.omega
            ir = link.inertia
            energy += 0.5*(ir[0, 0]*w[0]**2 + ir[1, 1]*w[1]**2 + ir[2, 2]*w[2]**2)
            pe += link.mass*(-self.GRAVITY)*(link.pos[1] - min_height)
            min_height -= self.link_length
            if self.plane:
                min_height = max(self.plane_height, min_height)
        return energy, pe

    def draw_world(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(1, 1, 3, 0, 0, 0, 0, 1, 0)

        self.draw_origin()
        for link in self.links:
            link.draw()

        glutSwapBuffers()

    @staticmethod
    def init_gl(width, height):
        glClearColor(1.0, 1.0, 1.0, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()  # Reset The Projection Matrix
        gluPerspective(45.0, float(width) / float(height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    @staticmethod
    def resize_gl_scene(width, height):
        if height == 0:  # Prevent A Divide By Zero If The Window Is Too Small
            height = 1
        glViewport(0, 0, width, height)  # Reset The Current Viewport And Perspective Transformation
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(width) / float(height), 0.1,
                       100.0)  # 45 deg horizontal field of view, aspect ratio, near, far
        glMatrixMode(GL_MODELVIEW)

    @staticmethod
    def draw_origin():
        glLineWidth(3.0)

        glColor3f(1, 0.5, 0.5)  # light red x-axis
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(1, 0, 0)
        glEnd()

        glColor3f(0.5, 1, 0.5)  # light green y-axis
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 1, 0)
        glEnd()

        glColor3f(0.5, 0.5, 1)  # light blue z-axis
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 1)
        glEnd()


print("Hit ESC/q to quit, r to reset, + and - to add or remove links (resetting the simulation).")
sim = Sim()
sim.main()
