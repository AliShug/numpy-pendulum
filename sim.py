from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys

#  from pyquaternion import Quaternion    # would be useful for 3D simulation
import numpy as np


def funkify(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class Link(object):
    def __init__(self):
        self.color = [0, 0, 0]  # draw color
        self.size = [1, 1, 1]  # dimensions
        self.mass = 1.0  # mass in kg
        self.inertia = np.identity(3)
        self.theta = 0  # 2D orientation  (will need to change for 3D)
        self.omega = np.array([0.0, 0.0, 0.0])
        self.pos = np.array([0.0, 0.0, 0.0])
        self.vel = np.array([0.0, 0.0, 0.0])

        self.display_force = np.array([0.0, 0.0, 0.0])

    def draw(self):  # steps to draw a link
        glPushMatrix()  # save copy of coord frame
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])  # move
        glRotatef(self.theta * (180/np.pi), 0, 0, 1)  # rotate
        glScale(self.size[0], self.size[1], self.size[2])  # set size
        glColor3f(self.color[0], self.color[1], self.color[2])  # set colour
        self.draw_cube()  # draw a scaled cube
        glPopMatrix()  # restore old coord frame
        glBegin(GL_LINES)
        glVertex3fv(self.pos + self.get_r())
        glVertex3fv(self.pos + self.get_r() + self.display_force)
        glEnd()

    def set_cuboid(self, mass, w, h, d):
        """Creates a cuboid of the specified width, depth and height (x, y, z respectively)"""
        self.mass = mass
        self.inertia = mass/12 * np.array([[h**2+d**2, 0, 0], [0, w**2+d**2, 0], [0, 0, w**2+h**2]])
        self.size = np.array([w, h, d])

    def get_r(self):
        return (self.size[1]/2)*np.array([-np.sin(self.theta), np.cos(self.theta), 0])

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
        self.window = 0  # number of the glut window
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
        self.kp = 1.0
        self.kd = 0.5
        self.damp = 0.2
        self.links = []
        self.sim_running = False
        self.sim_time = 0.0

    def main(self):
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)  # display mode
        glutInitWindowSize(640, 480)  # window size
        glutInitWindowPosition(0, 0)  # window coords for mouse start at top-left
        self.window = glutCreateWindow(b"CPSC 526 Simulation Template")
        glutDisplayFunc(self.draw_world)  # register the function to draw the world
        # glutFullScreen()               # full screen
        glutIdleFunc(self.simulate_world)  # when doing nothing, redraw the scene
        glutReshapeFunc(self.resize_gl_scene)  # register the function to call when window is resized
        glutKeyboardFunc(self.key_pressed)  # register the function to call when keyboard is pressed
        self.init_gl(640, 480)  # initialize window

        self.reset_sim(2)

        glutMainLoop()  # start event processing loop

    def reset_sim(self, num_links):
        print("Simulation reset")
        self.sim_running = True
        self.sim_time = 0

        colors = ([0.5, 0.5, 0.5], [0.9, 0.9, 0.9])
        angle = np.pi/4

        # clear existing links
        self.links = []

        # first link
        # links MUST start at rest
        link = Link()
        link.set_cuboid(self.link_mass, self.link_thickness, self.link_length, self.link_thickness)
        link.color = colors[0]
        link.theta = angle
        link.pos = self.anchor - link.get_r()
        self.links.append(link)
        print("pos 0", link.pos)

        for i in range(1, num_links):
            link = Link()
            prev_link = self.links[i-1]
            link.set_cuboid(self.link_mass, self.link_thickness, self.link_length, self.link_thickness)
            link.color = colors[i % 2]
            link.theta = angle
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
            self.reset_sim(8)

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
                m = link.mass * np.identity(3)
                mat[i*6:i*6+3, i*6:i*6+3] = m
                rhs[i*6:i*6+3] = m @ grav
                ir = link.inertia
                mat[i*6+3:i*6+6, i*6+3:i*6+6] = ir
                w = link.omega
                r = link.get_r()
                r_s = funkify(r)
                rhs[i*6+3:i*6+6] = -np.cross(w, ir@w) - self.damp*w
                if i == 0:
                    # first link has space-anchored constraint
                    pos_drift = (link.pos + r) - self.anchor
                    vel_drift = (link.vel + np.cross(w, r))
                    mat[0:3, offs+i*3:offs+i*3+3] = -np.identity(3)
                    mat[3:6, offs+i*3:offs+i*3+3] = -r_s
                    # first constraint
                    mat[offs+i*3:offs+i*3+3, 0:3] = -np.identity(3)
                    mat[offs+i*3:offs+i*3+3, 3:6] = r_s
                    rhs[offs+i*3:offs+i*3+3] =\
                        np.cross(w, np.cross(w, r)) +\
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
                    rhs[offs+i*3:offs+i*3+3] =\
                        np.cross(wa, np.cross(wa, ra)) - np.cross(w, np.cross(w, r)) -\
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
                link.theta += link.omega[2] * self.dT
                link.omega += w_dot * self.dT
            self.sim_time += self.dT

        # draw the updated state
        self.draw_world()
        print("simTime=%.2f" % self.sim_time)

        # print("Diff from anchor: ", self.links[0].pos + self.links[0].get_r() - self.anchor)

    def draw_world(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear The Screen And The Depth Buffer
        glLoadIdentity()
        gluLookAt(1, 1, 3, 0, 0, 0, 0, 1, 0)

        self.draw_origin()
        for link in self.links:
            link.draw()

        glutSwapBuffers()  # swap the buffers to display what was just drawn

    @staticmethod
    def init_gl(width, height):  # We call this right after our OpenGL window is created.
        glClearColor(1.0, 1.0, 0.9, 0.0)  # This Will Clear The Background Color To Black
        glClearDepth(1.0)  # Enables Clearing Of The Depth Buffer
        glDepthFunc(GL_LESS)  # The Type Of Depth Test To Do
        glEnable(GL_DEPTH_TEST)  # Enables Depth Testing
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glShadeModel(GL_SMOOTH)  # Enables Smooth Color Shading
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


print("Hit ESC key to quit.")
sim = Sim()
sim.main()
