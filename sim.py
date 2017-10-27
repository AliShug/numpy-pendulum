from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys

#  from pyquaternion import Quaternion    # would be useful for 3D simulation
import numpy as np

window = 0  # number of the glut window
theta = 0.0
sim_time = 0
dT = 0.01
sim_running = True
RAD_TO_DEG = 180.0 / np.pi
GRAVITY = -9.81
link1 = link2 = None
kp = 0.0
kd = 0.0


class Link(object):
    color = [0, 0, 0]  # draw color
    size = [1, 1, 1]  # dimensions
    mass = 1.0  # mass in kg
    inertia = np.identity(3)
    theta = 0  # 2D orientation  (will need to change for 3D)
    omega = 0  # 2D angular velocity
    pos = np.array([0.0, 0.0, 0.0])  # 3D position (keep z=0 for 2D)
    vel = np.array([0.0, 0.0, 0.0])  # initial velocity

    display_force = np.array([0.0, 0.0, 0.0])

    def draw(self):  # steps to draw a link
        glPushMatrix()  # save copy of coord frame
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])  # move
        glRotatef(self.theta * RAD_TO_DEG, 0, 0, 1)  # rotate
        glScale(self.size[0], self.size[1], self.size[2])  # set size
        glColor3f(self.color[0], self.color[1], self.color[2])  # set colour
        draw_cube()  # draw a scaled cube
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


def main():
    global window
    global link1, link2
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)  # display mode
    glutInitWindowSize(640, 480)  # window size
    glutInitWindowPosition(0, 0)  # window coords for mouse start at top-left
    window = glutCreateWindow(b"CPSC 526 Simulation Template")
    glutDisplayFunc(draw_world)  # register the function to draw the world
    # glutFullScreen()               # full screen
    glutIdleFunc(simulate_world)  # when doing nothing, redraw the scene
    glutReshapeFunc(resize_gl_scene)  # register the function to call when window is resized
    glutKeyboardFunc(key_pressed)  # register the function to call when keyboard is pressed
    init_gl(640, 480)  # initialize window

    link1 = Link()
    link2 = Link()
    reset_sim()

    glutMainLoop()  # start event processing loop


def reset_sim():
    global link1, link2
    global sim_time, sim_running

    print("Simulation reset")
    sim_running = True
    sim_time = 0

    link1.set_cuboid(1.0, 0.04, 1.0, 0.12)
    link1.color = [1, 0.9, 0.9]
    link1.pos = np.array([0.0, 0.0, 0.0])
    link1.vel = np.array([0.0, 0.0, 0.0])
    link1.theta = np.pi/2
    link1.omega = np.array([0., 0., 0.])

    link2.set_cuboid(1.0, 0.04, 1.0, 0.12)
    link2.color = [0.9, 0.9, 1.0]
    link2.pos = np.array([1.0, 0.0, 0.0])
    link2.vel = np.array([0.0, 0.0, 0.0])
    link2.theta = np.pi/2
    link2.omega = np.array([0., 0., 0.0])  # radians per second


def key_pressed(key, x, y):
    global sim_running
    ch = key.decode("utf-8")
    if ch == ' ':  # toggle the simulation
        if sim_running:
            sim_running = False
        else:
            sim_running = True
    elif ch == chr(27):  # ESC key
        sys.exit()
    elif ch == 'q':  # quit
        sys.exit()
    elif ch == 'r':  # reset simulation
        reset_sim()


def funkify(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def simulate_world():
    global sim_time, dT, sim_running
    global link1, link2
    global GRAVITY
    global kp, kd

    if not sim_running:  # is simulation stopped?
        return

    # solve for the equations of motion (simple in this case!)
    grav = np.array([0, GRAVITY, 0])  # linear acceleration = [0, -G, 0]

    # construct dynamics matrix
    M1 = link1.mass * np.identity(3)
    M2 = link2.mass * np.identity(3)
    M1_g = M1 @ grav
    M2_g = M2 @ grav
    I1 = link1.inertia
    I2 = link2.inertia
    w1 = link1.omega
    w2 = link2.omega
    r0 = link1.get_r()
    r1 = -r0
    r2 = link2.get_r()
    funky_r0 = funkify(r0)
    funky_r1 = funkify(r1)
    funky_r2 = funkify(r2)
    omega_I1 = np.cross(-w1, I1@w1)
    omega_I2 = np.cross(-w2, I2@w2)
    omega_r0 = np.cross(w1, np.cross(w1, r0))
    # pos_drift = (link1.pos + r1) - (link2.pos + r2)
    # vel_drift = (link1.vel + np.cross(w1, r1)) - (link2.vel + np.cross(w2, r2))
    constraint = np.cross(w1, np.cross(w1, r1)) - np.cross(w2, np.cross(w2, r2))# - kp*pos_drift - kd*vel_drift
    dim = 18
    mat = np.zeros((dim, dim))
    mat[0:3, 0:3] = M1
    mat[0:3, 12:15] = -np.identity(3)
    # mat[0:3, 15:18] = -np.identity(3)
    mat[3:6, 3:6] = I1
    mat[3:6, 12:15] = -funky_r0
    # mat[3:6, 15:18] = -funky_r1
    mat[6:9, 6:9] = M2
    mat[6:9, 15:18] = np.identity(3)
    mat[9:12, 9:12] = I2
    mat[9:12, 15:18] = funky_r2
    mat[12:15, 0:3] = -np.identity(3)
    mat[12:15, 3:6] = funky_r0
    # mat[15:18, 0:3] = -np.identity(3)
    # mat[15:18, 3:6] = funky_r1
    mat[15:18, 6:9] = np.identity(3)
    mat[15:18, 9:12] = -funky_r2
    # solve
    rhs = np.concatenate([M1_g, omega_I1, M2_g, omega_I2, omega_r0, constraint])
    results = np.linalg.solve(mat, rhs)
    # extract results
    acc1 = results[0:3]
    omega_dot1 = results[3:6]
    acc2 = results[6:9]
    omega_dot2 = results[9:12]
    f1 = results[12:15]
    f2 = results[15:18]
    link1.display_force = f1 + M1_g
    link2.display_force = -f2 + M2_g
    # print(f1, f2)

    # show error - diff between point accelerations
    # p1c_acc = acc1 + np.cross(omega_dot1, r1) + np.cross(w1, np.cross(w1, r1))
    # p2c_acc = acc2 + np.cross(omega_dot2, r2) + np.cross(w2, np.cross(w2, r2))
    # print(np.linalg.norm(p1c_acc-p2c_acc))

    print(omega_dot1, omega_dot2)

    # explicit Euler integration to update the state
    link1.pos += link1.vel * dT
    link1.vel += acc1 * dT
    link1.theta += link1.omega[2] * dT
    link1.omega += omega_dot1 * dT

    link2.pos += link2.vel * dT
    link2.vel += acc2 * dT
    link2.theta += link2.omega[2] * dT
    link2.omega += omega_dot2 * dT

    sim_time += dT

    # draw the updated state
    draw_world()
    print("simTime=%.2f" % sim_time)


def draw_world():
    global link1, link2

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear The Screen And The Depth Buffer
    glLoadIdentity()
    gluLookAt(1, 1, 3, 0, 0, 0, 0, 1, 0)

    draw_origin()
    link1.draw()
    link2.draw()

    glutSwapBuffers()  # swap the buffers to display what was just drawn


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


def resize_gl_scene(width, height):
    if height == 0:  # Prevent A Divide By Zero If The Window Is Too Small
        height = 1
    glViewport(0, 0, width, height)  # Reset The Current Viewport And Perspective Transformation
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(width) / float(height), 0.1,
                   100.0)  # 45 deg horizontal field of view, aspect ratio, near, far
    glMatrixMode(GL_MODELVIEW)


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


print("Hit ESC key to quit.")
main()
