package wrap.glwrap;

import opengl.glutDisplayFunc$func;
import opengl.glutIdleFunc$func;
import opengl.glutKeyboardFunc$func;
import opengl.glutMotionFunc$func;
import opengl.glutMouseFunc$func;
import opengl.glutPassiveMotionFunc$func;
import opengl.glutReshapeFunc$func;

import java.lang.foreign.Arena;

import static opengl.opengl_h.glutDisplayFunc;
import static opengl.opengl_h.glutIdleFunc;
import static opengl.opengl_h.glutKeyboardFunc;
import static opengl.opengl_h.glutMotionFunc;
import static opengl.opengl_h.glutMouseFunc;
import static opengl.opengl_h.glutPassiveMotionFunc;
import static opengl.opengl_h.glutReshapeFunc;


public  class GLFuncEventHandler implements GLEventHandler {
    @Override
    public void addEvents(Arena arena, GLWindow glWindow) {
        glutReshapeFunc(glutReshapeFunc$func.allocate(glWindow::reshape, arena));
        glutDisplayFunc(glutDisplayFunc$func.allocate(glWindow::display, arena));
        glutIdleFunc(glutIdleFunc$func.allocate(glWindow::onIdle, arena));
        glutKeyboardFunc(glutKeyboardFunc$func.allocate(glWindow::keyboard, arena));
        glutMouseFunc(glutMouseFunc$func.allocate(glWindow::mouse, arena));
        glutMotionFunc(glutMotionFunc$func.allocate(glWindow::mouseMotion, arena));
        glutPassiveMotionFunc(glutPassiveMotionFunc$func.allocate(glWindow::mousePassiveMotion, arena));

    }
}
