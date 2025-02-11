package wrap.glwrap;

import java.lang.foreign.Arena;

import static opengl.opengl_h.*;


public  class GLCallbackEventHandler implements GLEventHandler {
    @Override
    public void addEvents(Arena arena,GLWindow glWindow) {
        glutDisplayFunc(opengl.glutDisplayFunc$callback.allocate(glWindow::display, arena));
        glutIdleFunc(opengl.glutIdleFunc$callback.allocate(glWindow::onIdle, arena));

        glutKeyboardFunc(opengl.glutKeyboardFunc$callback.allocate(glWindow::keyboard, arena));
        glutMouseFunc(opengl.glutMouseFunc$callback.allocate(glWindow::mouse, arena));
        glutMotionFunc(opengl.glutMotionFunc$callback.allocate(glWindow::mouseMotion, arena));
        glutPassiveMotionFunc(opengl.glutPassiveMotionFunc$callback.allocate(glWindow::mousePassiveMotion, arena));

    }
}
