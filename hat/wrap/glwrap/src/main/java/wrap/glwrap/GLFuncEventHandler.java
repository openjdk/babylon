/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
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
