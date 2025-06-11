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


import wrap.ArenaHolder;
import wrap.Wrap;

import java.lang.foreign.Arena;
import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;

import static java.lang.foreign.MemorySegment.NULL;
import static opengl.opengl_h.*;

public abstract class GLWindow implements ArenaHolder {

    public record key(byte key, int x, int y) {
        @Override
        public String toString() {
            return "keyboard: " + key + " x: " + x + " y: " + y;
        }
    }

    public record mouse(int button, int state, int x, int y) {
        @Override
        public String toString() {
            return "mouse: " + button + " state: " + state + " x: " + x + " y: " + y;
        }
    }

    public record mouseMotion(int x, int y) {
        @Override
        public String toString() {
            return "mouse motion:  x: " + x + " y: " + y;
        }
    }

    public record mousePassiveMotion(int x, int y) {
        @Override
        public String toString() {
            return "mouse passive motion:  x: " + x + " y: " + y;
        }
    }

    protected key lastKey = null;
    protected mouse lastMouse = null;
    protected mouseMotion lastMouseMotion = null;
    protected mousePassiveMotion lastPassiveMouseMotion;

    public void keyboard(byte key, int x, int y) {
        lastKey = new key(key, x, y);
        // System.out.println(lastKey);
    }

    public void mouse(int button, int state, int x, int y) {
        lastMouse = new mouse(button, state, x, y);
        //   System.out.println(lastMouse);
    }

    public void mouseMotion(int x, int y) {
        lastMouseMotion = new mouseMotion(x, y);
        //  System.out.println(lastMouseMotion);
    }

    public void mousePassiveMotion(int x, int y) {
        lastPassiveMouseMotion = new mousePassiveMotion(x, y);
        //  System.out.println(lastPassiveMouseMotion);
    }

    public Arena arena;
    @Override public Arena arena(){
        return arena;
    }
    public int width;
    public int height;
    public String name;
    public GLTexture[] textures;
    public Wrap.IntArr textureBuf;



    public GLWindow( Arena arena, int width, int height, String name, int mode,
                    GLTexture... textures) {
        this.arena = arena;
        this.width = width;
        this.height = height;
        this.name = name;
        this.textures = textures;

        var useLighting = false;

        var argc = intPtr(0);
        var argv = ptrArr( NULL);
        glutInit(argc.ptr(), argv.ptr());
        glutInitDisplayMode(mode);
        glutInitWindowSize(width, height);
        var windowName = cstr(name);
        glutCreateWindow(windowName.ptr());
        System.out.println("GL_VENDOR                    : "+ cstr(glGetString(GL_VENDOR())));
        System.out.println("GL_RENDERER                  : "+ cstr(glGetString(GL_RENDERER())));
        System.out.println("GL_VERSION                   : "+ cstr(glGetString(GL_VERSION())));
        System.out.println("GL_SHADING_LANGUAGE_VERSION  : "+ cstr(glGetString(GL_SHADING_LANGUAGE_VERSION())));


        glShadeModel(GL_SMOOTH());
        glEnable(GL_BLEND());
        glBlendFunc(GL_SRC_ALPHA(), GL_ONE());

        if (textures != null && textures.length > 0 && textures[0] != null) {
            //  glEnable(GL_TEXTURE_2D());
            textureBuf = ofInts(textures.length);
            glGenTextures(textures.length, textureBuf.ptr());
            int[] count = {0};
            Arrays.stream(textures).forEach(texture -> {
                texture.idx = count[0]++;
                glBindTexture(GL_TEXTURE_2D(), textureBuf.get(texture.idx));
                glTexImage2D(GL_TEXTURE_2D(), 0, GL_RGBA(), texture.width,
                        texture.height, 0, GL_RGBA(), GL_UNSIGNED_BYTE(), texture.data);
                glTexParameteri(GL_TEXTURE_2D(), GL_TEXTURE_MAG_FILTER(), GL_LINEAR());
                glTexParameteri(GL_TEXTURE_2D(), GL_TEXTURE_MIN_FILTER(), GL_NEAREST());
            });
        }

        // Setup Lighting see  https://www.khronos.org/opengl/wiki/How_lighting_works

        if (useLighting) {
            glEnable(GL_LIGHTING());
            var light = GL_LIGHT0(); // .... LIGHT_0 .. -> 7
            glLightfv(light, GL_POSITION(), ofFloats(0.0f, 15.0f, -15.0f, 0).ptr());
            glLightfv(light, GL_AMBIENT(), ofFloats(1f, 0.0f, 0.0f, 0.0f).ptr());
            glLightfv(light, GL_DIFFUSE(), ofFloats(1f, 1f, 1f, 0.0f).ptr());
            glLightfv(light, GL_SPECULAR(), ofFloats(1.0f, 1.0f, 0.0f, 0.0f).ptr());

            var shini = floatPtr(113);
            glMaterialfv(GL_FRONT(), GL_SHININESS(), shini.ptr());

            var useColorMaterials = false;
            if (useColorMaterials) {
                glEnable(GL_COLOR_MATERIAL());
            } else {
                glDisable(GL_COLOR_MATERIAL());
            }
            glEnable(light);
            glEnable(GL_DEPTH_TEST());
        } else {
            glDisable(GL_LIGHTING());
        }


    }

    public GLWindow bindEvents(String... eventHandleClassNames){
        for (String eventHandleClassName : eventHandleClassNames) {
            try {
                var clazz = Class.forName(eventHandleClassName);
                if (clazz.getDeclaredConstructor().newInstance() instanceof GLEventHandler handler){
                    handler.addEvents(arena(),this);
                }

            } catch (ClassNotFoundException | InvocationTargetException
                     | InstantiationException | IllegalAccessException |
                     NoSuchMethodException e) {
                // ok
            }

        }
        return this;

    }
    public GLWindow bindEvents(){
        return bindEvents(
                "wrap.glwrap.GLCallbackEventHandler",
                "wrap.glwrap.GLFuncEventHandler"
        );
    }


    public void glQuads(Runnable r) {
        glBegin(GL_QUADS());
        r.run();
        glEnd();
    }

    public void glPushMatrix1( Runnable r) {
        glPushMatrix();
        r.run();
        glPopMatrix();
    }

    public abstract void display();

    public  void reshape(int w, int h){}

    public void onIdle() {
        glutPostRedisplay();
    }

    public void mainLoop() {
        glutMainLoop();
    }


}
