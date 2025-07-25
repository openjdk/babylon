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
package wrap.opengl;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static opengl.opengl_h.*;

public class Fonts {
    static class FontsWindow extends GLWindow {
        public FontsWindow( Arena arena, int width, int height, String name, int mode, GLTexture... textures) {
            super( arena, width, height, name, mode, textures);
        }

        @Override
        public void reshape(int w, int h) {
            width = w;
            height = h;
            System.out.println("reshaped " + w + ", " + h);
            double size;
            double aspect;
            glViewport(0, 0, w, h);
            glMatrixMode(GL_PROJECTION());
            glLoadIdentity();
            size = (double) ((w >= h) ? w : h) / 2.0;
            if (w <= h) {
                aspect = (double) h / (double) w;
                glOrtho(-size, size, -size * aspect, size * aspect, -100000.0, 100000.0);
            } else {
                aspect = (double) w / (double) h;
                glOrtho(-size * aspect, size * aspect, -size, size, -100000.0, 100000.0);
            }
            glScaled(aspect, aspect, 1.0);
            glMatrixMode(GL_MODELVIEW());
        }

        @Override
        public void display() {
            glClearColor(0f, 0f, 0f, 0f);
            glPushMatrix1(() -> {
                float x = -225.0f;
                float y = 70.0f;
                float ystep = 10.0f;
                boolean stroke = true;
                if (stroke) {
                    float stroke_scale = 0.2f;
                    glTranslatef(x, y, 0.0f);
                    MemorySegment font = glutStrokeRoman$segment();
                    for (int j = 0; j < 4; j++) {
                        glPushMatrix1(() -> {
                            glScalef(stroke_scale, stroke_scale, stroke_scale);
                            for (int c : "This text stroked".getBytes()) {
                                glutStrokeCharacter(font, c);
                            }
                        });
                        glTranslatef(0f, -ystep * 10 + j, 0.0f);
                    }
                } else {
                    glColor3f(0.0f, 1.0f, 0.0f);
                    var font = glutBitmapTimesRoman24$segment();
                    for (int j = 0; j < 4; j++) {
                        glRasterPos2f(10f, 10f + ystep * j * 10);
                        for (int c : "This text ".getBytes()) {
                            glutBitmapCharacter(font, c);
                        }
                    }
                }
            });
            glutSwapBuffers();
        }


        @Override
        public void onIdle() {
            super.onIdle();
        }
    }
    public static void main(String[] args) throws IOException {

        try (var arena = Arena.ofConfined()) {
       //     var particleTexture = new GLTexture(arena, Fonts.class.getResourceAsStream("/particle.png"));

            new FontsWindow( arena, 1000, 1000, "Fonts", GLUT_RGB() | GLUT_DOUBLE() /*particleTexture  particleTexture */).bindEvents().mainLoop();
        }
    }
}

