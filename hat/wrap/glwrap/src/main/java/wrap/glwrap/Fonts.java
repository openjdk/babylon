package wrap.glwrap;
/*
 * Copyright (c) 2020, Oracle and/or its affiliates. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *   - Neither the name of Oracle nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */



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

