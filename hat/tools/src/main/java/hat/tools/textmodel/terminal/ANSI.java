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
package hat.tools.textmodel.terminal;


import hat.tools.textmodel.tokens.Token;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.function.Consumer;
import java.util.function.Function;

public interface ANSI<T extends ANSI<T>> extends  Function<String, T> {
    T apply(String o);

    default T apply(Token t) {
        return apply(t.asString());
    }

    default T self() {
        return (T) this;
    }

    default void skip(){
    }

    int BLACK = 30;
    int RED = 31;
    int GREEN = 32;
    int YELLOW = 33;
    int BLUE = 34;
    int PURPLE = 35;
    int CYAN = 36;
    int WHITE = 37;

    default T esc() {
        return apply("\033");
    }

    default T color(int c, Consumer<T> cc) {
        csi().ints(c).apply("m");
        cc.accept(self());
        return reset();
    }

    default T color(int c1, int c2, Consumer<T> cc) {
        csi().ints(c1, c2).apply("m");
        cc.accept(self());

        return reset();
    }
    default T fg(int color, Consumer<T> cc) {
        return color(0, color, cc);
    }


    default T bold(int color, Consumer<T> cc) {
        return color(1, color, cc);
    }


    default T boldAndBright(int color, Consumer<T> cc) {
        return color(0, color + 60, cc);
    }



    default T bright(int color, Consumer<T> cc) {
        return color(1, color + 60, cc);
    }



    default T bg(int color,Consumer<T> cc) {
        return color(color + 10, cc);
    }

    default T csi() {
        return esc().apply("[");
    }

    default T home(){
        return csi().ints(0,0).apply("H");
    }

    default T csiQuery() {
        return csi().apply("?");
    }

    default T ints(int... n) {
        apply(String.valueOf(n[0]));
        for (int i = 1; i < n.length; i++) {
            apply(";" + n[i]);
        }
        return self();
    }


    default T inv() {
        return csi().ints(7).apply("m");
    }

    default T reset() {
        return csi().ints(0).apply("m");
    }

    default T inv(Consumer<T> c) {
        inv();
        c.accept(self());
        return reset();
    }



    record Adaptor(Consumer<String> consumer) implements ANSI<Adaptor> {
        @Override
        public Adaptor apply(String s) {
            consumer.accept(s);
            return self();
        }
    }

    default T repeat(String s, int count) {
        return apply(s.repeat(count));
    }

    default T fill(int cols, String s) {
        return apply(s).repeat(" ", Math.max(0, cols - s.length()));
    }

    class IMPL implements ANSI<IMPL> {
        private final PrintStream printStream;

        @Override
        public IMPL apply(String s) {
            printStream.append(s);
            return self();
        }

        IMPL(PrintStream printStream) {
            this.printStream = printStream;
        }
    }

    static IMPL of(PrintStream printStream) {
        return new IMPL(printStream);
    }


     class DotImage {
        final private ANSI<?> ansi;
        final public int width;
        final public int height;
        final private byte[] bytes;
        int charWidth;
        int charHeight;
        final private  char[] chars;
        public DotImage(ANSI<?> ansi, int width, int height){
            this.ansi = ansi;
            this.width = width;
            this.height = height;
            this.bytes = new byte[width*height];
            this.charWidth = width/2;
            this.charHeight = height/4;
            this.chars = new char[charWidth*charHeight];
        }
        public void set(int x, int y){
            bytes[(y*width)+x]=1;//0xff;
        }
        void reset(int x, int y){
            bytes[(y*width)+x]=0;
        }
        int i(int x, int y){
            byte b = bytes[(y*width)+x];
            return (int)(b<0?b+256:b);
        }
        /**
         See the unicode mapping table here
         https://images.app.goo.gl/ntxis4mKzn7GmrGb7
         */
        char dotchar(int bytebits){
           // int mapped = (bytebits&0x07)|(bytebits&0x70)>>1|(bytebits&0x08)<<3|(bytebits&0x80);
            //char brail = (char)(0x2800+(bytebits&0x07)|(bytebits&0x70)>>1|(bytebits&0x08)<<3|(bytebits&0x80));
            return  (char)(0x2800+(bytebits&0x07)|(bytebits&0x70)>>1|(bytebits&0x08)<<3|(bytebits&0x80));
        }
        public DotImage home(){

            ansi.home();
            return this;
        }
        public DotImage delay(int ms){
            try{ Thread.sleep(ms); }catch(Throwable t){ }
            return this;
        }

        public DotImage clean(){
            Arrays.fill(bytes,(byte)0);
            Arrays.fill(chars,(char)' ');
            return this;
        }
        public DotImage map(){
            for (int cx = 0; cx<charWidth; cx++){
                for (int cy = 0; cy<charHeight; cy++){
                    int bytebits=0;
                    for (int dx=0;dx<2;dx++){
                        for (int dy=0;dy<4;dy++){
                            bytebits|=i(cx*2+dx,cy*4+dy)<<(dx*4+dy);
                        }
                    }
                    chars[cy*charWidth+cx]= dotchar(bytebits);
                }
            }
            return this;
        }

        public DotImage write(){
            ansi.apply("+");
            for (int i=0;i<charWidth; i++){
                ansi.apply("-");
            }
            ansi.apply("+\n|");
            for (int i=0;i<chars.length; i++){
                if (i>0 && (i%charWidth)==0){
                    ansi.apply("|\n|");
                }
                ansi.apply(Character.toString(chars[i]));
            }
            ansi.apply("|\n+");
            for (int i=0;i<charWidth; i++){
                ansi.apply("-");
            }
            ansi.apply("+\n");
            return this;
        }
    }
    default DotImage img(int width, int height){
        return new DotImage(this, width,height);
    }

}
