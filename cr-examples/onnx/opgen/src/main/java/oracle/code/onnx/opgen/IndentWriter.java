/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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

package oracle.code.onnx.opgen;

import java.io.IOException;
import java.io.Writer;

// @@@ Note this leaves trailing white space for empty lines
final class IndentWriter extends Writer {
    static final int INDENT = 4;

    final Writer w;
    int indent;
    boolean writeIndent = true;

    IndentWriter(Writer w) {
        this(w, 0);
    }

    IndentWriter(Writer w, int indent) {
        this.w = w;
        this.indent = indent;
    }

    @Override
    public void write(char[] cbuf, int off, int len) throws IOException {
        if (writeIndent) {
            w.write(" ".repeat(indent));
            writeIndent = false;
        }

        int end = off + len;
        for (int i = off; i < end; i++) {
            if (cbuf[i] != '\n') {
                continue;
            }

            if (writeIndent) {
                w.write(" ".repeat(indent));
            }

            w.write(cbuf, off, i - off + 1);
            writeIndent = true;
            off = i + 1;
        }
        if (off < end) {
            w.write(cbuf, off, end - off);
        }
    }

    @Override
    public void flush() throws IOException {
        w.flush();
    }

    @Override
    public void close() throws IOException {
        w.close();
    }

    void in() {
        in(INDENT);
    }

    void in(int i) {
        indent += i;
    }

    void out() {
        out(INDENT);
    }

    void out(int i) {
        indent -= i;
    }
}
