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
package hat.tools.textmodel.tokens;

import hat.tools.textmodel.tokens.defaults.Close;
import hat.tools.textmodel.tokens.defaults.LineComment;
import hat.tools.textmodel.tokens.defaults.MultiLineComment;
import hat.tools.textmodel.tokens.defaults.Open;

public interface Factory {
    default Open open(Parent parent, Pos pos, char ch) {
        return new Open(parent, pos, ch);
    }

    default Close close(Parent parent, Pos pos, char ch) {
        return new Close(parent, pos, ch);
    }

    default LineComment lineComment(Parent parent, Pos pos, int len) {
        return new LineComment(parent, pos, len);
    }

    default MultiLineComment multiLineComment(Parent parent, Pos pos, int len) {
        return new MultiLineComment(parent, pos, len);
    }

    default Ws ws(Parent parent, Pos pos, int len) {
        return new hat.tools.textmodel.tokens.defaults.Ws(parent, pos, len);
    }

    default CharLiteral charLiteral(Parent parent, Pos pos, int len) {
        return new hat.tools.textmodel.tokens.defaults.CharLiteral(parent, pos, len);
    }

    default StringLiteral stringLiteral(Parent parent, Pos pos, int len) {
        return new hat.tools.textmodel.tokens.defaults.StringLiteral(parent, pos, len);
    }

    default Nl nl(Parent parent, Pos pos) {
        return new hat.tools.textmodel.tokens.defaults.Nl(parent, pos);
    }

    default Parenthesis parenthesis(Parent parent, Pos pos, Parenthesis.OpenClose openClose) {
        return new hat.tools.textmodel.tokens.defaults.Parenthesis(parent, pos,  openClose);
    }

    default Seq seq(Parent parent, Pos pos, int len) {
        return new hat.tools.textmodel.tokens.defaults.Seq(parent, pos, len);
    }

    default Ch ch(Parent parent, Pos pos) {
        return new hat.tools.textmodel.tokens.defaults.Ch(parent, pos);
    }
}
