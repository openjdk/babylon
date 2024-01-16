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

package jdk.code.tools.renderer;

import java.io.StringWriter;

public final class CStyleRenderer extends CommonRenderer<CStyleRenderer> {
    public CStyleRenderer() {
        super();
    }

    public CStyleRenderer func(String identifier, TextRenderer.NestedRendererSAM<CStyleRenderer> args,
                               TextRenderer.NestedRendererSAM<CStyleRenderer> body) {
        return keyword("func").space().identifier(identifier).parenthesized(args).braced(body).nl();
    }

    public CStyleRenderer forLoop(TextRenderer.NestedRendererSAM<CStyleRenderer> init, TextRenderer.NestedRendererSAM<CStyleRenderer> cond,
                                  TextRenderer.NestedRendererSAM<CStyleRenderer> mutator, TextRenderer.NestedRendererSAM<CStyleRenderer> body) {
        return forKeyword().oparen().nest(init).semicolon().space().nest(cond).semicolon().space().
                nest(mutator).cparen().braced(body).nl();
    }

    public CStyleRenderer whileLoop(TextRenderer.NestedRendererSAM<CStyleRenderer> cond, TextRenderer.NestedRendererSAM<CStyleRenderer> body) {
        return whileKeyword().oparen().nest(cond).cparen().braced(body).nl();
    }

    public CStyleRenderer ifCondRaw(TextRenderer.NestedRendererSAM<CStyleRenderer> cond, TextRenderer.NestedRendererSAM<CStyleRenderer> thenBody) {
        return ifKeyword().oparen().nest(cond).cparen().braced(thenBody);
    }

    public CStyleRenderer ifCond(TextRenderer.NestedRendererSAM<CStyleRenderer> cond, TextRenderer.NestedRendererSAM<CStyleRenderer> thenBody) {
        return ifCondRaw(cond, thenBody).nl();
    }

    public CStyleRenderer ifCond(TextRenderer.NestedRendererSAM<CStyleRenderer> cond, TextRenderer.NestedRendererSAM<CStyleRenderer> thenBody,
                                 TextRenderer.NestedRendererSAM<CStyleRenderer> elseBody) {
        return ifCondRaw(cond, thenBody).elseKeyword().braced(elseBody).nl();
    }

    public CStyleRenderer var(Class<?> clazz, String name) {
        return type(clazz.getName()).space().identifier(name);
    }

    public CStyleRenderer assign(String identifier) {
        return identifier(identifier).equal();
    }

    static public void main(String[] args) {
        StringWriter writer = new StringWriter();
        CStyleRenderer renderer = new CStyleRenderer().writer(writer).colorize();
        renderer.lineComment("A new function");
        renderer.func("funcName",
                (as) -> as.var(int.class, "name").comma().space()
                        .var(int.class, "name2"),
                (fb) -> fb.lineComment("Inside body of func")
                        .append("here;\nis;\nsome text").semicolon().nl()
                        .forLoop(
                                (in) -> in.var(int.class, "a").equal().decLiteral(0),
                                (cc) -> cc.identifier("a").op("<").decLiteral(10),
                                (mu) -> mu.assign("a").identifier("a").op("+").decLiteral(1),
                                (lb) -> lb.lineComment("in loop")
                                        .ifCond(
                                                (cc) -> cc.identifier("a").op(">").decLiteral(2),
                                                (th) -> th.lineComment("positive"),
                                                (el) -> el.lineComment("not so much ")
                                        )
                        )
                        .nl()
        ).nl();
        System.out.println(writer);

    }
}