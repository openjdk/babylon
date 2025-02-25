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

package jdk.incubator.code.internal;

import com.sun.tools.javac.code.Symbol.MethodSymbol;
import com.sun.tools.javac.code.Symbol.ModuleSymbol;
import com.sun.tools.javac.code.Symtab;
import com.sun.tools.javac.code.Type;
import com.sun.tools.javac.code.Type.ArrayType;
import com.sun.tools.javac.code.Type.MethodType;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.List;
import com.sun.tools.javac.util.Names;

import static com.sun.tools.javac.code.Flags.PUBLIC;
import static com.sun.tools.javac.code.Flags.STATIC;
import static com.sun.tools.javac.code.Flags.VARARGS;

public class CodeReflectionSymbols {
    public final Type quotedType;
    public final Type quotableType;
    public final Type codeReflectionType;
    public final MethodSymbol opInterpreterInvoke;
    public final MethodSymbol opParserFromString;
    public final MethodSymbol methodHandlesLookup;
    public final Type opType;
    public final Type funcOpType;
    public final Type opFactoryType;
    public final Type typeElementFactoryType;

    CodeReflectionSymbols(Context context) {
        Symtab syms = Symtab.instance(context);
        Names names = Names.instance(context);
        ModuleSymbol jdk_incubator_code = syms.enterModule(names.jdk_incubator_code);
        codeReflectionType = syms.enterClass(jdk_incubator_code, "jdk.incubator.code.CodeReflection");
        quotedType = syms.enterClass(jdk_incubator_code, "jdk.incubator.code.Quoted");
        quotableType = syms.enterClass(jdk_incubator_code, "jdk.incubator.code.Quotable");
        Type opInterpreterType = syms.enterClass(jdk_incubator_code, "jdk.incubator.code.interpreter.Interpreter");
        opType = syms.enterClass(jdk_incubator_code, "jdk.incubator.code.Op");
        funcOpType = syms.enterClass(jdk_incubator_code, "jdk.incubator.code.op.CoreOp$FuncOp");
        opInterpreterInvoke = new MethodSymbol(PUBLIC | STATIC | VARARGS,
                names.fromString("invoke"),
                new MethodType(List.of(syms.methodHandleLookupType, opType, new ArrayType(syms.objectType, syms.arrayClass)), syms.objectType,
                        List.nil(), syms.methodClass),
                opInterpreterType.tsym);
        Type opParserType = syms.enterClass(jdk_incubator_code, "jdk.incubator.code.parser.OpParser");
        opParserFromString = new MethodSymbol(PUBLIC | STATIC,
                names.fromString("fromStringOfFuncOp"),
                new MethodType(List.of(syms.stringType), opType,
                        List.nil(), syms.methodClass),
                opParserType.tsym);
        methodHandlesLookup = new MethodSymbol(PUBLIC | STATIC,
                names.fromString("lookup"),
                new MethodType(List.nil(), syms.methodHandleLookupType,
                        List.nil(), syms.methodClass),
                syms.methodHandlesType.tsym);
        syms.synthesizeEmptyInterfaceIfMissing(quotedType);
        syms.synthesizeEmptyInterfaceIfMissing(quotableType);
        opFactoryType = syms.enterClass(jdk_incubator_code, "jdk.incubator.code.op.OpFactory");
        typeElementFactoryType = syms.enterClass(jdk_incubator_code, "jdk.incubator.code.type.TypeElementFactory");


    }
}
