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
package experiments;

import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.CodeContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.Query;
import optkl.Trxfmr;
import optkl.codebuilders.JavaCodeBuilder;
import optkl.util.Regex;

import java.io.PrintStream;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.Comparator;
import java.util.List;
import java.util.SequencedSet;
import java.util.TreeSet;

import static optkl.OpHelper.Named.NamedStaticOrInstance.Invoke;
import static optkl.OpHelper.asOpFromResultOrNull;

public class Queries {

    @Reflect
    static int m(int a, int b) {
        a += 2;
        b += 2;
        // Group these
        System.out.println(a);
        System.out.println(b);
        return a + b;
    }


    public static void main(String[] args) throws Throwable {
        var lookup = MethodHandles.lookup();
        Method m = Queries.class.getDeclaredMethod("m", int.class, int.class);
        CoreOp.FuncOp mModel = Op.ofMethod(m).orElseThrow();

        var query = Query.InvokeQuery.create(lookup);
        Invoke.stream(lookup,mModel).forEach(invoke->{
            if (query.test(invoke.op()) instanceof Query.InvokeQuery.Match<?> match){
                System.out.println(((Invoke)match.helper()).name());
            }else{
                System.out.println("failed");
            }
        });

    }
}
