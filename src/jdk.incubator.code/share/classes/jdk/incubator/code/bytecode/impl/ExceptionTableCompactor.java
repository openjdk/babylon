/*
 * Copyright (c) 2024, 2025, Oracle and/or its affiliates. All rights reserved.
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
package jdk.incubator.code.bytecode.impl;

import java.lang.classfile.CodeBuilder;
import java.lang.classfile.CodeElement;
import java.lang.classfile.CodeTransform;
import java.lang.classfile.instruction.ExceptionCatch;

/**
 * ExceptionTableCompactor defragments exception table.
 */
public final class ExceptionTableCompactor implements CodeTransform {

    ExceptionCatch last = null;

    @Override
    public void accept(CodeBuilder cob, CodeElement coe) {
        if (coe instanceof ExceptionCatch ec) {
            if (ec.tryStart() != ec.tryEnd()) {
                if (last != null) {
                    if (last.handler() == ec.handler() && last.catchType().equals(ec.catchType())) {
                        if (last.tryStart() == ec.tryEnd()) {
                            last = ExceptionCatch.of(last.handler(), ec.tryStart(), last.tryEnd(), last.catchType());
                            return;
                        } else if (last.tryEnd() == ec.tryStart()) {
                            last = ExceptionCatch.of(last.handler(), last.tryStart(), ec.tryEnd(), last.catchType());
                            return;
                        }
                    }
                    cob.with(last);
                }
                last = ec;
            }
        } else {
            cob.with(coe);
        }
    }

    @Override
    public void atEnd(CodeBuilder cob) {
        if (last != null) {
            cob.with(last);
            last = null;
        }
    }
}
