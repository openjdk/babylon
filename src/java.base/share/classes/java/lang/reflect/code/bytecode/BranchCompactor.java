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
package java.lang.reflect.code.bytecode;

import java.lang.classfile.CodeBuilder;
import java.lang.classfile.CodeElement;
import java.lang.classfile.CodeTransform;
import java.lang.classfile.PseudoInstruction;
import java.lang.classfile.instruction.BranchInstruction;
import java.lang.classfile.instruction.LabelTarget;
import java.util.ArrayList;
import java.util.List;

final class BranchCompactor implements CodeTransform {

    private BranchInstruction firstBranch;
    private final List<PseudoInstruction> pseudoBuffer = new ArrayList<>();

    private final CodeTransform INIT_STATE = new CodeTransform() {
        @Override
        public void accept(CodeBuilder cob, CodeElement coe) {
            if (coe instanceof BranchInstruction bi) {
                firstBranch = bi;
                activeState = LOOKING_FOR_TARGET;
            } else {
                cob.with(coe);
            }
        }
    };

    private final CodeTransform LOOKING_FOR_TARGET = new CodeTransform() {
        @Override
        public void accept(CodeBuilder cob, CodeElement coe) {
            switch (coe) {
                case LabelTarget lt -> {
                    if (firstBranch.target() == lt.label()) {
                        pseudoBuffer.forEach(cob::with);
                        pseudoBuffer.clear();
                        activeState = INIT_STATE;
                        cob.with(coe);
                    } else {
                        pseudoBuffer.add(lt);
                    }
                }
                case PseudoInstruction pi -> {
                    pseudoBuffer.add(pi);
                }
                case BranchInstruction bi -> {
                    atEnd(cob);
                    firstBranch = bi;
                }
                default -> {
                    atEnd(cob);
                    activeState = INIT_STATE;
                    cob.with(coe);
                }
            }
        }
        @Override
        public void atEnd(CodeBuilder cob) {
            cob.accept(firstBranch);
            pseudoBuffer.forEach(cob::with);
            pseudoBuffer.clear();
        }
    };

    private CodeTransform activeState = INIT_STATE;

    @Override
    public void accept(CodeBuilder cob, CodeElement coe) {
        activeState.accept(cob, coe);
    }

    @Override
    public void atEnd(CodeBuilder cob) {
        activeState.atEnd(cob);
    }
}
