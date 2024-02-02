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

/**
 * BranchCompactor is a CodeTransform skipping redundant branches to immediate targets.
 */
public final class BranchCompactor implements CodeTransform {

    private BranchInstruction branch;
    private final List<PseudoInstruction> buffer = new ArrayList<>();

    public BranchCompactor() {
    }

    @Override
    public void accept(CodeBuilder cob, CodeElement coe) {
        if (branch == null) {
            if (coe instanceof BranchInstruction bi && bi.opcode().isUnconditionalBranch()) {
                //unconditional branch is stored
                branch = bi;
            } else {
                //all other elements are passed
                cob.with(coe);
            }
        } else {
            switch (coe) {
                case LabelTarget lt -> {
                    if (branch.target() == lt.label()) {
                        //skip branch to immediate target
                        branch = null;
                        //flush the buffer
                        atEnd(cob);
                        //pass the target
                        cob.with(lt);
                    } else {
                        //buffer other targets
                        buffer.add(lt);
                    }
                }
                case PseudoInstruction pi -> {
                    //buffer pseudo instructions
                    buffer.add(pi);
                }
                default -> {
                    //any other instruction flushes the branch and buffer
                    atEnd(cob);
                    //replay the code element
                    accept(cob, coe);
                }
            }
        }
    }

    @Override
    public void atEnd(CodeBuilder cob) {
        if (branch != null) {
            //flush the branch
            cob.with(branch);
            branch = null;
        }
        //flush the buffer
        buffer.forEach(cob::with);
        buffer.clear();
    }
}
