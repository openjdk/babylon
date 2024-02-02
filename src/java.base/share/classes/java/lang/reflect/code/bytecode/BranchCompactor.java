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
import java.lang.classfile.Opcode;
import java.lang.classfile.PseudoInstruction;
import java.lang.classfile.instruction.BranchInstruction;
import java.lang.classfile.instruction.LabelTarget;
import java.util.ArrayList;
import java.util.List;

/**
 * BranchCompactor is a CodeTransform working as a state machine.
 * It identifies and compacts redundant sequences of branches in a single pass.
 */
public final class BranchCompactor implements CodeTransform {

    public BranchCompactor() {
    }

    private BranchInstruction firstBranch, secondBranch;
    private final List<PseudoInstruction> pseudoBuffer1 = new ArrayList<>(),
                                          pseudoBuffer2 = new ArrayList<>();

    //BranchCompactor is in INIT_STATE until a branch instruction appears
    private final CodeTransform INIT_STATE = new CodeTransform() {
        @Override
        public void accept(CodeBuilder cob, CodeElement coe) {
            if (coe instanceof BranchInstruction bi) {
                firstBranch = bi;
                activeState = LOOKING_FOR_SHORT_JUMP;
            } else {
                //all other instructions and pseudo instructions are passed
                cob.with(coe);
            }
        }
    };

    //In this state we are looking for immediate target of the firstBranch
    //or also for a second (unconditional) branch when the firstBranch is conditional
    //all pseudo instructions are buffered as they do not represent a real bytecode
    private final CodeTransform LOOKING_FOR_SHORT_JUMP = new CodeTransform() {
        @Override
        public void accept(CodeBuilder cob, CodeElement coe) {
            switch (coe) {
                case LabelTarget lt -> {
                    if (firstBranch.target() == lt.label()) {
                        //here we have immediate target, so the first branch is skipped
                        //all pseudo instructions are passed (including the actual target)
                        //and BranchCompactor returns to INIT_STATE
                        pseudoBuffer1.forEach(cob::with);
                        pseudoBuffer1.clear();
                        activeState = INIT_STATE;
                        cob.with(coe);
                    } else {
                        //here we buffer the label as a pseudo instructions
                        pseudoBuffer1.add(lt);
                    }
                }
                case PseudoInstruction pi -> {
                    //here we buffer pseudo instructions
                    pseudoBuffer1.add(pi);
                }
                case BranchInstruction bi -> {
                    if (!firstBranch.opcode().isUnconditionalBranch() && bi.opcode().isUnconditionalBranch()) {
                        //second (unconditional) branch appears and the firstBranch is conditional
                        //so the BranchCompactor moves to LOOKING_FOR_DOUBLE_JUMP state
                        secondBranch = bi;
                        activeState = LOOKING_FOR_DOUBLE_JUMP;
                    } else {
                        //branches do not meet criteria to look for double jumps
                        //so we flush the firstBranch and all buffered pseudo instructions
                        //and continue in this state with a new firstBranch
                        atEnd(cob);
                        firstBranch = bi;
                    }
                }
                default -> {
                    //any other instruction flushes the firstBranch and pseudo instructions
                    //and returns BranchCompactor to INIT_STATE
                    atEnd(cob);
                    activeState = INIT_STATE;
                    cob.with(coe);
                }
            }
        }
        @Override
        public void atEnd(CodeBuilder cob) {
            //here we flush the firstBranch and pseudo instructions
            cob.accept(firstBranch);
            pseudoBuffer1.forEach(cob::with);
            pseudoBuffer1.clear();
        }
    };

    //This state assumes we have a sequence of one conditional and one unconditional branch
    //and we are trying to merge them with reversed condition
    private final CodeTransform LOOKING_FOR_DOUBLE_JUMP = new CodeTransform() {
        @Override
        public void accept(CodeBuilder cob, CodeElement coe) {
            switch (coe) {
                case LabelTarget lt -> {
                    if (secondBranch.target() == lt.label()) {
                        //second branch has been identified as short-circuit
                        //so we move to LOOKING_FOR_SHORT_JUMP state
                        //and replay all pseudo instructions from the second buffer
                        //as there might be another short target
                        activeState = LOOKING_FOR_SHORT_JUMP;
                        pseudoBuffer2.forEach(pi -> activeState.accept(cob, pi));
                        pseudoBuffer2.clear();
                        activeState.accept(cob, lt);
                    } else if (firstBranch.target() == lt.label()) {
                        //double branch has been detected
                        //we replace firstBranch instruction with reverted condition and secondBranch target
                        //move to LOOKING_FOR_SHORT_JUMP state and replay pseudoBuffer2
                        firstBranch = BranchInstruction.of(reverseBranchOpcode(firstBranch.opcode()), secondBranch.target());
                        activeState = LOOKING_FOR_SHORT_JUMP;
                        pseudoBuffer2.forEach(pi -> activeState.accept(cob, pi));
                        pseudoBuffer2.clear();
                        activeState.accept(cob, lt);
                    } else {
                        //here we buffer the label as a pseudo instruction following the secondBranch
                        pseudoBuffer2.add(lt);
                    }
                }
                case PseudoInstruction pi -> {
                    //here we buffer pseudo instructions following the secondBranch
                    pseudoBuffer2.add(pi);
                }
                case BranchInstruction bi -> {
                    //third branch has been detected, so we flush the firstBranch and its pseudo instructions
                    //move to LOOKING_FOR_SHORT_JUMP state, shift secondBranch to the firstBranch
                    //replay the secondBranch pseudo instructions and this actual branch
                    LOOKING_FOR_SHORT_JUMP.atEnd(cob);
                    activeState = LOOKING_FOR_SHORT_JUMP;
                    firstBranch = secondBranch;
                    pseudoBuffer2.forEach(pi -> activeState.accept(cob, pi));
                    pseudoBuffer2.clear();
                    activeState.accept(cob, bi);
                }
                default -> {
                    //any other instruction flushes all the branches and buffered pseudo instructions
                    atEnd(cob);
                    activeState = INIT_STATE;
                    cob.with(coe);
                }
            }
        }
        @Override
        public void atEnd(CodeBuilder cob) {
            //here we flush everything
            LOOKING_FOR_SHORT_JUMP.atEnd(cob);
            cob.accept(secondBranch);
            pseudoBuffer2.forEach(cob::with);
            pseudoBuffer2.clear();
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

    static Opcode reverseBranchOpcode(Opcode op) {
        return switch (op) {
            case IFEQ -> Opcode.IFNE;
            case IFNE -> Opcode.IFEQ;
            case IFLT -> Opcode.IFGE;
            case IFGE -> Opcode.IFLT;
            case IFGT -> Opcode.IFLE;
            case IFLE -> Opcode.IFGT;
            case IF_ICMPEQ -> Opcode.IF_ICMPNE;
            case IF_ICMPNE -> Opcode.IF_ICMPEQ;
            case IF_ICMPLT -> Opcode.IF_ICMPGE;
            case IF_ICMPGE -> Opcode.IF_ICMPLT;
            case IF_ICMPGT -> Opcode.IF_ICMPLE;
            case IF_ICMPLE -> Opcode.IF_ICMPGT;
            case IF_ACMPEQ -> Opcode.IF_ACMPNE;
            case IF_ACMPNE -> Opcode.IF_ACMPEQ;
            case IFNULL -> Opcode.IFNONNULL;
            case IFNONNULL -> Opcode.IFNULL;
            default -> throw new IllegalArgumentException("Unknown branch instruction: " + op);
        };
    }
}
