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

import java.lang.classfile.Attributes;
import java.lang.classfile.ClassTransform;
import java.lang.classfile.CodeModel;
import java.lang.classfile.MethodModel;
import java.lang.classfile.TypeKind;
import java.lang.classfile.attribute.StackMapFrameInfo;
import java.lang.classfile.instruction.BranchInstruction;
import java.lang.classfile.instruction.IncrementInstruction;
import java.lang.classfile.instruction.LabelTarget;
import java.lang.classfile.instruction.LoadInstruction;
import java.lang.classfile.instruction.LookupSwitchInstruction;
import java.lang.classfile.instruction.TableSwitchInstruction;
import java.lang.classfile.instruction.StoreInstruction;
import java.lang.constant.ClassDesc;
import java.lang.reflect.AccessFlag;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static java.lang.classfile.attribute.StackMapFrameInfo.SimpleVerificationTypeInfo.*;
import static java.lang.constant.ConstantDescs.CD_double;
import static java.lang.constant.ConstantDescs.CD_long;

/**
 * LocalsCompactor is a CodeTransform reducing maxLocals.
 */
public final class LocalsCompactor {

    public static final ClassTransform INSTANCE = (clb,cle) -> {
        if (cle instanceof MethodModel mm) {
            clb.transformMethod(mm, (mb, me) -> {
                if (me instanceof CodeModel com) {
                    int[] slotMap = new LocalsCompactor(com, countParamSlots(mm)).slotMap;
                    mb.transformCode(com, (cob, coe) -> {
                        switch (coe) {
                            case LoadInstruction li ->
                                cob.loadLocal(li.typeKind(), slotMap[li.slot()]);
                            case StoreInstruction si ->
                                cob.storeLocal(si.typeKind(), slotMap[si.slot()]);
                            case IncrementInstruction ii ->
                                cob.iinc(slotMap[ii.slot()], ii.constant());
                            default ->
                                cob.with(coe);
                        }
                    });
                } else {
                    mb.with(me);
                }
            });
        } else {
            clb.with(cle);
        }
    };

    private static int countParamSlots(MethodModel mm) {
        int slots = mm.flags().has(AccessFlag.STATIC) ? 0 : 1;
        for (ClassDesc p : mm.methodTypeSymbol().parameterList()) {
            slots += p == CD_long || p == CD_double ? 2 : 1;
        }
        return slots;
    }

    private final List<BitSet> maps;
    private final BitSet doubleSlots;
    private final int[] slotMap;

    private LocalsCompactor(CodeModel com, int fixedSlots) {
        var frames = com.findAttribute(Attributes.stackMapTable()).map(
                smta -> smta.entries().stream().collect(
                        Collectors.toMap(StackMapFrameInfo::target, StackMapFrameInfo::locals)))
                .orElse(Map.of());
        var exceptionHandlers = com.exceptionHandlers();
        maps = new ArrayList<>();
        doubleSlots = new BitSet();
        int pc = 0;
        // Initialization of fixed slots
        for (int slot = 0; slot < fixedSlots; slot++) {
            getMap(slot).set(0);
        }
        for (var e : com) {
            switch(e) {
                case LabelTarget lt -> {
                    var frame = frames.get(lt.label());
                    if (frame != null) {
                        int slot = 0;
                        for (var vti : frame) {
                            boolean doubleSlot = vti == ITEM_LONG || vti == ITEM_DOUBLE;
                            if (vti != ITEM_TOP) {
                                store(pc, slot, doubleSlot);
                            }
                            slot += doubleSlot ? 2 : 1;
                        }
                    }
                    for (var eh : exceptionHandlers) {
                        if (eh.tryStart() == lt.label()) {
                            load(pc, frames.get(eh.handler()));
                        }
                    }
                }
                case LoadInstruction li ->
                    load(pc, li.slot(), li.typeKind());
                case StoreInstruction si ->
                    store(pc, si.slot(), si.typeKind() == TypeKind.LongType || si.typeKind() == TypeKind.DoubleType);
                case IncrementInstruction ii ->
                    load(pc, ii.slot(), false);
                case BranchInstruction bi ->
                    load(pc, frames.get(bi.target()));
                case LookupSwitchInstruction si -> {
                    load(pc, frames.get(si.defaultTarget()));
                    for (var sc : si.cases()) {
                        load(pc, frames.get(sc.target()));
                    }
                }
                case TableSwitchInstruction si -> {
                    load(pc, frames.get(si.defaultTarget()));
                    for (var sc : si.cases()) {
                        load(pc, frames.get(sc.target()));
                    }
                }
                default -> {}
            }
            pc++;
        }
        slotMap = new int[maps.size()];
        List<Boolean> isDouble = new ArrayList<>(slotMap.length);
        for (int slot = 0; slot < slotMap.length; slot++) {
            slotMap[slot] = slot;
            isDouble.add(doubleSlots.get(slot));
        }
        for (int targetSlot = 0; targetSlot < maps.size() - 1; targetSlot++) {
            for (int sourceSlot = Math.max(targetSlot + 1, fixedSlots); sourceSlot < maps.size(); sourceSlot++) {
                if (!isDouble.get(sourceSlot)) {
                    BitSet targetMap = maps.get(targetSlot);
                    BitSet sourceMap = maps.get(sourceSlot);
                    if (!targetMap.intersects(sourceMap)) {
                        targetMap.or(sourceMap);
                        maps.remove(sourceSlot);
                        isDouble.remove(sourceSlot);
                        for (int slot = 0; slot < slotMap.length; slot++) {
                            if (slotMap[slot] == sourceSlot) {
                                slotMap[slot] = targetSlot;
                            } else if (slotMap[slot] > sourceSlot) {
                                slotMap[slot]--;
                            }
                        }
                    }
                }
            }
        }
    }

    private BitSet getMap(int slot) {
        while (slot >= maps.size()) {
            maps.add(new BitSet());
        }
        return maps.get(slot);
    }

    private void load(int pc, int slot, boolean doubleSlot) {
        BitSet map = getMap(slot);
        int start = map.nextSetBit(0) + 1;
        map.set(start, pc + 1);
        if (doubleSlot) {
            doubleSlots.set(slot, slot + 2);
            getMap(slot + 1).set(start, pc + 1);
        }
    }

    private void load(int pc, int slot, TypeKind tk) {
        load(pc, slot, tk == TypeKind.LongType || tk == TypeKind.DoubleType);
    }

    private void load(int pc, List<StackMapFrameInfo.VerificationTypeInfo> frame) {
        if (frame != null) {
            int slot = 0;
            for (var vti : frame) {
                boolean doubleSlot = vti == ITEM_LONG || vti == ITEM_DOUBLE;
                if (vti != ITEM_TOP) {
                    load(pc, slot, doubleSlot);
                }
                slot += doubleSlot ? 2 : 1;
            }
        }
    }

    private void store(int pc, int slot, boolean doubleSlot) {
        getMap(slot).set(pc);
        if (doubleSlot) {
            doubleSlots.set(slot, slot + 2);
            getMap(slot + 1).set(pc);
        }
    }
}
