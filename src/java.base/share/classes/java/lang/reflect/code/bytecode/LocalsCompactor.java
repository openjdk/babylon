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
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.CodeElement;
import java.lang.classfile.CodeModel;
import java.lang.classfile.CodeTransform;
import java.lang.classfile.Label;
import java.lang.classfile.MethodModel;
import java.lang.classfile.TypeKind;
import java.lang.classfile.attribute.StackMapFrameInfo;
import java.lang.classfile.instruction.BranchInstruction;
import java.lang.classfile.instruction.ExceptionCatch;
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
 * LocalsCompactor transforms class to reduce allocation of local slots in the Code attribute (max_locals).
 * It collects slot maps, compacts them and transforms the Code attribute accordingly.
 * <p>
 * Example of maps before compaction (max_locals = 13):
 * <pre>
 *  slots:  0   1   2   3   4   5   6   7   8   9   10  11  12  13
 *  ---------------------------------------------------------------
 *  bci 0:  *   *
 *      8:      *   *   *
 *     10:      *   *   *
 *     15:      *   *   *   *   *
 *     17:      *   *   *   *   *
 *     18:      *           *   *
 *     25:      *                   *   *
 *     27:      *                   *   *
 *     32:      *                   *   *   *   *
 *     34:      *                   *   *   *   *
 *     36:      *                           *   *
 *     43:      *                                   *   *
 *     45:      *                                   *   *
 *     50:                                          *   *   *   *
 *     52:                                          *   *   *   *
 *     54:                                                  *   *
 * </pre>
 * Compact form of the same maps (max_locals = 5):
 * <pre>
 *  slots:   0   1   2   3   4   5
 *         +12 +13  +6  +7  +8  +9
 *                 +10 +11
 *  -------------------------------
 *  bci 0:  *   *
 *      8:      *   *   *
 *     10:      *   *   *
 *     15:      *   *   *   *   *
 *     17:      *   *   *   *   *
 *     18:      *           *   *
 *     25:      *   *   *
 *     27:      *   *   *
 *     32:      *   *   *   *   *
 *     34:      *   *   *   *   *
 *     36:      *           *   *
 *     43:      *   *   *
 *     45:      *   *   *
 *     50:  *   *   *   *
 *     52:  *   *   *   *
 *     54:  *   *
 * </pre>
 */
public final class LocalsCompactor {

    static class ExceptionTableCompactor implements CodeTransform {
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

    public static final ClassTransform INSTANCE = (clb,cle) -> {
        if (cle instanceof MethodModel mm) {
            clb.transformMethod(mm, (mb, me) -> {
                if (me instanceof CodeModel com) {
                    int[] slotMap = new LocalsCompactor(com, countParamSlots(mm)).slotMap;
                    // @@@ ExceptionTableCompactor can be chained on ClassTransform level when the recent Class-File API is merged into code-reflection
                    mb.transformCode(com, new ExceptionTableCompactor().andThen((cob, coe) -> {
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
                    }));
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

    static final class Slot {
        final BitSet map = new BitSet(); // Liveness map of the slot
        int flags; // 0 - single slot, 1 - first of double slots, 2 - second of double slots, 3 - mixed
    }

    private final List<Slot> maps; // Intermediate slots liveness maps
    private final Map<Label, List<StackMapFrameInfo.VerificationTypeInfo>> frames;
    private final int[] slotMap; // Output mapping of the slots

    private LocalsCompactor(CodeModel com, int fixedSlots) {
        frames = com.findAttribute(Attributes.stackMapTable()).map(
                smta -> smta.entries().stream().collect(
                        Collectors.toMap(StackMapFrameInfo::target, StackMapFrameInfo::locals)))
                .orElse(Map.of());
        var exceptionHandlers = com.exceptionHandlers();
        maps = new ArrayList<>();
        int pc = 0;
        // Initialization of fixed slots
        for (int slot = 0; slot < fixedSlots; slot++) {
            getMap(slot).map.set(0);
        }
        // Filling the slots liveness maps
        for (var e : com) {
            switch(e) {
                case LabelTarget lt -> {
                    for (var eh : exceptionHandlers) {
                        if (eh.tryStart() == lt.label()) {
                            mergeFrom(pc, eh.handler());
                        }
                    }
                }
                case LoadInstruction li ->
                    load(pc, li.slot(), li.typeKind());
                case StoreInstruction si ->
                    store(pc, si.slot(), si.typeKind());
                case IncrementInstruction ii ->
                    loadSingle(pc, ii.slot());
                case BranchInstruction bi ->
                    mergeFrom(pc, bi.target());
                case LookupSwitchInstruction si -> {
                    mergeFrom(pc, si.defaultTarget());
                    for (var sc : si.cases()) {
                        mergeFrom(pc, sc.target());
                    }
                }
                case TableSwitchInstruction si -> {
                    mergeFrom(pc, si.defaultTarget());
                    for (var sc : si.cases()) {
                        mergeFrom(pc, sc.target());
                    }
                }
                default -> pc--;
            }
            pc++;
        }
        // Initialization of slots mapping
        slotMap = new int[maps.size()];
        for (int slot = 0; slot < slotMap.length; slot++) {
            slotMap[slot] = slot;
        }
        // Iterative compaction of slots
        for (int targetSlot = 0; targetSlot < maps.size() - 1; targetSlot++) {
            for (int sourceSlot = Math.max(targetSlot + 1, fixedSlots); sourceSlot < maps.size(); sourceSlot++) {
                Slot source = maps.get(sourceSlot);
                // Re-mapping single slot
                if (source.flags == 0) {
                    Slot target = maps.get(targetSlot);
                    if (!target.map.intersects(source.map)) {
                        // Single re-mapping, merge of the liveness maps and shift of the following slots by 1 left
                        target.map.or(source.map);
                        maps.remove(sourceSlot);
                        for (int slot = 0; slot < slotMap.length; slot++) {
                            if (slotMap[slot] == sourceSlot) {
                                slotMap[slot] = targetSlot;
                            } else if (slotMap[slot] > sourceSlot) {
                                slotMap[slot]--;
                            }
                        }
                    }
                } else if (source.flags == 1 && sourceSlot > targetSlot + 1) {
                    Slot source2 = maps.get(sourceSlot + 1);
                    // Re-mapping distinct double slot
                    if (source2.flags == 2) {
                        Slot target = maps.get(targetSlot);
                        Slot target2 = maps.get(targetSlot + 1);
                        if (!target.map.intersects(source.map) && !target2.map.intersects(source2.map)) {
                            // Double re-mapping, merge of the liveness maps and shift of the following slots by 2 left
                            target.map.or(source.map);
                            target2.map.or(source2.map);
                            maps.remove(sourceSlot + 1);
                            maps.remove(sourceSlot);
                            for (int slot = 0; slot < slotMap.length; slot++) {
                                if (slotMap[slot] == sourceSlot) {
                                    slotMap[slot] = targetSlot;
                                } else if (slotMap[slot] == sourceSlot + 1) {
                                    slotMap[slot] = targetSlot + 1;
                                } else if (slotMap[slot] > sourceSlot + 1) {
                                    slotMap[slot] -= 2;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private Slot getMap(int slot) {
        while (slot >= maps.size()) {
            maps.add(new Slot());
        }
        return maps.get(slot);
    }

    private Slot loadSingle(int pc, int slot) {
        Slot s =  getMap(slot);
        int start = s.map.nextSetBit(0) + 1;
        s.map.set(start, pc + 1);
        return s;
    }

    private void load(int pc, int slot, TypeKind tk) {
        load(pc, slot, tk.slotSize() == 2);
    }

    private void load(int pc, int slot, boolean dual) {
        if (dual) {
            loadSingle(pc, slot).flags |= 1;
            loadSingle(pc, slot + 1).flags |= 2;
        } else {
            loadSingle(pc, slot);
        }
    }

    private void mergeFrom(int pc, Label target) {
        int slot = 0;
        for (var vti : frames.get(target)) {
            if (vti != ITEM_TOP) {
                if (vti == ITEM_LONG || vti == ITEM_DOUBLE) {
                    load(pc, slot++, true);
                } else {
                    loadSingle(pc, slot);
                }
            }
            slot++;
        }
    }

    private Slot storeSingle(int pc, int slot) {
        Slot s = getMap(slot);
        s.map.set(pc);
        return s;
    }

    private void store(int pc, int slot, TypeKind tk) {
        if (tk.slotSize() == 2) {
            storeSingle(pc, slot).flags |= 1;
            storeSingle(pc, slot + 1).flags |= 2;
        } else {
            storeSingle(pc, slot);
        }
    }
}
