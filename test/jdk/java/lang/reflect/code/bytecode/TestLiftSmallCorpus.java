/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
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

import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.Label;
import java.lang.classfile.MethodModel;
import java.lang.classfile.Opcode;
import java.lang.classfile.instruction.*;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.reflect.code.bytecode.BytecodeLift;

import java.net.URI;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.testng.Assert;
import org.testng.annotations.Test;

/*
 * @test
 * @enablePreview
 * @run testng TestLiftSmallCorpus
 */

public class TestLiftSmallCorpus {

    private static final FileSystem JRT = FileSystems.getFileSystem(URI.create("jrt:/"));
    private static final ClassFile CF = ClassFile.of(ClassFile.DebugElementsOption.DROP_DEBUG,
                                                     ClassFile.LineNumbersOption.DROP_LINE_NUMBERS);
    private static final int COLUMN_WIDTH = 80;

    @Test
    public void testDoubleRoundTripStability() throws Exception {
        boolean passed = true;
        for (Path p : Files.walk(JRT.getPath("modules/java.base/java"))
                .filter(p -> Files.isRegularFile(p) && p.toString().endsWith(".class"))
                .toList()) {
            passed &= testDoubleRoundTripStability(p);
        }
        Assert.assertTrue(passed);
    }

    private static boolean testDoubleRoundTripStability(Path path) throws Exception {
        var clm = CF.parse(path);
        boolean passed = true;
        for (var originalMethod : clm.methods()) {
            MethodModel firstRound = null, secondRound = null;
            try {
                firstRound = roundTrip(originalMethod);
                secondRound = roundTrip(firstRound);
            } catch (Exception e) {
                //ignore for now
            }

            if (secondRound != null) {
                // test only methods passing lift and generation
                var firstNormalized = normalize(firstRound);
                var secondNormalized = normalize(secondRound);
                if (!firstNormalized.equals(secondNormalized)) {
                    passed = false;
                    System.out.println(clm.thisClass().asInternalName() + "::" + originalMethod.methodName().stringValue() + originalMethod.methodTypeSymbol().displayDescriptor());
                    for (int i = 0; i < firstNormalized.size() || i < secondNormalized.size(); i++) {
                        String s = i < firstNormalized.size() ? firstNormalized.get(i) : "";
                        System.out.println("    " + s + (s.length() < COLUMN_WIDTH ? " ".repeat(COLUMN_WIDTH - s.length()) : "") + " | " + (i < secondNormalized.size() ? secondNormalized.get(i) : ""));
                    }
                    System.out.println();
                }
            }
        }
        return passed;
    }

    private static MethodModel roundTrip(MethodModel mm) {
        return CF.parse(BytecodeGenerator.generateClassData(
                MethodHandles.lookup(),
                SSA.transform(BytecodeLift.lift(mm).transform((block, op) -> {
                    if (op instanceof Op.Lowerable lop) {
                        return lop.lower(block);
                    } else {
                        block.op(op);
                        return block;
                    }
                })))).methods().get(0);
    }


    public static List<String> normalize(MethodModel mm) {
        record ElementRecord(String format, Label... targets) {
            public String toString(Map<Label, ElementRecord> targetsMap) {
                return (targets.length == 0) ? format : format.toString().formatted(Stream.of(targets).map(l -> targetsMap.get(l).toString(targetsMap)).toArray());
            }
        }

        Map<Label, ElementRecord> targetsMap = new HashMap<>();
        List<ElementRecord> elements = new ArrayList<>();
        Label lastLabel = null;
        for (var e : mm.code().orElseThrow()) {
            var er = switch (e) {
                case LabelTarget lt -> {
                    lastLabel = lt.label();
                    yield null;
                }
                case ExceptionCatch ec ->
                    new ElementRecord("ExceptionCatch start:(%s) end:(%s) handler:(%s)" + ec.catchType().map(ct -> " catch type: " + ct.asInternalName()).orElse(""), ec.tryStart(), ec.tryEnd(), ec.handler());
                case BranchInstruction i ->
                    new ElementRecord(trim(i.opcode()) + " target:(%s)", i.target());
                case ConstantInstruction i ->
                    new ElementRecord("LDC " + i.constantValue());
                case FieldInstruction i ->
                    new ElementRecord(trim(i.opcode()) + " " + i.owner().asInternalName() + "." + i.name().stringValue());
                case InvokeDynamicInstruction i ->
                    new ElementRecord(trim(i.opcode())+ " " + i.name().stringValue() + i.typeSymbol() + " " + i.bootstrapMethod() + "(" + i.bootstrapArgs() + ")");
                case InvokeInstruction i ->
                    new ElementRecord(trim(i.opcode()) + " " + i.owner().asInternalName() + "::" + i.name().stringValue() + i.typeSymbol().displayDescriptor());
                case LoadInstruction i ->
                    new ElementRecord(trim(i.opcode()) + " " + i.slot());
                case StoreInstruction i ->
                    new ElementRecord(trim(i.opcode()) + " " + i.slot());
                case IncrementInstruction i ->
                    new ElementRecord(trim(i.opcode()) + " " + i.slot() + " " + i.constant());
                case LookupSwitchInstruction i ->
                    new ElementRecord(trim(i.opcode()) + " default:(%s)" + i.cases().stream().map(c -> ", " + c.caseValue() + ":(%s)").collect(Collectors.joining()),
                            Stream.concat(Stream.of(i.defaultTarget()), i.cases().stream().map(SwitchCase::target)).toArray(Label[]::new));
                case NewMultiArrayInstruction i ->
                    new ElementRecord(trim(i.opcode()) + " " + i.arrayType().asInternalName() + "(" + i.dimensions() + ")");
                case NewObjectInstruction i ->
                    new ElementRecord(trim(i.opcode()) + " " + i.className().asInternalName());
                case NewPrimitiveArrayInstruction i ->
                    new ElementRecord(trim(i.opcode()) + " " + i.typeKind());
                case NewReferenceArrayInstruction i ->
                    new ElementRecord(trim(i.opcode()) + " " + i.componentType().asInternalName());
                case TableSwitchInstruction i ->
                    new ElementRecord(trim(i.opcode()) + " default:(%s)" + i.cases().stream().map(c -> ", " + c.caseValue() + ":(%s)").collect(Collectors.joining()),
                            Stream.concat(Stream.of(i.defaultTarget()), i.cases().stream().map(SwitchCase::target)).toArray(Label[]::new));
                case TypeCheckInstruction i ->
                    new ElementRecord(trim(i.opcode()) + " " + i.type().asInternalName());
                case Instruction i ->
                    new ElementRecord(trim(i.opcode()));
                default -> null;
            };
            if (er != null) {
                elements.add(er);
                if (lastLabel != null) {
                    targetsMap.put(lastLabel, er);
                    lastLabel = null;
                }
            }
        }
        return elements.stream().map(el -> el.toString(targetsMap)).toList();
    }

    private static String trim(Opcode opcode) {
        var name = opcode.toString();
        int i = name.indexOf('_');
        return i > 0 ? name.substring(0, i) : name;
    }
}
