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

import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.Label;
import java.lang.classfile.MethodModel;
import java.lang.classfile.Opcode;
import java.lang.classfile.instruction.*;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.AccessFlag;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.reflect.code.bytecode.BytecodeLift;
import java.lang.reflect.code.op.CoreOps;
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
    private static final int COLUMN_WIDTH = 150;

    private int passed, notMatching;
    private Map<String, Map<String, Integer>> errorStats;

    @Test
    public void testDoubleRoundtripStability() throws Exception {
        passed = 0;
        notMatching = 0;
        errorStats = new LinkedHashMap<>();
        for (Path p : Files.walk(JRT.getPath("modules/java.base/java"))
                .filter(p -> Files.isRegularFile(p) && p.toString().endsWith(".class"))
                .toList()) {
            testDoubleRoundtripStability(p);
        }

        for (var stats : errorStats.entrySet()) {
            System.out.println(STR."""

            \{stats.getKey()} errors:
            -----------------------------------------------------
            """);
            stats.getValue().entrySet().stream().sorted((e1, e2) -> Integer.compare(e2.getValue(), e1.getValue())).forEach(e -> System.out.println(e.getValue() +"x " + e.getKey() + "\n"));
        }


        // @@@ There is still several failing cases and a lot of errors
        Assert.assertTrue(notMatching < 31 && passed > 5400, STR."""

                    passed: \{passed}
                    not matching: \{notMatching}
                    \{errorStats.entrySet().stream().map(e -> e.getKey() + " errors: "
                            + e.getValue().values().stream().collect(Collectors.summingInt(Integer::intValue))).collect(Collectors.joining("\n    "))}
                """);
    }

    private void testDoubleRoundtripStability(Path path) throws Exception {
        var clm = CF.parse(path);
        for (var originalModel : clm.methods()) {
            if (originalModel.flags().has(AccessFlag.STATIC) && originalModel.code().isPresent()) try {
                CoreOps.FuncOp firstLift = lift(originalModel);
                try {
                    CoreOps.FuncOp firstTransform = transform(firstLift);
                    try {
                        MethodModel firstModel = lower(firstTransform);
                        try {
                            CoreOps.FuncOp secondLift = lift(firstModel);
                            try {
                                CoreOps.FuncOp secondTransform = transform(secondLift);
                                try {
                                    MethodModel secondModel = lower(secondTransform);

                                    // testing only methods passing through
                                    var firstNormalized = normalize(firstModel);
                                    var secondNormalized = normalize(secondModel);
                                    if (!firstNormalized.equals(secondNormalized)) {
                                        notMatching++;
                                        System.out.println(clm.thisClass().asInternalName() + "::" + originalModel.methodName().stringValue() + originalModel.methodTypeSymbol().displayDescriptor());
                                        printInColumns(firstLift, secondLift);
                                        printInColumns(firstTransform, secondTransform);
                                        printInColumns(firstNormalized, secondNormalized);
                                        System.out.println();
                                    } else {
                                        passed++;
                                    }
                                } catch (Exception e) {
                                    error("second lower", e);
                                }
                            } catch (Exception e) {
                                error("second transform", e);
                            }
                        } catch (Exception e) {
                            error("second lift", e);
                        }
                    } catch (Exception e) {
                        error("first lower", e);
                    }
                } catch (Exception e) {
                    error("first transform", e);
                }
            } catch (Exception e) {
                error("first lift", e);
            }
        }
    }
    private static void printInColumns(CoreOps.FuncOp first, CoreOps.FuncOp second) {
        StringWriter fw = new StringWriter();
        first.writeTo(fw);
        StringWriter sw = new StringWriter();
        second.writeTo(sw);
        printInColumns(fw.toString().lines().toList(), sw.toString().lines().toList());
    }

    private static void printInColumns(List<String> first, List<String> second) {
        System.out.println("-".repeat(COLUMN_WIDTH ) + "--+-" + "-".repeat(COLUMN_WIDTH ));
        for (int i = 0; i < first.size() || i < second.size(); i++) {
            String f = i < first.size() ? first.get(i) : "";
            String s = i < second.size() ? second.get(i) : "";
            System.out.println(" " + f + (f.length() < COLUMN_WIDTH ? " ".repeat(COLUMN_WIDTH - f.length()) : "") + (f.equals(s) ? " | " : " x ") + s);
        }
    }

    private static CoreOps.FuncOp lift(MethodModel mm) {
        return BytecodeLift.lift(mm);
    }

    private static CoreOps.FuncOp transform(CoreOps.FuncOp func) {
        return SSA.transform(func.transform((block, op) -> {
                    if (op instanceof Op.Lowerable lop) {
                        return lop.lower(block);
                    } else {
                        block.op(op);
                        return block;
                    }
                }));
    }

    private static MethodModel lower(CoreOps.FuncOp func) {
        return CF.parse(BytecodeGenerator.generateClassData(
                MethodHandles.lookup(),
                func)).methods().get(0);
    }


    public static List<String> normalize(MethodModel mm) {
        record El(int index, String format, Label... targets) {
            public El(int index, Instruction i, Object format, Label... targets) {
                this(index, trim(i.opcode()) + " " + format, targets);
            }
            public String toString(Map<Label, Integer> targetsMap) {
                return "%3d: ".formatted(index) + (targets.length == 0 ? format : format.formatted(Stream.of(targets).map(l -> targetsMap.get(l)).toArray()));
            }
        }

        Map<Label, Integer> targetsMap = new HashMap<>();
        List<El> elements = new ArrayList<>();
        Label lastLabel = null;
        int i = 0;
        for (var e : mm.code().orElseThrow()) {
            var er = switch (e) {
                case LabelTarget lt -> {
                    lastLabel = lt.label();
                    yield null;
                }
                case ExceptionCatch ec ->
                    new El(i++, "ExceptionCatch start: @%d end: @%d handler: @%d" + ec.catchType().map(ct -> " catch type: " + ct.asInternalName()).orElse(""), ec.tryStart(), ec.tryEnd(), ec.handler());
                case BranchInstruction ins ->
                    new El(i++, ins, "@%d", ins.target());
                case ConstantInstruction ins ->
                    new El(i++, "LDC " + ins.constantValue());
                case FieldInstruction ins ->
                    new El(i++, ins, ins.owner().asInternalName() + "." + ins.name().stringValue());
                case InvokeDynamicInstruction ins ->
                    new El(i++, ins, ins.name().stringValue() + ins.typeSymbol() + " " + ins.bootstrapMethod() + "(" + ins.bootstrapArgs() + ")");
                case InvokeInstruction ins ->
                    new El(i++, ins, ins.owner().asInternalName() + "::" + ins.name().stringValue() + ins.typeSymbol().displayDescriptor());
                case LoadInstruction ins ->
                    new El(i++, ins, "#" + ins.slot());
                case StoreInstruction ins ->
                    new El(i++, ins, "#" + ins.slot());
                case IncrementInstruction ins ->
                    new El(i++, ins, "#" + ins.slot() + " " + ins.constant());
                case LookupSwitchInstruction ins ->
                    new El(i++, ins, "default: @%d" + ins.cases().stream().map(c -> ", " + c.caseValue() + ": @%d").collect(Collectors.joining()),
                            Stream.concat(Stream.of(ins.defaultTarget()), ins.cases().stream().map(SwitchCase::target)).toArray(Label[]::new));
                case NewMultiArrayInstruction ins ->
                    new El(i++, ins, ins.arrayType().asInternalName() + "(" + ins.dimensions() + ")");
                case NewObjectInstruction ins ->
                    new El(i++, ins, ins.className().asInternalName());
                case NewPrimitiveArrayInstruction ins ->
                    new El(i++, ins, ins.typeKind());
                case NewReferenceArrayInstruction ins ->
                    new El(i++, ins, ins.componentType().asInternalName());
                case TableSwitchInstruction ins ->
                    new El(i++, ins, "default: @%d" + ins.cases().stream().map(c -> ", " + c.caseValue() + ": @%d").collect(Collectors.joining()),
                            Stream.concat(Stream.of(ins.defaultTarget()), ins.cases().stream().map(SwitchCase::target)).toArray(Label[]::new));
                case TypeCheckInstruction ins ->
                    new El(i++, ins, ins.type().asInternalName());
                case Instruction ins ->
                    new El(i++, ins, "");
                default -> null;
            };
            if (er != null) {
                if (lastLabel != null) {
                    targetsMap.put(lastLabel, elements.size());
                    lastLabel = null;
                }
                elements.add(er);
            }
        }
        return elements.stream().map(el -> el.toString(targetsMap)).toList();
    }

    private static String trim(Opcode opcode) {
        var name = opcode.toString();
        int i = name.indexOf('_');
        return i > 2 ? name.substring(0, i) : name;
    }

    private void error(String category, Exception e) {
        StringWriter sw = new StringWriter();
        e.printStackTrace(new PrintWriter(sw));
        errorStats.computeIfAbsent(category, _ -> new HashMap<>())
                  .compute(sw.toString(), (_, i) -> i == null ? 1 : i + 1);
    }
}
