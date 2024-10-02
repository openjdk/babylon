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

import java.io.StringWriter;
import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.Label;
import java.lang.classfile.MethodModel;
import java.lang.classfile.Opcode;
import java.lang.classfile.attribute.CodeAttribute;
import java.lang.classfile.components.ClassPrinter;
import java.lang.classfile.instruction.*;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.reflect.code.bytecode.BytecodeLift;
import java.lang.reflect.code.interpreter.Verifier;
import java.lang.reflect.code.op.CoreOp;
import java.net.URI;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.testng.Assert;
import org.testng.annotations.Ignore;
import org.testng.annotations.Test;

/*
 * @test
 * @enablePreview
 * @modules java.base/java.lang.invoke:open
 * @run testng TestSmallCorpus
 */
public class TestSmallCorpus {

    private static final String ROOT_PATH = "modules/java.base/";
    private static final String CLASS_NAME_SUFFIX = ".class";
    private static final String METHOD_NAME = null;
    private static final int ROUNDS = 3;

    private static final FileSystem JRT = FileSystems.getFileSystem(URI.create("jrt:/"));
    private static final ClassFile CF = ClassFile.of();
    private static final int COLUMN_WIDTH = 150;
    private static final MethodHandles.Lookup TRUSTED_LOOKUP;
    static {
        try {
            var lf = MethodHandles.Lookup.class.getDeclaredField("IMPL_LOOKUP");
            lf.setAccessible(true);
            TRUSTED_LOOKUP = (MethodHandles.Lookup)lf.get(null);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    private MethodModel bytecode;
    CoreOp.FuncOp reflection;
    private int stable, unstable;
    private Long[] stats = new Long[6];

    @Ignore
    @Test
    public void testRoundTripStability() throws Exception {
        stable = 0;
        unstable = 0;
        Arrays.fill(stats, 0l);
        for (Path p : Files.walk(JRT.getPath(ROOT_PATH))
                .filter(p -> Files.isRegularFile(p) && p.toString().endsWith(CLASS_NAME_SUFFIX))
                .toList()) {
            testRoundTripStability(p);
        }

        System.out.println("""
        statistics     original  generated
        code length: %1$,10d %4$,10d
        max locals:  %2$,10d %5$,10d
        max stack:   %3$,10d %6$,10d
        """.formatted((Object[])stats));

        // Roundtrip is >99% stable, no exceptions, no verification errors
        Assert.assertTrue(stable > 54000 && unstable < 100, String.format("stable: %d unstable: %d", stable, unstable));
    }

    private void testRoundTripStability(Path path) throws Exception {
        var clm = CF.parse(path);
        for (var originalModel : clm.methods()) {
            if (originalModel.code().isPresent() && (METHOD_NAME == null || originalModel.methodName().equalsString(METHOD_NAME))) try {
                bytecode = originalModel;
                reflection = null;
                MethodModel prevBytecode = null;
                CoreOp.FuncOp prevReflection = null;
                for (int round = 1; round <= ROUNDS; round++) try {
                    prevBytecode = bytecode;
                    prevReflection = reflection;
                    lift();
                    verifyReflection();
                    generate();
                    verifyBytecode();
                } catch (UnsupportedOperationException uoe) {
                    throw uoe;
                } catch (Throwable t) {
                    System.out.println(" at " + path + " " + originalModel.methodName() + originalModel.methodType() + " round " + round);
                    throw t;
                }
                if (ROUNDS > 0) {
                    var normPrevBytecode = normalize(prevBytecode);
                    var normBytecode = normalize(bytecode);
                    if (normPrevBytecode.equals(normBytecode)) {
                        stable++;
                    } else {
                        unstable++;
                        System.out.println("Unstable code " + path + " " + originalModel.methodName() + originalModel.methodType() + " after " + ROUNDS +" round(s)");
                        if (prevReflection != null) printInColumns(prevReflection, reflection);
                        printInColumns(normPrevBytecode, normBytecode);
                        System.out.println();
                    }
                    var ca = (CodeAttribute)originalModel.code().get();
                    stats[0] += ca.codeLength();
                    stats[1] += ca.maxLocals();
                    stats[2] += ca.maxStack();
                    ca = (CodeAttribute)bytecode.code().get();
                    stats[3] += ca.codeLength();
                    stats[4] += ca.maxLocals();
                    stats[5] += ca.maxStack();
                }
            } catch (UnsupportedOperationException uoe) {
                // InvokeOp when InvokeKind == SUPER
            }
        }
    }

    private void verifyReflection() {
        var errors = Verifier.verify(TRUSTED_LOOKUP, reflection);
        if (!errors.isEmpty()) {
            printBytecode();
            System.out.println("Code reflection model verification failed");
            errors.forEach(System.out::println);
            System.out.println(errors.getFirst().getPrintedContext());
            throw errors.getFirst();
        }
    }

    private void verifyBytecode() {
        for (var e : ClassFile.of().verify(bytecode.parent().get())) {
            if (!e.getMessage().contains("Illegal call to internal method")) {
                printReflection();
                printBytecode();
                System.out.println("Bytecode verification failed");
                throw e;
            }
        }
    }

    private static void printInColumns(CoreOp.FuncOp first, CoreOp.FuncOp second) {
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

    private void lift() {
        try {
            reflection = BytecodeLift.lift(bytecode);
        } catch (Throwable t) {
            printReflection();
            printBytecode();
            System.out.println("Lift failed");
            throw t;
        }
    }

    private void generate() {
        try {
            bytecode = CF.parse(BytecodeGenerator.generateClassData(
                TRUSTED_LOOKUP,
                reflection)).methods().getFirst();
        } catch (Throwable t) {
            printBytecode();
            printReflection();
            System.out.println("Generation failed");
            throw t;
        }
    }

    private void printBytecode() {
        ClassPrinter.toYaml(bytecode, ClassPrinter.Verbosity.CRITICAL_ATTRIBUTES, System.out::print);
    }

    private void printReflection() {
        if (reflection != null) System.out.println(reflection.toText());
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
}
