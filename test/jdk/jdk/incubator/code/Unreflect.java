/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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

import java.lang.classfile.Attributes;
import java.lang.classfile.ClassFile;
import java.lang.classfile.ClassModel;
import java.lang.classfile.CodeModel;
import java.lang.classfile.MethodModel;
import java.lang.classfile.MethodTransform;
import java.lang.classfile.TypeKind;
import java.lang.classfile.constantpool.ClassEntry;
import java.lang.classfile.instruction.InvokeDynamicInstruction;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.DirectMethodHandleDesc;
import java.lang.constant.DynamicCallSiteDesc;
import java.lang.constant.MethodHandleDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.reflect.AccessFlag;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

import jdk.incubator.code.Reflect;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.runtime.CodeModelBootstraps;
import jdk.incubator.code.runtime.ReflectableLambdaMetafactory;

public final class Unreflect {

    static final ClassDesc CD_Reflect = Reflect.class.describeConstable().get();
    static final ClassDesc CD_CodeModelBootstraps = CodeModelBootstraps.class.describeConstable().get();
    static final ClassDesc CD_ReflectableLambdaMetafactory = ReflectableLambdaMetafactory.class.describeConstable().get();
    static final ClassDesc CD_CallerSensitive = ClassDesc.of("jdk.internal.reflect.CallerSensitive");

    static boolean isReflective(MethodModel mm) {
        return mm.findAttribute(Attributes.runtimeVisibleAnnotations())
                 .map(aa -> aa.annotations().stream().anyMatch(a -> a.classSymbol().equals(CD_Reflect)))
                 .orElse(false);
    }

    static byte[] transform(ClassModel clm) {
        Set<String> modelAccessors = new HashSet<>();
        MethodTypeDesc modelType = MethodTypeDesc.of(Op.class.describeConstable().orElseThrow());
        clm.methods().stream().filter(m -> m.methodTypeSymbol().equals(modelType))
                .forEach(m -> modelAccessors.add(m.methodName().stringValue()));
        return ClassFile.of(ClassFile.ConstantPoolSharingOption.NEW_POOL).transformClass(clm, (clb, cle) -> {
            if (cle instanceof MethodModel mm) {
                if (mm.methodName().equalsString("<init>") || mm.code().isEmpty()
                        || mm.findAttribute(Attributes.runtimeVisibleAnnotations())
                                .map(a -> a.annotations().stream().anyMatch(ann -> ann.classSymbol().equals(CD_CallerSensitive)))
                                .orElse(false)) {
                    clb.with(mm);
                } else if (isReflective(mm) || modelAccessors.contains(modelAccessorName(clm, mm))) {
                    clb.transformMethod(mm, MethodTransform.dropping(me -> me instanceof CodeModel)
                            .andThen(MethodTransform.endHandler(mb -> mb.withCode(cob -> {
                                MethodTypeDesc mts = mm.methodTypeSymbol();
                                boolean hasReceiver = !mm.flags().has(AccessFlag.STATIC);
                                if (hasReceiver) {
                                    cob.aload(cob.receiverSlot());
                                }
                                for (int i = 0; i < mts.parameterCount(); i++) {
                                    cob.loadLocal(TypeKind.from(mts.parameterType(i)), cob.parameterSlot(i));
                                }
                                boolean isInterface = clm.flags().has(AccessFlag.INTERFACE);
                                DirectMethodHandleDesc.Kind kind = hasReceiver
                                        ? (isInterface ? DirectMethodHandleDesc.Kind.INTERFACE_VIRTUAL : DirectMethodHandleDesc.Kind.VIRTUAL)
                                        : (isInterface ? DirectMethodHandleDesc.Kind.INTERFACE_STATIC : DirectMethodHandleDesc.Kind.STATIC);
                                cob.invokedynamic(DynamicCallSiteDesc.of(ConstantDescs.ofCallsiteBootstrap(CD_CodeModelBootstraps, "linkMethod", ConstantDescs.CD_CallSite, ConstantDescs.CD_MethodHandle),
                                        mm.methodName().stringValue(),
                                        hasReceiver ? mts.insertParameterTypes(0, clm.thisClass().asSymbol()) : mts,
                                        MethodHandleDesc.ofMethod(kind, clm.thisClass().asSymbol(), mm.methodName().stringValue(), mts)));
                                cob.return_(TypeKind.from(mts.returnType()));
                            }))));
                } else {
                    clb.transformMethod(mm, MethodTransform.transformingCode((cob, coe) -> {
                        DirectMethodHandleDesc bsm;
                        if (coe instanceof InvokeDynamicInstruction i
                                && (bsm = i.bootstrapMethod()).owner().equals(CD_ReflectableLambdaMetafactory)) {
                            // redirect metafactory and altMetafactory
                            cob.invokedynamic(DynamicCallSiteDesc.of(MethodHandleDesc.ofMethod(DirectMethodHandleDesc.Kind.STATIC,
                                                              CD_CodeModelBootstraps,
                                                              bsm.methodName(),
                                                              bsm.invocationType()),
                                    i.name().stringValue(),
                                    MethodTypeDesc.ofDescriptor(i.type().stringValue()),
                                    i.bootstrapArgs().toArray(ConstantDesc[]::new)));
                        } else {
                            cob.with(coe);
                        }
                    }));
                }
            } else {
                clb.with(cle);
            }
        });
    }

    static String modelAccessorName(ClassModel clm, MethodModel mm) {
        MethodTypeDesc type = mm.methodTypeSymbol();
        return MethodRef.method(JavaType.type(clm.thisClass().asSymbol()), mm.methodName().stringValue(),
                JavaType.type(type.returnType()), type.parameterList().stream().map(JavaType::type).toList())
                .toString().replace('.', '$').replace(';', '$').replace('[', '$').replace('/', '$');
    }

    public static void main(String[] args) throws Exception {
        // process class files from arguments
        var toUnreflect = new ArrayDeque<String>();
        for (String arg : args) {
            Path path = Path.of(arg);
            if (Files.isDirectory(path)) {
                try (var files = Files.walk(path)) {
                    for (Path file : files.filter(Files::isRegularFile)
                            .filter(p -> p.toString().endsWith(".class")).toList()) {
                        System.out.println("unreflecting " + file);
                        Files.write(file, transform(ClassFile.of().parse(Files.readAllBytes(file))));
                    }
                }
            } else {
                toUnreflect.add(arg);
            }
        }
        var done = new HashSet<String>();
        while (!toUnreflect.isEmpty()) {
            String arg = toUnreflect.pop();
            if (!arg.endsWith(".class")) arg += ".class";
            if (done.add(arg)) {
                System.out.println("unreflecting " + arg);
                Path clsFile = Path.of(Unreflect.class.getResource(arg).toURI());
                ClassModel clm = ClassFile.of().parse(Files.readAllBytes(clsFile));
                // unreflect all nest members
                clm.findAttribute(Attributes.nestMembers())
                        .ifPresent(nma -> toUnreflect.addAll(
                                nma.nestMembers().stream().map(ClassEntry::asInternalName).toList()));
                Files.write(clsFile, transform(clm));
            }
        }
    }
}
