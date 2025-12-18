/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
import java.lang.classfile.ClassBuilder;
import java.lang.classfile.ClassElement;
import java.lang.classfile.ClassFile;
import java.lang.classfile.ClassModel;
import java.lang.classfile.ClassTransform;
import java.lang.classfile.CodeModel;
import java.lang.classfile.MethodModel;
import java.lang.classfile.MethodTransform;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.DirectMethodHandleDesc;
import java.lang.constant.DynamicCallSiteDesc;
import java.lang.constant.MethodHandleDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.invoke.CallSite;
import java.lang.invoke.ConstantCallSite;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.AccessFlag;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.bytecode.BytecodeGenerator;

public final class Unreflect {

    static final ClassDesc CD_Reflect = Reflect.class.describeConstable().get();

    // lazy initialized
    static MethodModel UNREFLECT_BSM;

    static boolean isReflective(MethodModel mm) {
        return mm.findAttribute(Attributes.runtimeVisibleAnnotations())
                 .map(aa -> aa.annotations().stream().anyMatch(a -> a.classSymbol().equals(CD_Reflect)))
                 .orElse(false);
    }

    static byte[] transform(byte[] classBytes) {
        ClassModel clm = ClassFile.of().parse(classBytes);
        return ClassFile.of(ClassFile.ConstantPoolSharingOption.NEW_POOL).transformClass(clm, new ClassTransform() {
            boolean injectBSM = false;

            @Override
            public void accept(ClassBuilder clb, ClassElement cle) {
                if (cle instanceof MethodModel mm) {
                    if (isReflective(mm)) {
                        clb.transformMethod(mm, MethodTransform.dropping(me -> me instanceof CodeModel)
                                .andThen(MethodTransform.endHandler(mb -> mb.withCode(cob -> {
                                    System.out.print('.');
                                    injectBSM = true;
                                    MethodTypeDesc mts = mm.methodTypeSymbol();
                                    boolean hasReceiver = !mm.flags().has(AccessFlag.STATIC);
                                    if (hasReceiver) {
                                        cob.loadLocal(TypeKind.REFERENCE, cob.receiverSlot());
                                    }
                                    for (int i = 0; i < mts.parameterCount(); i++) {
                                        cob.loadLocal(TypeKind.from(mts.parameterType(i)), cob.parameterSlot(i));
                                    }
                                    cob.invokedynamic(DynamicCallSiteDesc.of(MethodHandleDesc.ofMethod(clm.flags().has(AccessFlag.INTERFACE)
                                                            ? DirectMethodHandleDesc.Kind.INTERFACE_STATIC
                                                            : DirectMethodHandleDesc.Kind.STATIC,
                                                    clm.thisClass().asSymbol(),
                                                    UNREFLECT_BSM.methodName().stringValue(),
                                                    UNREFLECT_BSM.methodTypeSymbol()),
                                            mm.methodName().stringValue(),
                                            hasReceiver ? mts.insertParameterTypes(0, clm.thisClass().asSymbol()): mts));
                                    cob.return_(TypeKind.from(mts.returnType()));
                                }))));
                    } else if (!mm.methodName().equals(UNREFLECT_BSM.methodName())) { // drop unreflect BSM if already present
                        clb.with(cle);
                    }
                } else {
                    clb.with(cle);
                }
            }

            @Override
            public void atEnd(ClassBuilder clb) {
                if (injectBSM) {
                    System.out.println();
                    clb.with(UNREFLECT_BSM);
                }
            }
        });
    }

    // BSM template
    private static CallSite $unreflectBSM(MethodHandles.Lookup caller, String methodName, MethodType methodType) throws NoSuchMethodException {
        for (Method m : caller.lookupClass().getDeclaredMethods()) {
            int firstParam;
            if (m.getName().equals(methodName)
                    && m.getReturnType() == methodType.returnType()
                    && m.getParameterCount() == methodType.parameterCount() - (firstParam = (m.getModifiers() & Modifier.STATIC) == 0 ? 1 : 0)
                    && Arrays.equals(m.getParameterTypes(), 0, m.getParameterCount(), methodType.parameterArray(), firstParam, methodType.parameterCount())) {
                return new ConstantCallSite(BytecodeGenerator.generate(caller, Op.ofMethod(m).orElseThrow()));
            }
        }
        throw new NoSuchMethodException(caller.lookupClass().getName() + "." + methodName + methodType);
    }

    public static void main(String[] args) throws Exception {
        // initialize unreflect method models
        UNREFLECT_BSM = ClassFile.of().parse(Unreflect.class.getResourceAsStream("Unreflect.class").readAllBytes())
                .methods().stream().filter(mm -> mm.methodName().equalsString("$unreflectBSM")).findFirst().orElseThrow();

        // process class files from arguments
        for (var arg : args) {
            if (!arg.endsWith(".class")) arg += ".class";
            System.out.println("unreflecting " + arg);
            Path clsFile = Path.of(Unreflect.class.getResource(arg).toURI());
            Files.write(clsFile, transform(Files.readAllBytes(clsFile)));
        }
    }
}