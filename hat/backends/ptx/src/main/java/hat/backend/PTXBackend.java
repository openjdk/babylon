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
package hat.backend;


import hat.ComputeContext;
import hat.NDRange;
import hat.buffer.Buffer;
import hat.callgraph.KernelCallGraph;
import hat.ifacemapper.BoundSchema;
import hat.optools.*;

import java.lang.reflect.code.*;
import java.lang.reflect.code.op.CoreOp;
import java.util.*;

public class PTXBackend extends C99NativeBackend {
    int major;
    int minor;
    String target;
    int addressSize;

    static HashMap<String, String> mathFns;
    final Set<String> usedMathFns;

    public PTXBackend() {
        super("ptx_backend");
        major = 7;
        minor = 5;
        target = "sm_52";
        addressSize = 64;
        mathFns = new HashMap<>();
        loadMathFns();
        usedMathFns = new HashSet<>();
        getBackend(null);
    }

    @Override
    public void computeContextHandoff(ComputeContext computeContext) {
        System.out.println("PTX backend recieved closed closure");
        System.out.println("PTX backend will mutate  " + computeContext.computeCallGraph.entrypoint + computeContext.computeCallGraph.entrypoint.method);
        injectBufferTracking(computeContext.computeCallGraph.entrypoint);
    }

    @Override
    public void dispatchKernel(KernelCallGraph kernelCallGraph, NDRange ndRange, Object... args) {
        // System.out.println("PTX dispatch kernel");
        // Here we recieve a callgraph from the kernel entrypoint
        // The first time we see this we need to convert the kernel entrypoint
        // and rechable methods to PTX.

        // sort the dag by rank means that we get the methods called by the entrypoint in dependency order
        // of course there may not be any of these
        kernelCallGraph.kernelReachableResolvedStream()
                .sorted((lhs, rhs) -> rhs.rank - lhs.rank)
                .forEach(kernelReachableResolvedMethod ->
                        System.out.println(" call to -> "+kernelReachableResolvedMethod.method.getName())
                );

        // System.out.println("Entrypoint ->"+kernelCallGraph.entrypoint.method.getName());
        String code = createCode(kernelCallGraph, new PTXCodeBuilder(), args);
        long programHandle = compileProgram(code);
        if (programOK(programHandle)) {
            long kernelHandle = getKernel(programHandle, kernelCallGraph.entrypoint.method.getName());
            CompiledKernel compiledKernel = new CompiledKernel(this, kernelCallGraph, code, kernelHandle, args);
            compiledKernel.dispatch(ndRange,args);
        }
    }

    public String createCode(KernelCallGraph kernelCallGraph, PTXCodeBuilder builder, Object[] args) {
        StringBuilder out = new StringBuilder();
        StringBuilder invokedMethods = new StringBuilder();
        FuncOpWrapper f = new FuncOpWrapper(kernelCallGraph.entrypoint.funcOpWrapper().op());
        FuncOpWrapper lowered = f.lower();
        HashMap<String, Object> argsMap = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            argsMap.put(f.paramTable().list().get(i).varOp.varName(), args[i]);
        }

        // printing out ptx header (device info)
        builder.ptxHeader(major, minor, target, addressSize);
        out.append(builder.getTextAndReset());

        for (KernelCallGraph.KernelReachableResolvedMethodCall k : kernelCallGraph.kernelReachableResolvedStream().toList()) {
            FuncOpWrapper calledFunc = new FuncOpWrapper(k.funcOpWrapper().op());
            FuncOpWrapper loweredFunc = calledFunc.lower();
            loweredFunc = transformPtrs(loweredFunc, argsMap);
            invokedMethods.append(createFunction(new PTXCodeBuilder(addressSize).nl().nl(), loweredFunc, false));
        }

        lowered = transformPtrs(lowered, argsMap);
        for (String s : usedMathFns) {
            out.append("\n").append(mathFns.get(s)).append("\n");
        }

        out.append(invokedMethods);

        out.append(createFunction(builder.nl().nl(), lowered, true));

        return out.toString();
    }

    public FuncOpWrapper transformPtrs(FuncOpWrapper func, HashMap<String, Object> argsMap) {
        return FuncOpWrapper.wrap(func.op().transform((block, op) -> {
            CopyContext cc = block.context();
            // use first operand of invoke to figure out schema
            if (op instanceof CoreOp.InvokeOp invokeOp
                    && OpWrapper.wrap(invokeOp) instanceof InvokeOpWrapper invokeOpWrapper) {
                if (invokeOpWrapper.isIfaceBufferMethod()
                        && invokeOp.operands().getFirst() instanceof Op.Result invokeResult
                        && invokeResult.op().operands().getFirst() instanceof Op.Result varLoadResult
                        && varLoadResult.op() instanceof CoreOp.VarOp varOp
                        && argsMap.get(varOp.varName()) instanceof Buffer buffer) {
                    List<Value> inputOperands = invokeOp.operands();
                    List<Value> outputOperands = cc.getValues(inputOperands);
                    Op.Result inputResult = invokeOp.result();
                    BoundSchema<?> boundSchema = Buffer.getBoundSchema(buffer);
                    PTXPtrOp ptxOp = new PTXPtrOp(inputResult.type(), invokeOp.invokeDescriptor().name(), outputOperands, boundSchema);
                    Op.Result outputResult = block.op(ptxOp);
                    cc.mapValue(inputResult, outputResult);
                } else if (invokeOpWrapper.op().invokeDescriptor().refType().toString().equals("java.lang.Math")
                        && mathFns.containsKey(invokeOpWrapper.op().invokeDescriptor().name() + "_" + invokeOpWrapper.resultType().toString())){
                    usedMathFns.add(invokeOpWrapper.op().invokeDescriptor().name() + "_" + invokeOpWrapper.resultType().toString());
                    block.apply(op);
                } else {
                    block.apply(op);
                }
            } else {
                block.apply(op);
            }
            return block;
        }));
    }

    public String createFunction(PTXCodeBuilder builder, FuncOpWrapper lowered, boolean entry) {
        FuncOpWrapper ssa = lowered.ssa();
        // System.out.println("--------------func--------------");
        // System.out.println(ssa.toText());
        String out, body;

        // building fn info (name, params)
        builder.functionHeader(lowered.functionName(), entry, lowered.op().body().yieldType());

        // printing out params
        builder.parameters(lowered.paramTable().list());

        // building body of fn
        builder.functionPrologue();

        out = builder.getTextAndReset();
        ssa.firstBody().blocks().forEach(block -> builder.blockBody(block, block.ops().stream().map(OpWrapper::wrap)));

        builder.functionEpilogue();
        body = builder.getTextAndReset();

        builder.ptxRegisterDecl();
        out += builder.getText() + body;
        return out;
    }

    public static void loadMathFns() {
        mathFns.put("log_float", """
                .func  (.param .b32 func_retval0) log(
                	.param .b32 log_param_0
                )
                {
                	.reg .pred 	%p<4>;
                	.reg .f32 	%f<36>;
                	.reg .b32 	%r<5>;
                                
                                
                	ld.param.f32 	%f5, [log_param_0];
                	setp.lt.f32 	%p1, %f5, 0f00800000;
                	mul.f32 	%f6, %f5, 0f4B000000;
                	selp.f32 	%f1, %f6, %f5, %p1;
                	selp.f32 	%f7, 0fC1B80000, 0f00000000, %p1;
                	mov.b32 	%r1, %f1;
                	add.s32 	%r2, %r1, -1059760811;
                	and.b32  	%r3, %r2, -8388608;
                	sub.s32 	%r4, %r1, %r3;
                	mov.b32 	%f8, %r4;
                	cvt.rn.f32.s32 	%f9, %r3;
                	mov.f32 	%f10, 0f34000000;
                	fma.rn.f32 	%f11, %f9, %f10, %f7;
                	add.f32 	%f12, %f8, 0fBF800000;
                	mov.f32 	%f13, 0f3E1039F6;
                	mov.f32 	%f14, 0fBE055027;
                	fma.rn.f32 	%f15, %f14, %f12, %f13;
                	mov.f32 	%f16, 0fBDF8CDCC;
                	fma.rn.f32 	%f17, %f15, %f12, %f16;
                	mov.f32 	%f18, 0f3E0F2955;
                	fma.rn.f32 	%f19, %f17, %f12, %f18;
                	mov.f32 	%f20, 0fBE2AD8B9;
                	fma.rn.f32 	%f21, %f19, %f12, %f20;
                	mov.f32 	%f22, 0f3E4CED0B;
                	fma.rn.f32 	%f23, %f21, %f12, %f22;
                	mov.f32 	%f24, 0fBE7FFF22;
                	fma.rn.f32 	%f25, %f23, %f12, %f24;
                	mov.f32 	%f26, 0f3EAAAA78;
                	fma.rn.f32 	%f27, %f25, %f12, %f26;
                	mov.f32 	%f28, 0fBF000000;
                	fma.rn.f32 	%f29, %f27, %f12, %f28;
                	mul.f32 	%f30, %f12, %f29;
                	fma.rn.f32 	%f31, %f30, %f12, %f12;
                	mov.f32 	%f32, 0f3F317218;
                	fma.rn.f32 	%f35, %f11, %f32, %f31;
                	setp.lt.u32 	%p2, %r1, 2139095040;
                	@%p2 bra 	$L__BB0_2;
                                
                	mov.f32 	%f33, 0f7F800000;
                	fma.rn.f32 	%f35, %f1, %f33, %f33;
                                
                $L__BB0_2:
                	setp.eq.f32 	%p3, %f1, 0f00000000;
                	selp.f32 	%f34, 0fFF800000, %f35, %p3;
                	st.param.f32 	[func_retval0+0], %f34;
                	ret;
                                
                }""");
        mathFns.put("log_double", """
                .func  (.param .b64 func_retval0) log(
                	.param .b64 log_param_0
                )
                {
                	.reg .pred 	%p<5>;
                	.reg .f32 	%f<2>;
                	.reg .b32 	%r<28>;
                	.reg .f64 	%fd<59>;
                                
                                
                	ld.param.f64 	%fd56, [log_param_0];
                	{
                	.reg .b32 %temp;\s
                	mov.b64 	{%temp, %r24}, %fd56;
                	}
                	{
                	.reg .b32 %temp;\s
                	mov.b64 	{%r25, %temp}, %fd56;
                	}
                	setp.gt.s32 	%p1, %r24, 1048575;
                	mov.u32 	%r26, -1023;
                	@%p1 bra 	$L__BB0_2;
                                
                	mul.f64 	%fd56, %fd56, 0d4350000000000000;
                	{
                	.reg .b32 %temp;\s
                	mov.b64 	{%temp, %r24}, %fd56;
                	}
                	{
                	.reg .b32 %temp;\s
                	mov.b64 	{%r25, %temp}, %fd56;
                	}
                	mov.u32 	%r26, -1077;
                                
                $L__BB0_2:
                	add.s32 	%r13, %r24, -1;
                	setp.lt.u32 	%p2, %r13, 2146435071;
                	@%p2 bra 	$L__BB0_4;
                	bra.uni 	$L__BB0_3;
                                
                $L__BB0_4:
                	shr.u32 	%r15, %r24, 20;
                	add.s32 	%r27, %r26, %r15;
                	and.b32  	%r16, %r24, -2146435073;
                	or.b32  	%r17, %r16, 1072693248;
                	mov.b64 	%fd57, {%r25, %r17};
                	setp.lt.s32 	%p4, %r17, 1073127583;
                	@%p4 bra 	$L__BB0_6;
                                
                	{
                	.reg .b32 %temp;\s
                	mov.b64 	{%r18, %temp}, %fd57;
                	}
                	{
                	.reg .b32 %temp;\s
                	mov.b64 	{%temp, %r19}, %fd57;
                	}
                	add.s32 	%r20, %r19, -1048576;
                	mov.b64 	%fd57, {%r18, %r20};
                	add.s32 	%r27, %r27, 1;
                                
                $L__BB0_6:
                	add.f64 	%fd12, %fd57, 0d3FF0000000000000;
                	mov.f64 	%fd13, 0d3FF0000000000000;
                	rcp.approx.ftz.f64 	%fd14, %fd12;
                	neg.f64 	%fd15, %fd12;
                	fma.rn.f64 	%fd16, %fd15, %fd14, %fd13;
                	fma.rn.f64 	%fd17, %fd16, %fd16, %fd16;
                	fma.rn.f64 	%fd18, %fd17, %fd14, %fd14;
                	add.f64 	%fd19, %fd57, 0dBFF0000000000000;
                	mul.f64 	%fd20, %fd19, %fd18;
                	fma.rn.f64 	%fd21, %fd19, %fd18, %fd20;
                	mul.f64 	%fd22, %fd21, %fd21;
                	mov.f64 	%fd23, 0d3ED0EE258B7A8B04;
                	mov.f64 	%fd24, 0d3EB1380B3AE80F1E;
                	fma.rn.f64 	%fd25, %fd24, %fd22, %fd23;
                	mov.f64 	%fd26, 0d3EF3B2669F02676F;
                	fma.rn.f64 	%fd27, %fd25, %fd22, %fd26;
                	mov.f64 	%fd28, 0d3F1745CBA9AB0956;
                	fma.rn.f64 	%fd29, %fd27, %fd22, %fd28;
                	mov.f64 	%fd30, 0d3F3C71C72D1B5154;
                	fma.rn.f64 	%fd31, %fd29, %fd22, %fd30;
                	mov.f64 	%fd32, 0d3F624924923BE72D;
                	fma.rn.f64 	%fd33, %fd31, %fd22, %fd32;
                	mov.f64 	%fd34, 0d3F8999999999A3C4;
                	fma.rn.f64 	%fd35, %fd33, %fd22, %fd34;
                	mov.f64 	%fd36, 0d3FB5555555555554;
                	fma.rn.f64 	%fd37, %fd35, %fd22, %fd36;
                	sub.f64 	%fd38, %fd19, %fd21;
                	add.f64 	%fd39, %fd38, %fd38;
                	neg.f64 	%fd40, %fd21;
                	fma.rn.f64 	%fd41, %fd40, %fd19, %fd39;
                	mul.f64 	%fd42, %fd18, %fd41;
                	mul.f64 	%fd43, %fd22, %fd37;
                	fma.rn.f64 	%fd44, %fd43, %fd21, %fd42;
                	xor.b32  	%r21, %r27, -2147483648;
                	mov.u32 	%r22, -2147483648;
                	mov.u32 	%r23, 1127219200;
                	mov.b64 	%fd45, {%r21, %r23};
                	mov.b64 	%fd46, {%r22, %r23};
                	sub.f64 	%fd47, %fd45, %fd46;
                	mov.f64 	%fd48, 0d3FE62E42FEFA39EF;
                	fma.rn.f64 	%fd49, %fd47, %fd48, %fd21;
                	neg.f64 	%fd50, %fd47;
                	fma.rn.f64 	%fd51, %fd50, %fd48, %fd49;
                	sub.f64 	%fd52, %fd51, %fd21;
                	sub.f64 	%fd53, %fd44, %fd52;
                	mov.f64 	%fd54, 0d3C7ABC9E3B39803F;
                	fma.rn.f64 	%fd55, %fd47, %fd54, %fd53;
                	add.f64 	%fd58, %fd49, %fd55;
                	bra.uni 	$L__BB0_7;
                                
                $L__BB0_3:
                	mov.f64 	%fd10, 0d7FF0000000000000;
                	fma.rn.f64 	%fd11, %fd56, %fd10, %fd10;
                	{
                	.reg .b32 %temp;\s
                	mov.b64 	{%temp, %r14}, %fd56;
                	}
                	mov.b32 	%f1, %r14;
                	setp.eq.f32 	%p3, %f1, 0f00000000;
                	selp.f64 	%fd58, 0dFFF0000000000000, %fd11, %p3;
                                
                $L__BB0_7:
                	st.param.f64 	[func_retval0+0], %fd58;
                	ret;
                                
                }""");
        mathFns.put("exp_float", """
                .func  (.param .b32 func_retval0) exp(
                	.param .b32 exp_param_0
                )
                {
                	.reg .f32 	%f<18>;
                	.reg .b32 	%r<3>;
                                
                                
                	ld.param.f32 	%f1, [exp_param_0];
                	mov.f32 	%f2, 0f3F000000;
                	mov.f32 	%f3, 0f3BBB989D;
                	fma.rn.f32 	%f4, %f1, %f3, %f2;
                	mov.f32 	%f5, 0f3FB8AA3B;
                	mov.f32 	%f6, 0f437C0000;
                	cvt.sat.f32.f32 	%f7, %f4;
                	mov.f32 	%f8, 0f4B400001;
                	fma.rm.f32 	%f9, %f7, %f6, %f8;
                	add.f32 	%f10, %f9, 0fCB40007F;
                	neg.f32 	%f11, %f10;
                	fma.rn.f32 	%f12, %f1, %f5, %f11;
                	mov.f32 	%f13, 0f32A57060;
                	fma.rn.f32 	%f14, %f1, %f13, %f12;
                	mov.b32 	%r1, %f9;
                	shl.b32 	%r2, %r1, 23;
                	mov.b32 	%f15, %r2;
                	ex2.approx.ftz.f32 	%f16, %f14;
                	mul.f32 	%f17, %f16, %f15;
                	st.param.f32 	[func_retval0+0], %f17;
                	ret;
                                
                }""");
        mathFns.put("exp_double", """
                .func  (.param .b64 func_retval0) exp(
                	.param .b64 exp_param_0
                )
                {
                	.reg .pred 	%p<4>;
                	.reg .f32 	%f<3>;
                	.reg .b32 	%r<16>;
                	.reg .f64 	%fd<41>;
                                
                                
                	ld.param.f64 	%fd5, [exp_param_0];
                	mov.f64 	%fd6, 0d4338000000000000;
                	mov.f64 	%fd7, 0d3FF71547652B82FE;
                	fma.rn.f64 	%fd8, %fd5, %fd7, %fd6;
                	{
                	.reg .b32 %temp;\s
                	mov.b64 	{%r1, %temp}, %fd8;
                	}
                	mov.f64 	%fd9, 0dC338000000000000;
                	add.rn.f64 	%fd10, %fd8, %fd9;
                	mov.f64 	%fd11, 0dBFE62E42FEFA39EF;
                	fma.rn.f64 	%fd12, %fd10, %fd11, %fd5;
                	mov.f64 	%fd13, 0dBC7ABC9E3B39803F;
                	fma.rn.f64 	%fd14, %fd10, %fd13, %fd12;
                	mov.f64 	%fd15, 0d3E928AF3FCA213EA;
                	mov.f64 	%fd16, 0d3E5ADE1569CE2BDF;
                	fma.rn.f64 	%fd17, %fd16, %fd14, %fd15;
                	mov.f64 	%fd18, 0d3EC71DEE62401315;
                	fma.rn.f64 	%fd19, %fd17, %fd14, %fd18;
                	mov.f64 	%fd20, 0d3EFA01997C89EB71;
                	fma.rn.f64 	%fd21, %fd19, %fd14, %fd20;
                	mov.f64 	%fd22, 0d3F2A01A014761F65;
                	fma.rn.f64 	%fd23, %fd21, %fd14, %fd22;
                	mov.f64 	%fd24, 0d3F56C16C1852B7AF;
                	fma.rn.f64 	%fd25, %fd23, %fd14, %fd24;
                	mov.f64 	%fd26, 0d3F81111111122322;
                	fma.rn.f64 	%fd27, %fd25, %fd14, %fd26;
                	mov.f64 	%fd28, 0d3FA55555555502A1;
                	fma.rn.f64 	%fd29, %fd27, %fd14, %fd28;
                	mov.f64 	%fd30, 0d3FC5555555555511;
                	fma.rn.f64 	%fd31, %fd29, %fd14, %fd30;
                	mov.f64 	%fd32, 0d3FE000000000000B;
                	fma.rn.f64 	%fd33, %fd31, %fd14, %fd32;
                	mov.f64 	%fd34, 0d3FF0000000000000;
                	fma.rn.f64 	%fd35, %fd33, %fd14, %fd34;
                	fma.rn.f64 	%fd36, %fd35, %fd14, %fd34;
                	{
                	.reg .b32 %temp;\s
                	mov.b64 	{%r2, %temp}, %fd36;
                	}
                	{
                	.reg .b32 %temp;\s
                	mov.b64 	{%temp, %r3}, %fd36;
                	}
                	shl.b32 	%r4, %r1, 20;
                	add.s32 	%r5, %r3, %r4;
                	mov.b64 	%fd40, {%r2, %r5};
                	{
                	.reg .b32 %temp;\s
                	mov.b64 	{%temp, %r6}, %fd5;
                	}
                	mov.b32 	%f2, %r6;
                	abs.f32 	%f1, %f2;
                	setp.lt.f32 	%p1, %f1, 0f4086232B;
                	@%p1 bra 	$L__BB0_3;
                                
                	setp.lt.f64 	%p2, %fd5, 0d0000000000000000;
                	add.f64 	%fd37, %fd5, 0d7FF0000000000000;
                	selp.f64 	%fd40, 0d0000000000000000, %fd37, %p2;
                	setp.geu.f32 	%p3, %f1, 0f40874800;
                	@%p3 bra 	$L__BB0_3;
                                
                	shr.u32 	%r7, %r1, 31;
                	add.s32 	%r8, %r1, %r7;
                	shr.s32 	%r9, %r8, 1;
                	shl.b32 	%r10, %r9, 20;
                	add.s32 	%r11, %r3, %r10;
                	mov.b64 	%fd38, {%r2, %r11};
                	sub.s32 	%r12, %r1, %r9;
                	shl.b32 	%r13, %r12, 20;
                	add.s32 	%r14, %r13, 1072693248;
                	mov.u32 	%r15, 0;
                	mov.b64 	%fd39, {%r15, %r14};
                	mul.f64 	%fd40, %fd38, %fd39;
                                
                $L__BB0_3:
                	st.param.f64 	[func_retval0+0], %fd40;
                	ret;
                                
                }""");
    }
}
