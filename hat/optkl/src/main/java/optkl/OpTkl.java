/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
package optkl;

import jdk.incubator.code.Block;
import jdk.incubator.code.CodeContext;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.analysis.SSA;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;
import optkl.annotations.Kernel;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Stream;

public interface OpTkl{
   Predicate<JavaOp.FieldAccessOp> AnyFieldAccess = _->true;
   Predicate<JavaOp.InvokeOp> AnyInvoke = _->true;
static OpTkl impl(LookupCarrier lookupCarrier){
   record Impl(MethodHandles.Lookup lookup) implements LookupCarrier,OpTkl{}
   return new Impl(lookupCarrier.lookup());
}

static <F extends Op, T extends Op> T copyLocation(F from, T to ){
   to.setLocation(from.location());
   return to;
}


static String fieldName(JavaOp.FieldAccessOp fieldAccessOp) {
   return fieldAccessOp.fieldDescriptor().name();
}

static Object getStaticFinalPrimitiveValue(MethodHandles.Lookup lookup, JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
   if (fieldLoadOp.fieldDescriptor().refType() instanceof ClassType classType) {
      Class<?> clazz = (Class<?>) classTypeToTypeOrThrow(lookup, classType);
      try {
         Field field = clazz.getField(fieldName(fieldLoadOp));
         field.setAccessible(true);
         return field.get(null);
      } catch (NoSuchFieldException | IllegalAccessException e) {
         throw new RuntimeException(e);
      }
   }
   throw new RuntimeException("Could not find field value" + fieldLoadOp);
}


static Type classTypeToTypeOrThrow(MethodHandles.Lookup lookup, ClassType classType) {
   try {
      return classType.resolve(lookup);
   } catch (ReflectiveOperationException e) {
      throw new RuntimeException(e);
   }
}

static boolean isAssignable(MethodHandles.Lookup lookup, TypeElement typeElement, Class<?>... classes) {
   if (typeElement instanceof ClassType classType) {
      Type type = classTypeToTypeOrThrow(lookup, classType);
      return Arrays.stream(classes).anyMatch(clazz -> clazz.isAssignableFrom((Class<?>) type));
   }
   return false;

}

static boolean isAssignableTo(MethodHandles.Lookup lookup, JavaType javaType, Class<?>... classes) {
   if (javaType instanceof ClassType classType) {
      Type type = classTypeToTypeOrThrow(lookup, classType);
      Class<?> evalKlass = (Class<?>) type;
      return Arrays.stream(classes).anyMatch(evalKlass::isAssignableFrom);
   }
   return false;

}

static JavaOp.InvokeOp getTargetInvokeOp(JavaOp.LambdaOp lambdaOp) {
   return lambdaOp.body().entryBlock().ops().stream()
           .filter(op -> op instanceof JavaOp.InvokeOp)
           .map(op -> (JavaOp.InvokeOp) op)
           .filter(invokeOp -> invokeOp.invokeKind().equals(JavaOp.InvokeOp.InvokeKind.STATIC))
           .filter(invokeOp -> invokeOp.operands().get(0).type() instanceof ClassType classType
                   && classType.toClassName() instanceof String name
                   && (name.equals("hat.ComputeContext")||name.equals("hat.KernelContext")))
           .findFirst()
           .orElseThrow();
}

static Object[] getQuotedCapturedValues(JavaOp.LambdaOp lambdaOp, Quoted quoted, Method method) {
   var block = lambdaOp.body().entryBlock();
   var ops = block.ops();
   Object[] varLoadNames = ops.stream()
           .filter(op -> op instanceof CoreOp.VarAccessOp.VarLoadOp)
           .map(op -> (CoreOp.VarAccessOp.VarLoadOp) op)
           .map(varLoadOp -> (Op.Result) varLoadOp.operands().getFirst())
           .map(varLoadOp -> (CoreOp.VarOp) varLoadOp.op())
           .map(CoreOp.VarOp::varName).toArray();
   Map<String, Object> nameValueMap = new HashMap<>();

   quoted.capturedValues().forEach((k, v) -> {
      if (k instanceof Op.Result result) {
         if (result.op() instanceof CoreOp.VarOp varOp) {
            nameValueMap.put(varOp.varName(), v);
         }
      }
   });
   Object[] args = new Object[method.getParameterCount()];
   if (args.length != varLoadNames.length) {
      throw new IllegalStateException("Why don't we have enough captures.!! ");
   }
   for (int i = 1; i < args.length; i++) {
      args[i] = nameValueMap.get(varLoadNames[i].toString());
      if (args[i] instanceof CoreOp.Var varbox) {
         args[i] = varbox.value();
      }
   }
   return args;
}


static JavaType javaRefType(JavaOp.InvokeOp op) {
   return (JavaType) op.invokeDescriptor().refType();
}

static JavaType javaReturnType(JavaOp.InvokeOp invokeOp) {
   return (JavaType) invokeOp.invokeDescriptor().type().returnType();
}
static boolean javaReturnTypeIsVoid(JavaOp.InvokeOp invokeOp) {
   return javaReturnType(invokeOp) instanceof PrimitiveType primitiveType && primitiveType.isVoid();
}

static Method methodOrThrow(MethodHandles.Lookup lookup, JavaOp.InvokeOp op) {
   try {
      return op.invokeDescriptor().resolveToMethod(lookup);
   } catch (ReflectiveOperationException e) {
      throw new RuntimeException(e);
   }
}
   static boolean isMethod(JavaOp.InvokeOp invokeOp, Predicate<String> namePredicate) {
      return namePredicate.test(invokeOp.invokeDescriptor().name());
   }


static Class<?> javaRefClassOrThrow(MethodHandles.Lookup lookup, JavaOp.InvokeOp op) {
   if (javaRefType(op) instanceof ClassType classType) {
      return (Class<?>) classTypeToTypeOrThrow(lookup, classType);
   } else {
      throw new IllegalStateException(" javaRef class is null");
   }
}


static Op.Result lhsResult(JavaOp.BinaryOp binaryOp){
   return (Op.Result)binaryOp.operands().get(0);
}

static Op.Result rhsResult(JavaOp.BinaryOp binaryOp){
   return (Op.Result)binaryOp.operands().get(1);
}

static List<Op> ops(JavaOp.JavaConditionalOp javaConditionalOp, int idx){
   return javaConditionalOp.bodies().get(idx).entryBlock().ops();
}

static List<Op> lhsOps(JavaOp.JavaConditionalOp javaConditionalOp){
   return ops(javaConditionalOp,0);
}

static List<Op> rhsOps(JavaOp.JavaConditionalOp javaConditionalOp){
   return ops(javaConditionalOp,1);
}

static Op.Result result(JavaOp.BinaryTestOp binaryTestOp, int idx){
   return (Op.Result)binaryTestOp.operands().get(idx);
}

static Op.Result lhsResult(JavaOp.BinaryTestOp binaryTestOp){
   return result(binaryTestOp,0);
}

static Op.Result rhsResult(JavaOp.BinaryTestOp binaryTestOp){
   return result(binaryTestOp,1);
}

static Op.Result result(JavaOp.ConvOp convOp){
   return (Op.Result)convOp.operands().getFirst();
}

static Op.Result result(CoreOp.ReturnOp returnOp){
   return (Op.Result)returnOp.operands().getFirst();
}

static Block block(JavaOp.ConditionalExpressionOp ternaryOp, int idx){
   return ternaryOp.bodies().get(idx).entryBlock();
}

static Block condBlock(JavaOp.ConditionalExpressionOp ternaryOp){
   return block(ternaryOp,0);
}

static Block thenBlock(JavaOp.ConditionalExpressionOp ternaryOp){
   return block(ternaryOp,1);
}

static Block elseBlock(JavaOp.ConditionalExpressionOp ternaryOp){
   return block(ternaryOp,2);
}

static String funcName(JavaOp.InvokeOp invokeOp) {
   return invokeOp.invokeDescriptor().name();
}

static Value operandOrNull(Op op, int idx) {
   return op.operands().size() > idx?op.operands().get(idx):null;
}

static Op.Result resultOrNull(Op op, int idx) {
   return (operandOrNull(op,idx) instanceof Op.Result result)?result:null;
}

static Block block(JavaOp.ForOp forOp, int idx){
   return forOp.bodies().get(idx).entryBlock();
}

static Block mutateBlock(JavaOp.ForOp forOp){
   return block(forOp,2);
}

static Block loopBlock(JavaOp.ForOp forOp){
   return block(forOp,3);
}

static Block condBlock(JavaOp.ForOp forOp){
   return  forOp.cond().entryBlock();
}

static Block initBlock(JavaOp.ForOp forOp){
   return  forOp.init().entryBlock();
}

static Block block(JavaOp.WhileOp whileOp, int idx){
   return  whileOp.bodies().get(idx).entryBlock();
}

static Block condBlock(JavaOp.WhileOp whileOp){
   return  block(whileOp,0);
}

static Block loopBlock(JavaOp.WhileOp whileOp){
   return  block(whileOp,1);
}

static Block blockOrNull(JavaOp.IfOp ifOp, int idx ){
   return ifOp.bodies().size() > idx?ifOp.bodies().get(idx).entryBlock():null;
}

static JavaOp.FieldAccessOp fieldAccessOpNameMatches(jdk.incubator.code.CodeElement<?,?> codeElement, Predicate<String> namePredicate) {
   return codeElement instanceof JavaOp.FieldAccessOp fieldAccessOp
           && namePredicate.test(fieldName(fieldAccessOp))?fieldAccessOp:null;
}


static PrimitiveType asPrimitiveResultOrNull(Value v){
   if (v instanceof Op.Result r){
      if (r.op().resultType() instanceof PrimitiveType primitiveType){
         return primitiveType;
      }
   }
   return null;
}
static boolean isPrimitiveResult(Value v){
   return (asPrimitiveResultOrNull(v)!=null);
}

static Op.Result asResultOrThrow(Value value) {
   if (value instanceof Op.Result r) {
      return r;
   }else{
      throw new RuntimeException("Value not a result");
   }
}

static Stream<Op.Result> operandsAsResults(jdk.incubator.code.CodeElement<?,?> codeElement) {
   return codeElement instanceof Op ?
           ((Op)codeElement).operands().stream().filter(o-> o instanceof Op.Result).map(o->(Op.Result)o)
           :Stream.of();
}
static Op.Result operandAsResult(jdk.incubator.code.CodeElement<?,?> codeElement, int n) {
   return codeElement instanceof Op op  && op.operands().size()>n && op.operands().get(n) instanceof Op.Result result?result:null;
}
static Op opFromOperandAsResult(jdk.incubator.code.CodeElement<?,?> codeElement, int n) {
   return operandAsResult(codeElement,n) instanceof Op.Result result?result.op():null;
}

static Op.Result asResultOrNull(Value operand) {
   return operand instanceof Op.Result result?result:null;
}
static boolean isResult(Value operand) {
   return Objects.nonNull(asResultOrNull(operand));
}

static Op opOfResultOrNull(Op.Result result) {
   return result.op() instanceof Op op?op:null;
}

static TypeElement resultTypeOrNull(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
   return varLoadOp.resultType() instanceof TypeElement typeElement?typeElement:null;
}

static CoreOp.VarAccessOp.VarLoadOp asVarLoadOrNull(Op op) {
   return  op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp?varLoadOp:null;
}

static boolean resultType(MethodHandles.Lookup lookup, CoreOp.VarAccessOp.VarLoadOp varLoadOp, Class<?>... classes) {
   return isAssignable(lookup, varLoadOp.resultType(), classes);
}

   static Stream<Op> loopBodyStatements(Op.Loop op) {
       var list = new ArrayList<>(statements( op.loopBody().entryBlock()).toList());
       if (list.getLast() instanceof JavaOp.ContinueOp) {
           list.removeLast();
       }
       return list.stream();
   }

   static  Op asStatementOpOrNull(CodeElement<?,?> ce){
       if (ce instanceof Op op){
              return (
                      (
                              (op instanceof CoreOp.VarAccessOp.VarStoreOp && op.operands().get(1).uses().size() < 2)
                               || (op instanceof CoreOp.VarOp || op.result().uses().isEmpty())
                               || (op instanceof StatementLikeOp)
                      )
                      && !(op instanceof CoreOp.VarOp varOp && isParamVar(varOp))//..ParamVar.of(varOp) != null)
                      && !(op instanceof CoreOp.YieldOp)
              )
                      ?op
                      :null;
       }else{
          return null;
       }

   }
   static  boolean isStatementOp(CodeElement<?,?> ce){
       return Objects.nonNull(asStatementOpOrNull(ce));
   }

   static Stream<Op> statements(Block block) {
       return block.ops().stream().filter(OpTkl::isStatementOp);
   }

   static CoreOp.FuncOp lower(CallSite callSite, CoreOp.FuncOp funcOp) {
   if (callSite.tracing()){
      System.out.println(callSite);
   }
   return funcOp.transform(CodeTransformer.LOWERING_TRANSFORMER);
}
static Stream<jdk.incubator.code.CodeElement<?,?>> elements(CallSite callSite, CoreOp.FuncOp funcOp) {
   if (callSite.tracing()){
      System.out.println(callSite);
   }
   return funcOp.elements();
}
static <T extends Op> Stream<T> ops(CallSite callSite, CoreOp.FuncOp funcOp,
                                    Predicate<jdk.incubator.code.CodeElement<?,?>> predicate,
                                    Function<CodeElement<?,?>,T> mapper
) {
   if (callSite.tracing()){
      System.out.println(callSite);
   }
   return funcOp.elements().filter(predicate).map(mapper);
}
static <T> Stream<T> opstream(CoreOp.FuncOp funcOp, Function<CodeElement<?,?>,T> mapper) {
   return funcOp.elements().map(mapper).filter(Objects::nonNull);
}


static CoreOp.FuncOp SSATransformLower(CallSite callSite, CoreOp.FuncOp funcOp){
   if (callSite.tracing()){
      System.out.println(callSite);
   }
   return  SSA.transform(lower(callSite,funcOp));
}
static CoreOp.FuncOp SSATransform(CallSite callSite, CoreOp.FuncOp funcOp){
   if (callSite.tracing()){
      System.out.println(callSite);
   }
   return  SSA.transform(funcOp);
}

static CoreOp.FuncOp transform(CallSite callSite, CoreOp.FuncOp funcOp, Predicate<Op> predicate, CodeTransformer CodeTransformer) {
   if (callSite.tracing()){
      System.out.println(callSite);
   }
   return funcOp.transform((blockBuilder, op) -> {
      if (predicate.test(op)){
         var builder = CodeTransformer.acceptOp(blockBuilder,op);
         if (builder != blockBuilder){
            throw new RuntimeException("Where does this builder come from "+builder);
         }
      }else {
         blockBuilder.op(op);
      }
      return blockBuilder;
   });
}

static CoreOp.FuncOp transform(CallSite callSite, CoreOp.FuncOp funcOp, CodeTransformer CodeTransformer) {
   if (callSite.tracing()){
      System.out.println(callSite);
   }
   return funcOp.transform(CodeTransformer);
}

record  OpMap(CoreOp.FuncOp fromFuncOp, CoreOp.FuncOp toFuncOp,  Map<Op,Op> fromToOpMap){}

static <InOp extends Op, OutOp extends Op> OutOp replaceOp(Block.Builder blockBuilder, InOp inOp,java.util.function.Function<List<Value>, OutOp> factory) {
   List<Value> inputOperands = inOp.operands();
   CodeContext context = blockBuilder.context();
   List<Value> outputOperands = context.getValues(inputOperands);
   OutOp outOp = factory.apply(outputOperands);
   Op.Result outputResult = blockBuilder.op(outOp);
   Op.Result inputResult = inOp.result();
   outOp.setLocation(inOp.location());
   context.mapValue(inputResult, outputResult);
   return outOp;
}
static < OutOp extends Op> OpMap simpleOpMappingTransform(CallSite here, CoreOp.FuncOp fromFuncOp, Predicate<Op> opPredicate,
                                                          java.util.function.Function<List<Value>, OutOp> opFactory){
   Map<Op,Op> fromToOpMap = new LinkedHashMap<>();
   CoreOp.FuncOp toFuncOp =  transform(here, fromFuncOp, (blockBuilder, inOp) -> {
      if (opPredicate.test(inOp)) {
         fromToOpMap.put(inOp, replaceOp(blockBuilder, inOp, opFactory));
      }else {
         var r = blockBuilder.op(inOp);
         fromToOpMap.put(inOp,r.op());
      }
      return blockBuilder;
   });
   return new OpMap(fromFuncOp, toFuncOp, fromToOpMap);
}




static boolean returnIsVoid(JavaOp.InvokeOp invokeOp){
   return javaReturnType(invokeOp) instanceof PrimitiveType primitiveType && primitiveType.isVoid();
}

// IMPORTANT:
// When we have patterns like:
//
// myiFaceArray.array().value(storeAValue);
//
// We need to generate extra parenthesis to make the struct pointer accessor "->" correct.
// This is a common pattern when we have a IFace type that contains a subtype based on
// struct or union.
//
// An example of this is for the type F16Array.
static boolean needExtraParenthesis(JavaOp.InvokeOp invokeOp) {

   // The following expression checks that the current invokeOp has at least 2 operands:
   // Why 2?
   // - The first one is another invokeOp to load the inner struct from an IFace data structure.
   //   The first operand is also assignable.
   // - The second one is the store value, but this depends on the semantics and definition
   //   of the user code.
   return invokeOp.operands().size() >= 2 && invokeOp.operands().get(0) instanceof Op.Result r1
           && r1.op() instanceof JavaOp.InvokeOp invokeOp2
           && javaReturnType(invokeOp2) instanceof ClassType;
}



static  Class<?> typeElementToClass(MethodHandles.Lookup lookup,TypeElement type) {
   class PrimitiveHolder {
      static final Map<PrimitiveType, Class<?>> primitiveToClass = Map.of(
              JavaType.BYTE, byte.class,
              JavaType.SHORT, short.class,
              JavaType.INT, int.class,
              JavaType.LONG, long.class,
              JavaType.FLOAT, float.class,
              JavaType.DOUBLE, double.class,
              JavaType.CHAR, char.class,
              JavaType.BOOLEAN, boolean.class
      );
   }
   try {
      if (type instanceof PrimitiveType primitiveType) {
         return PrimitiveHolder.primitiveToClass.get(primitiveType);
      } else if (type instanceof ClassType classType) {
         return ((Class<?>) classType.resolve(lookup));
      } else {
         throw new IllegalArgumentException("given type cannot be converted to class");
      }
   } catch (ReflectiveOperationException e) {
      throw new RuntimeException("given type cannot be converted to class");
   }
}
   static boolean isParamVar(CoreOp.VarOp varOp) {
      return !varOp.isUninitialized()
              && varOp.operands().getFirst() instanceof Block.Parameter parameter
              && parameter.invokableOperation() instanceof CoreOp.FuncOp funcOp;
   }
}


