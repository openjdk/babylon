package jdk.incubator.code.internal;

import com.sun.tools.javac.code.*;
import com.sun.tools.javac.tree.JCTree;
import com.sun.tools.javac.tree.TreeMaker;
import com.sun.tools.javac.util.*;
import jdk.incubator.code.*;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.op.ExternalizableOp;
import jdk.incubator.code.op.OpFactory;
import jdk.incubator.code.type.*;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static com.sun.tools.javac.code.Flags.*;

public class CodeModelToAST {

    private final TreeMaker treeMaker;
    private final Names names;
    private final Symtab syms;
    private final Symbol.ClassSymbol currClassSym;
    private final CodeReflectionSymbols crSym;
    private final Map<Value, JCTree> valueToTree = new HashMap<>();
    private final Map<JavaType, Type> jtToType;
    private Symbol.MethodSymbol ms;

    public CodeModelToAST(TreeMaker treeMaker, Names names, Symtab syms,
                          Symbol.ClassSymbol currClassSym, CodeReflectionSymbols crSym) {
        this.treeMaker = treeMaker;
        this.names = names;
        this.syms = syms;
        this.currClassSym = currClassSym;
        this.crSym = crSym;
        this.jtToType = mappingFromJavaTypeToType();
    }

    private Map<JavaType, Type> mappingFromJavaTypeToType() {
        Map<JavaType, Type> m = new HashMap<>();
        Symbol.ModuleSymbol jdk_incubator_code = syms.enterModule(names.jdk_incubator_code);
        Class<?>[] crTypes = {Body.Builder.class, TypeElement.ExternalizedTypeElement.class, TypeElement.class,
                FunctionType.class, Block.Builder.class, Value.class, Block.Reference.class, Op.Result.class,
                Op.class, TypeElementFactory.class, OpFactory.class, ExternalizableOp.ExternalizedOp.class,
                MethodRef.class, Block.Parameter.class
        };
        for (Class<?> crType : crTypes) {
            JavaType jt = JavaType.type(crType.describeConstable().get());
            Type t = syms.enterClass(jdk_incubator_code, jt.externalize().toString());
            m.put(jt, t);
        }
        Class<?>[] javaBaseTypes = {HashMap.class, String.class, Object.class, Map.class, java.util.List.class};
        for (Class<?> javaBaseType : javaBaseTypes) {
            JavaType jt = JavaType.type(javaBaseType.describeConstable().get());
            Type t = syms.enterClass(syms.java_base, jt.externalize().toString());
            m.put(jt, t);
        }

        m.putAll(Map.ofEntries(
                Map.entry(JavaType.BOOLEAN, syms.booleanType),
                Map.entry(JavaType.BYTE, syms.byteType),
                Map.entry(JavaType.SHORT, syms.shortType),
                Map.entry(JavaType.CHAR, syms.charType),
                Map.entry(JavaType.INT, syms.intType),
                Map.entry(JavaType.LONG, syms.longType),
                Map.entry(JavaType.FLOAT, syms.floatType),
                Map.entry(JavaType.DOUBLE, syms.doubleType)
        ));

        return m;
    }

    private Type typeElementToType(TypeElement te) {
//        if (te != null) {
//            System.out.println(te.externalize().toString());
//        }
//        Assert.check(te instanceof JavaType, te.getClass().getName() + "not a java type");
        JavaType jt = (JavaType) te;
        return switch (jt) {
            case ClassType ct when ct.hasTypeArguments() -> {
                ClassType enclosingType = ct.enclosingType().orElse(null);
                ListBuffer<Type> typeArgs = new ListBuffer<>();
                for (JavaType typeArgument : ct.typeArguments()) {
                    typeArgs.add(typeElementToType(typeArgument));
                }
                yield new Type.ClassType(typeElementToType(enclosingType), typeArgs.toList(),
                        typeElementToType(ct.rawType()).tsym);
            }
            case ArrayType at -> {
                yield new Type.ArrayType(typeElementToType(at.componentType()), syms.arrayClass);
            }
            case null -> Type.noType;
            default -> {
                if (!jtToType.containsKey(te)) {
                    throw new IllegalStateException("JavaType -> Type not found for: " + te.externalize().toString());
                }
                yield jtToType.get(te);
            }
        };
        // what about List<Value>
        // how a type like List<Value> is represented in AST ? debug to know
        // it's represeneted as TypeApply
    }
    // we have two modules, java.base and jdk.incubator.code
    // we can use the same method as in CodeReflectionSymbols
//    private Type typeElementToType(TypeElement te) {
//        // @@@ TODO TypeElement -> Type
//        // te is JavaType
//        // look at reverse
//        if (te instanceof ArrayType arrayType) {
//            return new Type.ArrayType(typeElementToType(((ArrayType) te).componentType()), syms.arraysType.tsym);
//        }
//        String s = te.externalize().toString();
//        Symbol.ModuleSymbol moduleSymbol;
//        if (s.startsWith(names.jdk_incubator_code.toString())) {
//            moduleSymbol = syms.enterModule(names.jdk_incubator_code);
//        } else { // java.base module
//            moduleSymbol = syms.enterModule(names.java_base);
//        }
//        return syms.enterClass(moduleSymbol, s);
//    }

    private JCTree.JCMethodInvocation invokeOpToJCMethodInvocation(CoreOp.InvokeOp invokeOp) {
        Method method;
        try {
            method = invokeOp.invokeDescriptor().resolveToDirectMethod(MethodHandles.lookup());
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
        var flags = method.getModifiers();
        var name = names.fromString(invokeOp.invokeDescriptor().name());
        var argTypes = new ListBuffer<Type>();
        for (Value operand : invokeOp.operands()) {
            argTypes.add(typeElementToType(operand.type()));
        }
        var restype = typeElementToType(invokeOp.resultType());
        var type = new Type.MethodType(argTypes.toList(), restype, List.nil(), syms.methodClass);
        var methodSym = new Symbol.MethodSymbol(flags, name, type,
                typeElementToType(invokeOp.invokeDescriptor().refType()).tsym);
        var meth = treeMaker.Ident(methodSym);
        var args = new ListBuffer<JCTree.JCExpression>();
        for (Value operand : invokeOp.operands()) {
            args.add((JCTree.JCExpression) valueToTree.get(operand));
        }
        return treeMaker.App(meth, args.toList());
    }

//    private JCTree.JCExpression valueToTree(Value v) {
//        if (valueToTree.containsKey(v)) {
//            return valueToTree.get(v);
//        }
//        if (v instanceof Op.Result opr) {
//            Op op = opr.op();
//            JCTree.JCExpression t = switch (op) {
////                case CoreOp.ConstantOp constantOp when constantOp.value() != null -> treeMaker.Literal(constantOp.value());
//                case CoreOp.ConstantOp constantOp -> treeMaker.Literal(typeElementToType(constantOp.resultType()).getTag(),
//                        constantOp.value());
//                case CoreOp.InvokeOp invokeOp -> invokeOpToJCMethodInvocation(invokeOp);
//                case CoreOp.NewOp newOp -> {
//                    var constructorType = typeElementToType(newOp.constructorType().returnType());
//                    var clazz = treeMaker.Ident(constructorType.tsym);
//                    var typeArgs = new ListBuffer<JCTree.JCExpression>();
//                    if (newOp.resultType() instanceof ClassType ct) {
//                        for (JavaType typeArgument : ct.typeArguments()) {
//                            typeArgs.add(treeMaker.Ident(typeElementToType(typeArgument).tsym));
//                        }
//                    }
//                    var args = new ListBuffer<JCTree.JCExpression>();
//                    for (Value operand : newOp.operands()) {
//                        args.add(valueToTree(operand));
//                    }
//                    // @@@ JCNewClass I create has constructorType and constructor null, why ?
//                    // ask Maurizio
//                    yield treeMaker.NewClass(null, typeArgs.toList(), clazz, args.toList(),null);
//                }
//                default -> throw new IllegalStateException("Op -> JCTree not supported for :" + op.getClass().getName());
//            };
//            valueToTree.put(v, t);
//            return t;
//        } else if (v instanceof Block.Parameter p) {
//            Assert.check(valueToTree.containsKey(v));
//            return valueToTree.get(p);
//        } else {
//            throw new IllegalStateException();
//        }
//    }

    // opr : op operands
    // operands are results of previous operations
    // because block params are first wrapped in vars and they are not used directely
    // before their uses, we first do var.load then use the result of that
    private JCTree opToTree(Op op) {
        JCTree tree = switch (op) {
            case CoreOp.ConstantOp constantOp when constantOp.resultType() instanceof PrimitiveType ->
                    treeMaker.Literal(constantOp.value());
            case CoreOp.ConstantOp constantOp -> {
                var literalType = typeElementToType(constantOp.resultType());
               yield treeMaker.Literal(literalType.getTag(), constantOp.value()).setType(literalType);
            }
            case CoreOp.InvokeOp invokeOp -> invokeOpToJCMethodInvocation(invokeOp);
            case CoreOp.NewOp newOp -> {
                var ownerType = typeElementToType(newOp.constructorType().returnType());
                var clazz = treeMaker.Ident(ownerType.tsym);
                var args = new ListBuffer<JCTree.JCExpression>();
                for (Value operand : newOp.operands()) {
                    args.add((JCTree.JCExpression) valueToTree.get(operand));
                }
                var nc = treeMaker.NewClass(null, null, clazz, args.toList(), null);
                var argTypes = new ListBuffer<Type>();
                for (Value operand : newOp.operands()) {
                    argTypes.add(typeElementToType(operand.type()));
                }
                nc.type = ownerType;
                nc.constructorType = new Type.MethodType(argTypes.toList(), syms.voidType, List.nil(), syms.methodClass);
                nc.constructor = new Symbol.MethodSymbol(PUBLIC, names.init, nc.constructorType, ownerType.tsym);
                yield nc;
            }
            case CoreOp.ReturnOp returnOp ->
                    treeMaker.Return((JCTree.JCExpression) valueToTree.get(returnOp.returnValue()));
            case CoreOp.VarOp varOp -> {
                var name = names.fromString(varOp.varName());
                var type = typeElementToType(varOp.varValueType());
                var init = valueToTree.get(varOp.initOperand());
                var v = new Symbol.VarSymbol(0, name, type, ms);
                yield treeMaker.VarDef(v, (JCTree.JCExpression) init);
            }
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> {
                yield treeMaker.Ident((JCTree.JCVariableDecl) valueToTree.get(varLoadOp.varOperand()));
            }
            default -> throw new IllegalStateException("Op -> JCTree not supported for :" + op.getClass().getName());
        };
        valueToTree.put(op.result(), tree);
        return tree;
    }

    public JCTree.JCMethodDecl transformFuncOpToAST(CoreOp.FuncOp funcOp, Name methodName) {
        Assert.check(funcOp.body().blocks().size() == 1);

        var paramTypes = List.of(crSym.opFactoryType, crSym.typeElementFactoryType);
        var mt = new Type.MethodType(paramTypes, crSym.opType, List.nil(), syms.methodClass);
        ms = new Symbol.MethodSymbol(PUBLIC | STATIC | SYNTHETIC, methodName, mt, currClassSym);
        currClassSym.members().enter(ms);

        // TODO add VarOps in OpBuilder
        funcOp = addVarsWhenNecessary(funcOp);

        var stats = new ListBuffer<JCTree.JCStatement>();
        for (Op op : funcOp.body().entryBlock().ops()) {
            var tree = opToTree(op);
            if (tree instanceof JCTree.JCStatement stat) {
                stats.add(stat);
            }
        }
        var mb = treeMaker.Block(0, stats.toList());

        return treeMaker.MethodDef(ms, mb);
    }

    // we are doing it wrong
    // e.g. m = new Map; m.put...; return m;
    // m.put... will be ignored

    // if an opr is used more than once, it should be in a var
    // (no need for the below case)
    // for cases like: m = new Map...; m.put...; foo(m); m is used more than once
    // else if an opr comes form NewOp and used with InvokeOp as receiver, it should be in a var
    // to deal with cases like: m = new Map; m.put...;
    // note that for cases like: m = new Map; foo(m); is equivalent to foo(new Map) and the var isn't necessary

    public static CoreOp.FuncOp addVarsWhenNecessary(CoreOp.FuncOp funcOp) {
        // using cc only is not possible
        // because at first opr --> varOpRes
        // at the first usage we would have to opr --> varLoad
        // meaning we would have to back up the mapping, update it, then restore it before transforming the next op

        Map<Value, CoreOp.VarOp> valueToVar = new HashMap<>();
        AtomicInteger varCounter = new AtomicInteger();

        return CoreOp.func(funcOp.funcName(), funcOp.body().bodyType()).body(block -> {
            var newParams = block.parameters();
            var oldParams = funcOp.parameters();
            for (int i = 0; i < newParams.size(); i++) {
                Op.Result var = block.op(CoreOp.var("_$" + varCounter.getAndIncrement(), newParams.get(i)));
                valueToVar.put(oldParams.get(i), ((CoreOp.VarOp) var.op()));
            }

            block.transformBody(funcOp.body(), java.util.List.of(), (Block.Builder b, Op op) -> {
                var cc = b.context();
                for (Value operand : op.operands()) {
                    if (valueToVar.containsKey(operand)) {
                        var varLoadRes = b.op(CoreOp.varLoad(valueToVar.get(operand).result()));
                        cc.mapValue(operand, varLoadRes);
                    }
                }
                var opr = b.op(op);
                if (!(op instanceof CoreOp.VarOp) && op.result().uses().size() > 1) {
                    var varOpRes = b.op(CoreOp.var("_$" + varCounter.getAndIncrement(), opr));
                    valueToVar.put(op.result(), ((CoreOp.VarOp) varOpRes.op()));
                }
                return b;
            });
        });
    }

    // TODO see if we can use LET AST node
    // TODO add vars in OpBuilder (later)




    // TODO explore builderOp --> java code (as string)
}
