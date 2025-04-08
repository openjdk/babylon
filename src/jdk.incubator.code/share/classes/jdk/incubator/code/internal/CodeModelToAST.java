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
    private int c = 0; // used to name variables we introduce in the AST

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
                MethodRef.class, Block.Parameter.class, FieldRef.class, CoreOp.InvokeOp.InvokeKind.class,
                ExternalizableOp.class, RecordTypeRef.class
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
    }

    private JCTree invokeOpToJCMethodInvocation(CoreOp.InvokeOp invokeOp) {
        Method method;
        try {
            method = invokeOp.invokeDescriptor().resolveToDirectMethod(MethodHandles.lookup());
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
        long flags = method.getModifiers();
        var name = names.fromString(invokeOp.invokeDescriptor().name());
        Value receiver = invokeOp.invokeKind() == CoreOp.InvokeOp.InvokeKind.INSTANCE ? invokeOp.operands().get(0) : null;
        List<Value> arguments = invokeOp.operands().stream().skip(receiver == null ? 0 : 1).collect(List.collector());
        var paramTypes = invokeOp.invokeDescriptor().type().parameterTypes().stream().map(this::typeElementToType)
                .collect(List.collector());
        var restype = typeElementToType(invokeOp.resultType());
        var type = new Type.MethodType(paramTypes, restype, List.nil(), syms.methodClass);
        var methodSym = new Symbol.MethodSymbol(flags, name, type,
                typeElementToType(invokeOp.invokeDescriptor().refType()).tsym);
        var meth = receiver == null ? treeMaker.Ident(methodSym) : treeMaker.Select(exprTree(receiver), methodSym);
        var args = new ListBuffer<JCTree.JCExpression>();
        for (Value operand : arguments) {
            args.add(exprTree(operand));
        }
        var methodInvocation = treeMaker.App(meth, args.toList());
        if (invokeOp.isVarArgs()) {
            var lastParam = invokeOp.invokeDescriptor().type().parameterTypes().getLast();
            Assert.check(lastParam instanceof ArrayType);
            methodInvocation.varargsElement = typeElementToType(((ArrayType) lastParam).componentType());
        }
        if (invokeOp.result().uses().isEmpty()) {
            return treeMaker.Exec(methodInvocation);
        }
        return methodInvocation;
    }

    private JCTree opToTree(Op op) {
        JCTree tree = switch (op) {
            case CoreOp.ConstantOp constantOp when constantOp.value() == null ->
                    treeMaker.Literal(TypeTag.BOT, null).setType(syms.botType);
            case CoreOp.ConstantOp constantOp -> treeMaker.Literal(constantOp.value());
            case CoreOp.InvokeOp invokeOp -> invokeOpToJCMethodInvocation(invokeOp);
            case CoreOp.NewOp newOp -> {
                if (newOp.resultType() instanceof ArrayType at) {
                    var elemType = treeMaker.Ident(typeElementToType(at.componentType()).tsym);
                    var dims = new ListBuffer<JCTree.JCExpression>();
                    for (int d = 0; d < at.dimensions(); d++) {
                        dims.add(exprTree(newOp.operands().get(d)));
                    }
                    var na = treeMaker.NewArray(elemType, dims.toList(), null);
                    na.type = typeElementToType(at);
                    yield na;
                }
                var ownerType = typeElementToType(newOp.constructorType().returnType());
                var clazz = treeMaker.Ident(ownerType.tsym);
                var args = new ListBuffer<JCTree.JCExpression>();
                for (Value operand : newOp.operands()) {
                    args.add(exprTree(operand));
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
                    treeMaker.Return(exprTree(returnOp.returnValue()));
            case CoreOp.FieldAccessOp.FieldLoadOp fieldLoadOp -> {
                // Type.fieldName
                // if instance field we will use the same thechnique as in invokeOpToTree
                int flags;
                try {
                    flags = fieldLoadOp.fieldDescriptor().resolveToMember(MethodHandles.lookup()).getModifiers();
                } catch (ReflectiveOperationException e) {
                    throw new RuntimeException(e);
                }
                var name = names.fromString(fieldLoadOp.fieldDescriptor().name());
                var type = typeElementToType(fieldLoadOp.resultType());
                var owner = typeElementToType(fieldLoadOp.fieldDescriptor().refType());
                var sym = new Symbol.VarSymbol(flags, name, type, owner.tsym);
                yield treeMaker.Select(treeMaker.Ident(owner.tsym), sym);
            }
            case CoreOp.ArrayAccessOp.ArrayStoreOp arrayStoreOp -> {
                var array = arrayStoreOp.operands().get(0);
                var val = arrayStoreOp.operands().get(1);
                var index = arrayStoreOp.operands().get(2);
                var as = treeMaker.Assign(
                        treeMaker.Indexed(exprTree(array), exprTree(index)), exprTree(val)
                );
                as.type = typeElementToType(((ArrayType) array.type()).componentType());
                yield as;
                // body builder are created but never passed when creating the op, why ?
            }
            default -> throw new IllegalStateException("Op -> JCTree not supported for :" + op.getClass().getName());
        };
        if (tree instanceof JCTree.JCExpression expr) {
            // introduce a local variable to hold the expr, to make sure an op's tree is inserted right away
            // for some operations this is essential, e.g. to ensure the correct order of operations
            Type type;
            if (op instanceof CoreOp.ConstantOp cop && cop.value() == null) {
                // if ConstantOp value is null, tree.type will be null_type
                // if null_type is used to create a VarSymbol, an exception will be thrown
                type = typeElementToType(cop.resultType());
            } else {
                type = tree.type;
            }
            var vs = new Symbol.VarSymbol(LocalVarFlags, names.fromString("_$" + c++), type, ms);
            var varDef = treeMaker.VarDef(vs, expr);
            map(op.result(), varDef);
            return varDef;
        } else {
            map(op.result(), tree);
            return tree;
        }
    }

    private JCTree.JCExpression exprTree(Value v) {
        JCTree tree = valueToTree.get(v);
        if (tree instanceof JCTree.JCVariableDecl vd) {
            return treeMaker.Ident(vd);
        } else if (tree instanceof JCTree.JCExpressionStatement exprStat) {
            return exprStat.expr;
        } else if (tree instanceof JCTree.JCExpression expr) {
            return expr;
        }
        throw new IllegalStateException("Value not mapped to VariableDeclaration nor to an ExpressionStatement");
    }

    private void map(Value v, JCTree t) {
        valueToTree.put(v, t);
    }

    public JCTree.JCMethodDecl transformFuncOpToAST(CoreOp.FuncOp funcOp, Name methodName) {
        Assert.check(funcOp.body().blocks().size() == 1);

        var paramTypes = List.of(crSym.opFactoryType, crSym.typeElementFactoryType);
        var mt = new Type.MethodType(paramTypes, crSym.opType, List.nil(), syms.methodClass);
        ms = new Symbol.MethodSymbol(PUBLIC | STATIC | SYNTHETIC, methodName, mt, currClassSym);
        currClassSym.members().enter(ms);

        for (int i = 0; i < funcOp.parameters().size(); i++) {
            map(funcOp.parameters().get(i), treeMaker.Ident(ms.params().get(i)));
        }

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

    // TODO see if we can use LET AST node
    // TODO add vars in OpBuilder (later)


    // TODO explore builderOp --> java code (as string)
}
