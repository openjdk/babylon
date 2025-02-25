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
        var meth = receiver == null ? treeMaker.Ident(methodSym) : treeMaker.Select((JCTree.JCExpression) valueToTree.get(receiver), methodSym);
        var args = new ListBuffer<JCTree.JCExpression>();
        for (Value operand : arguments) {
            args.add((JCTree.JCExpression) valueToTree.get(operand));
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
            case CoreOp.VarOp varOp when varOp.initOperand() instanceof Block.Parameter p -> valueToTree.get(p);
            case CoreOp.VarOp varOp -> {
                var name = names.fromString(varOp.varName());
                var type = typeElementToType(varOp.varValueType());
                var v = new Symbol.VarSymbol(LocalVarFlags, name, type, ms);
                yield treeMaker.VarDef(v, (JCTree.JCExpression) valueToTree.get(varOp.initOperand()));
            }
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp
                    when varLoadOp.varOp().initOperand() instanceof Block.Parameter p2 -> valueToTree.get(p2);
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp ->
                    treeMaker.Ident((JCTree.JCVariableDecl) valueToTree.get(varLoadOp.varOperand()));
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
        for (int i = 0; i < funcOp.parameters().size(); i++) {
            valueToTree.put(funcOp.parameters().get(i), treeMaker.Ident(ms.params().get(i)));
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
                var M_BLOCK_BUILDER_OP = MethodRef.method(Block.Builder.class, "op", Op.Result.class, Op.class);
                // we introduce VarOp to hold an opr that's used multiple times
                // or to mark that an InvokeOp must be mapped to a Statement
                // specifically call to Bloc.Builder#op, we want this call to map to a statement so that it get added
                // to the opMethod body immediately to ensure correct order of operations
                var isBlockOpInvocation = op instanceof CoreOp.InvokeOp invokeOp && M_BLOCK_BUILDER_OP.equals(invokeOp.invokeDescriptor());
                if (!(op instanceof CoreOp.VarOp) && (op.result().uses().size() > 1 || isBlockOpInvocation)) {
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
