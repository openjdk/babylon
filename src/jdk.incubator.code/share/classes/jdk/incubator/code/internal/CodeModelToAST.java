package jdk.incubator.code.internal;

import com.sun.tools.javac.code.*;
import com.sun.tools.javac.code.Symbol.MethodSymbol;
import com.sun.tools.javac.code.Symbol.VarSymbol;
import com.sun.tools.javac.comp.AttrContext;
import com.sun.tools.javac.comp.Env;
import com.sun.tools.javac.comp.Resolve;
import com.sun.tools.javac.tree.JCTree;
import com.sun.tools.javac.tree.JCTree.JCExpression;
import com.sun.tools.javac.tree.TreeInfo;
import com.sun.tools.javac.tree.TreeMaker;
import com.sun.tools.javac.util.*;
import com.sun.tools.javac.util.List;
import jdk.incubator.code.*;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.type.*;

import java.util.*;

import static com.sun.tools.javac.code.Flags.*;

public class CodeModelToAST {

    private final TreeMaker treeMaker;
    private final Names names;
    private final Symtab syms;
    private final Env<AttrContext> attrEnv;
    private final Resolve resolve;
    private final Types types;
    private final Symbol.ClassSymbol currClassSym;
    private final CodeReflectionSymbols crSym;
    private Symbol.MethodSymbol ms;
    private int localVarCount = 0; // used to name variables we introduce in the AST
    private final Map<Value, JCTree> valueToTree = new HashMap<>();
    private static final MethodRef M_BLOCK_BUILDER_OP = MethodRef.method(Block.Builder.class, "op",
            Op.Result.class, Op.class);
    private static final MethodRef M_BLOCK_BUILDER_PARAM = MethodRef.method(Block.Builder.class, "parameter",
            Block.Parameter.class, TypeElement.class);

    public CodeModelToAST(TreeMaker treeMaker, Names names, Symtab syms, Resolve resolve,
                          Types types, Env<AttrContext> attrEnv, CodeReflectionSymbols crSym) {
        this.treeMaker = treeMaker;
        this.names = names;
        this.syms = syms;
        this.resolve = resolve;
        this.types = types;
        this.attrEnv = attrEnv;
        this.currClassSym = attrEnv.enclClass.sym;
        this.crSym = crSym;
    }

    private Type typeElementToType(TypeElement jt) {
        return switch (jt) {
            case PrimitiveType pt when pt == JavaType.BOOLEAN -> syms.booleanType;
            case PrimitiveType pt when pt == JavaType.BYTE -> syms.byteType;
            case PrimitiveType pt when pt == JavaType.CHAR -> syms.charType;
            case PrimitiveType pt when pt == JavaType.INT -> syms.intType;
            case PrimitiveType pt when pt == JavaType.LONG -> syms.longType;
            case PrimitiveType pt when pt == JavaType.FLOAT -> syms.floatType;
            case PrimitiveType pt when pt == JavaType.DOUBLE -> syms.doubleType;
            case ClassType ct when ct.hasTypeArguments() -> {
                Type enclosing = ct.enclosingType().map(this::typeElementToType).orElse(Type.noType);
                List<Type> typeArgs = List.from(ct.typeArguments()).map(this::typeElementToType);
                yield new Type.ClassType(enclosing, typeArgs, typeElementToType(ct.rawType()).tsym);
            }
            case ClassType ct -> types.erasure(syms.enterClass(attrEnv.toplevel.modle, ct.toClassName()));
            case ArrayType at -> new Type.ArrayType(typeElementToType(at.componentType()), syms.arrayClass);
            default -> throw new IllegalStateException("Unsupported type: " + jt);
        };
    }

    private JCExpression toExpr(JCTree t) {
        return switch (t) {
            case JCExpression e -> e;
            case JCTree.JCVariableDecl vd -> treeMaker.Ident(vd);
            case null, default -> throw new IllegalArgumentException();
        };
    }

    private JCTree invokeOpToJCMethodInvocation(CoreOp.InvokeOp invokeOp) {
        Value receiver = (invokeOp.invokeKind() == CoreOp.InvokeOp.InvokeKind.INSTANCE) ?
                invokeOp.operands().get(0) : null;
        List<Value> arguments = invokeOp.operands().stream()
                .skip(receiver == null ? 0 : 1)
                .collect(List.collector());
        var methodSym = methodDescriptorToSymbol(invokeOp.invokeDescriptor());
        var meth = (receiver == null) ?
                treeMaker.Ident(methodSym) :
                treeMaker.Select(toExpr(opToTree(receiver)), methodSym);
        var args = new ListBuffer<JCTree.JCExpression>();
        for (Value operand : arguments) {
            args.add(toExpr(opToTree(operand)));
        }
        var methodInvocation = treeMaker.App(meth, args.toList());
        if (invokeOp.isVarArgs()) {
            setVarargs(methodInvocation, invokeOp.invokeDescriptor().type());
        }
        return methodInvocation;
    }

    void setVarargs(JCExpression tree, FunctionType type) {
        var lastParam = type.parameterTypes().getLast();
        if (lastParam instanceof ArrayType varargType) {
            TreeInfo.setVarargsElement(tree, typeElementToType(varargType.componentType()));
        } else {
            Assert.error("Expected trailing array type: " + type);
        }
    }

    public JCTree.JCMethodDecl transformFuncOpToAST(CoreOp.FuncOp funcOp, Name methodName) {
        Assert.check(funcOp.body().blocks().size() == 1);

        var paramTypes = List.of(crSym.opFactoryType, crSym.typeElementFactoryType);
        var mt = new Type.MethodType(paramTypes, crSym.opType, List.nil(), syms.methodClass);
        ms = new Symbol.MethodSymbol(PUBLIC | STATIC | SYNTHETIC, methodName, mt, currClassSym);
        currClassSym.members().enter(ms);

        for (int i = 0; i < funcOp.parameters().size(); i++) {
            valueToTree.put(funcOp.parameters().get(i), treeMaker.Ident(ms.params().get(i)));
        }

        java.util.List<Value> rootValues = funcOp.traverse(new ArrayList<>(), (l, ce) -> {
            if (ce instanceof Op op && op.result() != null && op.result().uses().size() != 1) {
                l.add(op.result());
            } else if (ce instanceof CoreOp.InvokeOp invokeOp && (invokeOp.invokeDescriptor().equals(M_BLOCK_BUILDER_OP)
                   || invokeOp.invokeDescriptor().equals(M_BLOCK_BUILDER_PARAM))) {
               l.add(invokeOp.result());
            }
            return l;
        });

        var stats = new ListBuffer<JCTree.JCStatement>();
        for (Value root : rootValues) {
            JCTree tree = opToTree(root);
            if (tree instanceof JCExpression e) {
                var vs = new Symbol.VarSymbol(LocalVarFlags, names.fromString("_$" + localVarCount++), tree.type, ms);
                tree = treeMaker.VarDef(vs, e);
                valueToTree.put(root, tree);
            }
            stats.add((JCTree.JCStatement) tree);
        }
        var mb = treeMaker.Block(0, stats.toList());

        return treeMaker.MethodDef(ms, mb);
    }

    private JCTree opToTree(Value v) {
        if (valueToTree.containsKey(v)) {
            return valueToTree.get(v);
        }
        Op op = ((Op.Result) v).op();
        JCTree tree = switch (op) {
            case CoreOp.ConstantOp constantOp when constantOp.value() == null ->
                    treeMaker.Literal(TypeTag.BOT, null).setType(syms.botType);
            case CoreOp.ConstantOp constantOp -> treeMaker.Literal(constantOp.value());
            case CoreOp.InvokeOp invokeOp -> invokeOpToJCMethodInvocation(invokeOp);
            case CoreOp.NewOp newOp when newOp.resultType() instanceof ArrayType at -> {
                var elemType = treeMaker.Ident(typeElementToType(at.componentType()).tsym);
                var dims = new ListBuffer<JCTree.JCExpression>();
                for (int d = 0; d < at.dimensions(); d++) {
                    dims.add(toExpr(opToTree(newOp.operands().get(d))));
                }
                var na = treeMaker.NewArray(elemType, dims.toList(), null);
                na.type = typeElementToType(at);
                yield na;
            }
            case CoreOp.NewOp newOp -> {
                var ownerType = typeElementToType(newOp.constructorDescriptor().refType());
                var clazz = treeMaker.Ident(ownerType.tsym);
                var args = new ListBuffer<JCTree.JCExpression>();
                for (Value operand : newOp.operands()) {
                    args.add(toExpr(opToTree(operand)));
                }
                var nc = treeMaker.NewClass(null, null, clazz, args.toList(), null);
                if (newOp.isVarargs()) {
                    setVarargs(nc, newOp.constructorDescriptor().type());
                }
                nc.type = ownerType;
                nc.constructor = constructorDescriptorToSymbol(newOp.constructorDescriptor());
                nc.constructorType = nc.constructor.type;
                yield nc;
            }
            case CoreOp.ReturnOp returnOp ->
                    treeMaker.Return(toExpr(opToTree(returnOp.returnValue())));
            case CoreOp.FieldAccessOp.FieldLoadOp fieldLoadOp -> {
                var sym = fieldDescriptorToSymbol(fieldLoadOp.fieldDescriptor());
                Assert.check(sym.isStatic());
                yield treeMaker.Select(treeMaker.Ident(sym.owner), sym);
            }
            case CoreOp.ArrayAccessOp.ArrayStoreOp arrayStoreOp -> {
                var array = arrayStoreOp.operands().get(0);
                var val = arrayStoreOp.operands().get(1);
                var index = arrayStoreOp.operands().get(2);
                var as = treeMaker.Assign(
                        treeMaker.Indexed(
                                toExpr(opToTree(array)), toExpr(opToTree(index))), toExpr(opToTree(val))
                );
                as.type = typeElementToType(((ArrayType) array.type()).componentType());
                yield as;
            }
            default -> throw new IllegalStateException("Op -> JCTree not supported for :" + op.getClass().getName());
        };
        valueToTree.put(v, tree);
        return tree;
    }

    VarSymbol fieldDescriptorToSymbol(FieldRef fieldRef) {
        Name name = names.fromString(fieldRef.name());
        Type site = typeElementToType(fieldRef.refType());
        return resolve.resolveInternalField(attrEnv.enclClass, attrEnv, site, name);
    }

    MethodSymbol methodDescriptorToSymbol(MethodRef methodRef) {
        Name name = names.fromString(methodRef.name());
        Type site = typeElementToType(methodRef.refType());
        List<Type> argtypes = methodRef.type().parameterTypes().stream()
                .map(this::typeElementToType).collect(List.collector());
        return resolve.resolveInternalMethod(attrEnv.enclClass, attrEnv, site, name, argtypes, List.nil());
    }

    MethodSymbol constructorDescriptorToSymbol(ConstructorRef constructorRef) {
        Type site = typeElementToType(constructorRef.refType());
        List<Type> argtypes = constructorRef.type().parameterTypes().stream()
                .map(this::typeElementToType).collect(List.collector());
        return resolve.resolveInternalConstructor(attrEnv.enclClass, attrEnv, site, argtypes, List.nil());
    }

    // TODO: generate AST in SSA form
    // TODO: drop addVarsWhenNecessary
    // TODO: maybe move back into ReflectMethods
}
