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

package com.sun.tools.javac.comp;

import com.sun.source.tree.LambdaExpressionTree;
import com.sun.source.tree.MemberReferenceTree.ReferenceMode;
import com.sun.tools.javac.code.Flags;
import com.sun.tools.javac.code.Kinds.Kind;
import com.sun.tools.javac.code.Symbol;
import com.sun.tools.javac.code.Symbol.MethodSymbol;
import com.sun.tools.javac.code.Symbol.VarSymbol;
import com.sun.tools.javac.code.Symtab;
import com.sun.tools.javac.code.Type;
import com.sun.tools.javac.code.Type.ArrayType;
import com.sun.tools.javac.code.Type.ClassType;
import com.sun.tools.javac.code.Type.MethodType;
import com.sun.tools.javac.code.Type.TypeVar;
import com.sun.tools.javac.code.Type.WildcardType;
import com.sun.tools.javac.code.TypeTag;
import com.sun.tools.javac.code.Types;
import com.sun.tools.javac.comp.DeferredAttr.FilterScanner;
import com.sun.tools.javac.jvm.Gen;
import com.sun.tools.javac.resources.CompilerProperties.Notes;
import com.sun.tools.javac.tree.JCTree;
import com.sun.tools.javac.tree.JCTree.JCArrayAccess;
import com.sun.tools.javac.tree.JCTree.JCAssign;
import com.sun.tools.javac.tree.JCTree.JCBinary;
import com.sun.tools.javac.tree.JCTree.JCBlock;
import com.sun.tools.javac.tree.JCTree.JCClassDecl;
import com.sun.tools.javac.tree.JCTree.JCExpression;
import com.sun.tools.javac.tree.JCTree.JCFieldAccess;
import com.sun.tools.javac.tree.JCTree.JCFunctionalExpression;
import com.sun.tools.javac.tree.JCTree.JCFunctionalExpression.CodeReflectionInfo;
import com.sun.tools.javac.tree.JCTree.JCIdent;
import com.sun.tools.javac.tree.JCTree.JCLambda;
import com.sun.tools.javac.tree.JCTree.JCLiteral;
import com.sun.tools.javac.tree.JCTree.JCMemberReference;
import com.sun.tools.javac.tree.JCTree.JCMemberReference.ReferenceKind;
import com.sun.tools.javac.tree.JCTree.JCMethodDecl;
import com.sun.tools.javac.tree.JCTree.JCMethodInvocation;
import com.sun.tools.javac.tree.JCTree.JCNewArray;
import com.sun.tools.javac.tree.JCTree.JCNewClass;
import com.sun.tools.javac.tree.JCTree.JCReturn;
import com.sun.tools.javac.tree.JCTree.JCVariableDecl;
import com.sun.tools.javac.tree.JCTree.JCAssert;
import com.sun.tools.javac.tree.JCTree.Tag;
import com.sun.tools.javac.tree.TreeInfo;
import com.sun.tools.javac.tree.TreeMaker;
import com.sun.tools.javac.tree.TreeTranslator;
import com.sun.tools.javac.util.Assert;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.JCDiagnostic.DiagnosticPosition;
import com.sun.tools.javac.util.ListBuffer;
import com.sun.tools.javac.util.Log;
import com.sun.tools.javac.util.Name;
import com.sun.tools.javac.util.Names;
import com.sun.tools.javac.util.Options;
import jdk.internal.java.lang.reflect.code.*;
import jdk.internal.java.lang.reflect.code.op.CoreOps;
import jdk.internal.java.lang.reflect.code.op.ExtendedOps;
import jdk.internal.java.lang.reflect.code.type.*;
import jdk.internal.java.lang.reflect.code.type.impl.JavaTypeImpl;

import javax.lang.model.element.Modifier;
import javax.lang.model.type.TypeKind;
import java.util.*;
import java.util.function.Function;
import java.util.function.Supplier;

import static com.sun.tools.javac.code.Flags.PARAMETER;
import static com.sun.tools.javac.code.Flags.SYNTHETIC;
import static com.sun.tools.javac.code.TypeTag.BOT;
import static com.sun.tools.javac.code.TypeTag.NONE;

/**
 * This a tree translator that adds the code model to all method declaration marked
 * with the {@code CodeReflection} annotation. The model is expressed using the code
 * reflection API (see jdk.internal.java.lang.reflect.code).
 */
public class ReflectMethods extends TreeTranslator {
    protected static final Context.Key<ReflectMethods> reflectMethodsKey = new Context.Key<>();

    public static ReflectMethods instance(Context context) {
        ReflectMethods instance = context.get(reflectMethodsKey);
        if (instance == null)
            instance = new ReflectMethods(context);
        return instance;
    }

    private final Types types;
    private final Names names;
    private final Symtab syms;
    private final Gen gen;
    private final Log log;
    private final boolean dumpIR;

    // @@@ Separate out mutable state
    private TreeMaker make;
    private ListBuffer<JCTree> classOps;
    // Also used by BodyScanner
    private Symbol.ClassSymbol currentClassSym;
    private int lambdaCount;

    @SuppressWarnings("this-escape")
    protected ReflectMethods(Context context) {
        context.put(reflectMethodsKey, this);
        Options options = Options.instance(context);
        dumpIR = options.isSet("dumpIR");
        names = Names.instance(context);
        syms = Symtab.instance(context);
        types = Types.instance(context);
        gen = Gen.instance(context);
        log = Log.instance(context);
    }

    // Cannot compute within constructor due to circular dependencies on bootstrap compilation
    // syms.objectType == null
    private Map<JavaType, Type> primitiveAndBoxTypeMap;
    Map<JavaType, Type> primitiveAndBoxTypeMap() {
        Map<JavaType, Type> m = primitiveAndBoxTypeMap;
        if (m == null) {
            m = primitiveAndBoxTypeMap = Map.ofEntries(
                    Map.entry(JavaType.BOOLEAN, syms.booleanType),
                    Map.entry(JavaType.BYTE, syms.byteType),
                    Map.entry(JavaType.SHORT, syms.shortType),
                    Map.entry(JavaType.CHAR, syms.charType),
                    Map.entry(JavaType.INT, syms.intType),
                    Map.entry(JavaType.LONG, syms.longType),
                    Map.entry(JavaType.FLOAT, syms.floatType),
                    Map.entry(JavaType.DOUBLE, syms.doubleType),
                    Map.entry(JavaType.J_L_OBJECT, syms.objectType),
                    Map.entry(JavaType.J_L_BOOLEAN, types.boxedTypeOrType(syms.booleanType)),
                    Map.entry(JavaType.J_L_BYTE, types.boxedTypeOrType(syms.byteType)),
                    Map.entry(JavaType.J_L_SHORT, types.boxedTypeOrType(syms.shortType)),
                    Map.entry(JavaType.J_L_CHARACTER, types.boxedTypeOrType(syms.charType)),
                    Map.entry(JavaType.J_L_INTEGER, types.boxedTypeOrType(syms.intType)),
                    Map.entry(JavaType.J_L_LONG, types.boxedTypeOrType(syms.longType)),
                    Map.entry(JavaType.J_L_FLOAT, types.boxedTypeOrType(syms.floatType)),
                    Map.entry(JavaType.J_L_DOUBLE, types.boxedTypeOrType(syms.doubleType))
            );
        }
        return m;
    }

    @Override
    public void visitMethodDef(JCMethodDecl tree) {
        if (tree.sym.attribute(syms.codeReflectionType.tsym) != null) {
            // if the method is annotated, scan it
            BodyScanner bodyScanner = new BodyScanner(tree);
            try {
                CoreOps.FuncOp funcOp = bodyScanner.scanMethod();
                if (dumpIR) {
                    // dump the method IR if requested
                    log.note(Notes.MethodIrDump(tree.sym.enclClass(), tree.sym, funcOp.toText()));
                }
                // create a static final field holding the op' string text.
                // The name of the field is foo$op, where 'foo' is the name of the corresponding method.
                classOps.add(opFieldDecl(tree.name, tree.getModifiers().flags, funcOp));
            } catch (UnsupportedASTException ex) {
                // whoops, some AST node inside the method body were not supported. Log it and move on.
                log.note(ex.tree, Notes.MethodIrSkip(tree.sym.enclClass(), tree.sym, ex.tree.getTag().toString()));
            }
        }
        super.visitMethodDef(tree);
    }

    @Override
    public void visitClassDef(JCClassDecl tree) {
        ListBuffer<JCTree> prevClassOps = classOps;
        Symbol.ClassSymbol prevClassSym = currentClassSym;
        int prevLambdaCount = lambdaCount;
        try {
            lambdaCount = 0;
            currentClassSym = tree.sym;
            classOps = new ListBuffer<>();
            super.visitClassDef(tree);
            tree.defs = tree.defs.prependList(classOps.toList());
        } finally {
            lambdaCount = prevLambdaCount;
            classOps = prevClassOps;
            currentClassSym = prevClassSym;
            result = tree;
        }
    }

    @Override
    public void visitLambda(JCLambda tree) {
        FunctionalExpressionKind kind = functionalKind(tree);
        if (kind.isQuoted) {
            // quoted lambda - scan it
            BodyScanner bodyScanner = new BodyScanner(tree, kind);
            try {
                CoreOps.FuncOp funcOp = bodyScanner.scanLambda();
                if (dumpIR) {
                    // dump the method IR if requested
                    log.note(Notes.QuotedIrDump(funcOp.toText()));
                }
                // create a static final field holding the op' string text.
                // The name of the field is foo$op, where 'foo' is the name of the corresponding method.
                JCVariableDecl opField = opFieldDecl(lambdaName(), 0, funcOp);
                classOps.add(opField);
                ListBuffer<JCExpression> capturedArgs = quotedCapturedArgs(tree, bodyScanner, null);

                switch (kind) {
                    case QUOTED_STRUCTURAL -> {
                        JCIdent opFieldId = make.Ident(opField.sym);
                        ListBuffer<JCExpression> interpreterArgs = new ListBuffer<>();
                        JCMethodInvocation parsedOp = make.App(make.Ident(syms.opParserFromString), com.sun.tools.javac.util.List.of(opFieldId));
                        interpreterArgs.append(parsedOp);
                        // append captured vars
                        interpreterArgs.appendList(capturedArgs.toList());

                        JCMethodInvocation interpreterInvoke = make.App(make.Ident(syms.opInterpreterInvoke), interpreterArgs.toList());
                        interpreterInvoke.varargsElement = syms.objectType;
                        super.visitLambda(tree);
                        result = interpreterInvoke;
                    }
                    case QUOTABLE -> {
                        // leave the lambda in place, but also leave a trail for LambdaToMethod
                        tree.codeReflectionInfo = new CodeReflectionInfo(opField.sym, capturedArgs.toList());
                        super.visitLambda(tree);
                    }
                }
            } catch (UnsupportedASTException ex) {
                // whoops, some AST node inside the quoted lambda body were not supported. Log it and move on.
                log.note(ex.tree, Notes.QuotedIrSkip(ex.tree.getTag().toString()));
                result = tree;
            }
        } else {
            super.visitLambda(tree);
        }
    }

    @Override
    public void visitReference(JCMemberReference tree) {
        FunctionalExpressionKind kind = functionalKind(tree);
        Assert.check(kind != FunctionalExpressionKind.QUOTED_STRUCTURAL,
                "structural quoting not supported for method references");
        MemberReferenceToLambda memberReferenceToLambda = new MemberReferenceToLambda(tree, currentClassSym);
        JCVariableDecl recvDecl = memberReferenceToLambda.receiverVar();
        JCLambda lambdaTree = memberReferenceToLambda.lambda();

        if (kind.isQuoted) {
            // quoted lambda - scan it
            BodyScanner bodyScanner = new BodyScanner(lambdaTree, kind);
            try {
                CoreOps.FuncOp funcOp = bodyScanner.scanLambda();
                if (dumpIR) {
                    // dump the method IR if requested
                    log.note(Notes.QuotedIrDump(funcOp.toText()));
                }
                // create a static final field holding the op' string text.
                // The name of the field is foo$op, where 'foo' is the name of the corresponding method.
                JCVariableDecl opField = opFieldDecl(lambdaName(), 0, funcOp);
                classOps.add(opField);
                ListBuffer<JCExpression> capturedArgs = quotedCapturedArgs(tree, bodyScanner, recvDecl);
                tree.codeReflectionInfo = new CodeReflectionInfo(opField.sym, capturedArgs.toList());
                super.visitReference(tree);
                if (recvDecl != null) {
                    result = copyReferenceWithReceiverVar(tree, recvDecl);
                }
            } catch (UnsupportedASTException ex) {
                // whoops, some AST node inside the quoted lambda body were not supported. Log it and move on.
                log.note(ex.tree, Notes.QuotedIrSkip(ex.tree.getTag().toString()));
                result = tree;
            }
        } else {
            super.visitReference(tree);
        }
    }

    ListBuffer<JCExpression> quotedCapturedArgs(DiagnosticPosition pos, BodyScanner bodyScanner, JCVariableDecl recvDecl) {
        ListBuffer<JCExpression> capturedArgs = new ListBuffer<>();
        for (Symbol capturedSym : bodyScanner.stack.localToOp.keySet()) {
            if (capturedSym.kind == Kind.TYP) {
                // captured this
                capturedArgs.add(make.at(pos).This(capturedSym.type));
            } else if (recvDecl != null && capturedSym == recvDecl.sym) {
                // captured method reference receiver
                capturedArgs.add(make.at(pos).Ident(recvDecl.sym));
            } else if (capturedSym.kind == Kind.VAR) {
                // captured var
                capturedArgs.add(make.at(pos).Ident(capturedSym));
            } else {
                throw new AssertionError("Unexpected captured symbol: " + capturedSym);
            }
        }
        return capturedArgs;
    }

    /*
     * Creates a let expression of the kind:
     * let $recv in $recv::memberRef
     *
     * This is required to make sure that LambdaToMethod doesn't end up emitting the
     * code for capturing the bound method reference receiver twice.
     */
    JCExpression copyReferenceWithReceiverVar(JCMemberReference ref, JCVariableDecl recvDecl) {
        JCMemberReference newRef = make.at(ref).Reference(ref.mode, ref.name, make.Ident(recvDecl.sym), ref.typeargs);
        newRef.type = ref.type;
        newRef.target = ref.target;
        newRef.refPolyKind = ref.refPolyKind;
        newRef.referentType = ref.referentType;
        newRef.kind = ref.kind;
        newRef.varargsElement = ref.varargsElement;
        newRef.ownerAccessible = ref.ownerAccessible;
        newRef.sym = ref.sym;
        newRef.codeReflectionInfo = ref.codeReflectionInfo;
        return make.at(ref).LetExpr(recvDecl, newRef).setType(newRef.type);
    }

    Name lambdaName() {
        return names.fromString("lambda").append('$', names.fromString(String.valueOf(lambdaCount++)));
    }

    private JCVariableDecl opFieldDecl(Name prefix, long flags, CoreOps.FuncOp op) {
        VarSymbol opFieldSym = new VarSymbol(flags | Flags.STATIC | Flags.FINAL | Flags.SYNTHETIC,
                prefix.append('$', names.fromString("op")),
                syms.stringType,
                currentClassSym);

        currentClassSym.members().enter(opFieldSym);
        JCLiteral opText = make.Literal(op.toText());
        JCVariableDecl opFieldTree = make.VarDef(opFieldSym, opText);
        return opFieldTree;
    }

    public JCTree translateTopLevelClass(JCTree cdef, TreeMaker make) {
        // note that this method does NOT support recursion.
        this.make = make;
        JCTree res = translate(cdef);
        return res;
    }

    public CoreOps.FuncOp getMethodBody(Symbol.ClassSymbol classSym, JCMethodDecl methodDecl, JCBlock attributedBody, TreeMaker make) {
        // if the method is annotated, scan it
        // Called from JavacElements::getBody
        try {
            this.make = make;
            currentClassSym = classSym;
            BodyScanner bodyScanner = new BodyScanner(methodDecl, attributedBody);
            return bodyScanner.scanMethod();
        } finally {
            currentClassSym = null;
            this.make = null;
        }
    }

    static class BodyStack {
        final BodyStack parent;

        // Tree associated with body
        final JCTree tree;

        // Body to add blocks
        final Body.Builder body;
        // Current block to add operations
        Block.Builder block;

        // Map of symbols (method arguments and local variables) to varOp values
        final Map<Symbol, Value> localToOp;

        // Label
        Map.Entry<String, Op.Result> label;

        BodyStack(BodyStack parent, JCTree tree, FunctionType bodyType) {
            this.parent = parent;

            this.tree = tree;

            this.body = Body.Builder.of(parent != null ? parent.body : null, bodyType);
            this.block = body.entryBlock();

            this.localToOp = new LinkedHashMap<>(); // order is important for captured values
        }

        public void setLabel(String labelName, Op.Result labelValue) {
            if (label != null) {
                throw new IllegalStateException("Label already defined: " + labelName);
            }
            label = Map.entry(labelName, labelValue);
        }
    }

    class BodyScanner extends FilterScanner {
        private final JCTree body;
        private final Name name;
        private final BodyStack top;
        private BodyStack stack;
        private Op lastOp;
        private Value result;
        private Type pt = Type.noType;
        private boolean isQuoted;
        private Type bodyTarget;

        // Only few AST nodes supported for now
        private static final Set<JCTree.Tag> SUPPORTED_TAGS =
                Set.of(Tag.VARDEF,
                        Tag.RETURN, Tag.THROW, Tag.BREAK, Tag.CONTINUE,
                        Tag.PLUS, Tag.MINUS, Tag.MUL, Tag.DIV, Tag.MOD,
                        Tag.NEG, Tag.NOT,
                        Tag.BITOR, Tag.BITAND, Tag.BITXOR,
                        Tag.PLUS_ASG, Tag.MINUS_ASG, Tag.MUL_ASG, Tag.DIV_ASG, Tag.MOD_ASG,
                        Tag.POSTINC, Tag.PREINC, Tag.POSTDEC, Tag.PREDEC,
                        Tag.EQ, Tag.NE, Tag.LT, Tag.LE, Tag.GT, Tag.GE,
                        Tag.AND, Tag.OR,
                        Tag.LITERAL, Tag.IDENT, Tag.TYPEIDENT, Tag.ASSIGN, Tag.EXEC, Tag.PARENS,
                        Tag.SELECT, Tag.INDEXED, Tag.APPLY,
                        Tag.TYPECAST, Tag.TYPETEST,
                        Tag.NEWCLASS, Tag.NEWARRAY, Tag.LAMBDA, Tag.REFERENCE,
                        Tag.BLOCK, Tag.IF, Tag.WHILELOOP, Tag.DOLOOP, Tag.FOREACHLOOP, Tag.FORLOOP, Tag.TRY,
                        Tag.SWITCH_EXPRESSION, Tag.YIELD,
                        Tag.CONDEXPR,
                        Tag.ASSERT,
                        Tag.LABELLED,
                        Tag.SKIP,
                        Tag.TYPEARRAY,
                        Tag.STRING_TEMPLATE);

        BodyScanner(JCMethodDecl tree) {
            this(tree, tree.body);
        }

        BodyScanner(JCMethodDecl tree, JCBlock body) {
            super(SUPPORTED_TAGS);
            this.body = body;
            this.name = tree.name;
            this.isQuoted = false;

            List<TypeElement> parameters = new ArrayList<>();
            int blockArgOffset = 0;
            // Instance methods model "this" as an additional argument occurring
            // before all other arguments.
            // @@@ Inner classes.
            // We need to capture all "this", in nested order, as arguments.
            if (!tree.getModifiers().getFlags().contains(Modifier.STATIC)) {
                parameters.add(typeToTypeElement(tree.sym.owner.type));
                blockArgOffset++;
            }
            tree.sym.type.getParameterTypes().stream().map(this::typeToTypeElement).forEach(parameters::add);

            FunctionType bodyType = FunctionType.functionType(
                    typeToTypeElement(tree.sym.type.getReturnType()), parameters);

            this.stack = this.top = new BodyStack(null, tree.body, bodyType);

            // @@@ this as local variable? (it can never be stored to)
            for (int i = 0 ; i < tree.params.size() ; i++) {
                Op.Result paramOp = append(CoreOps.var(
                        tree.params.get(i).name.toString(),
                        top.block.parameters().get(blockArgOffset + i)));
                top.localToOp.put(tree.params.get(i).sym, paramOp);
            }

            bodyTarget = tree.sym.type.getReturnType();
        }

        BodyScanner(JCLambda tree, FunctionalExpressionKind kind) {
            super(SUPPORTED_TAGS);
            assert kind != FunctionalExpressionKind.NOT_QUOTED;

            this.body = tree;
            this.name = names.fromString("quotedLambda");
            this.isQuoted = true;

            com.sun.tools.javac.util.List<Type> nil = com.sun.tools.javac.util.List.nil();
            Type quotedOpType = kind == FunctionalExpressionKind.QUOTABLE
                    ? syms.lambdaOpType
                    : syms.closureOpType;
            Type quotedReturnType = new ClassType(null,
                    com.sun.tools.javac.util.List.of(quotedOpType), syms.quotedType.tsym);
            MethodType mtype = new MethodType(nil, quotedReturnType, nil, syms.methodClass);
            FunctionType mtDesc = FunctionType.functionType(typeToTypeElement(mtype.restype),
                    mtype.getParameterTypes().map(this::typeToTypeElement));

            this.stack = this.top = new BodyStack(null, tree.body, mtDesc);

            bodyTarget = tree.target.getReturnType();
        }

        void pushBody(JCTree tree, FunctionType bodyType) {
            stack = new BodyStack(stack, tree, bodyType);
            lastOp = null; // reset
        }

        void popBody() {
            stack = stack.parent;
        }

        Value varOpValue(Symbol sym) {
            BodyStack s = stack;
            while (s != null) {
                Value v = s.localToOp.get(sym);
                if (v != null) {
                    return v;
                }
                s = s.parent;
            }
            if (isQuoted) {
                return capturedOpValue(sym);
            } else {
                throw new NoSuchElementException(sym.toString());
            }
        }

        Value capturedOpValue(Symbol sym) {
            var capturedVar = top.localToOp.get(sym);
            if (capturedVar == null) {
                var capturedArg = top.block.parameter(typeToTypeElement(sym.type));
                capturedVar = top.block.op(CoreOps.var(sym.name.toString(), capturedArg));
                top.localToOp.put(sym, capturedVar);
            }
            return capturedVar;
        }

        Value thisValue() { // @@@: outer this?
            if (isQuoted) {
                // capture this - add captured class symbol to the stack top local mappings
                var capturedThis = top.localToOp.get(currentClassSym);
                if (capturedThis == null) {
                    capturedThis = top.block.parameter(typeToTypeElement(currentClassSym.type));
                    top.localToOp.put(currentClassSym, capturedThis);
                }
                return capturedThis;
            } else {
                return top.block.parameters().get(0);
            }
        }

        Value getLabel(String labelName) {
            BodyStack s = stack;
            while (s != null) {
                if (s.label != null && s.label.getKey().equals(labelName)) {
                    return s.label.getValue();
                }
                s = s.parent;
            }
            throw new NoSuchElementException(labelName);
        }

        private Op.Result append(Op op) {
            return append(op, stack);
        }

        private Op.Result append(Op op, BodyStack stack) {
            lastOp = op;
            return stack.block.apply(op);
        }

        private <O extends Op & Op.Terminating> void appendTerminating(Supplier<O> sop) {
            // Append only if an existing terminating operation is not present
            if (lastOp == null || !(lastOp instanceof Op.Terminating)) {
                append(sop.get());
            }
        }

        public Value toValue(JCTree tree) {
            return toValue(tree, Type.noType);
        }

        public Value toValue(JCTree tree, Type target) {
            result = null; // reset
            Type prevPt = pt;
            try {
                pt = target;
                scan(tree);
                return result != null ?
                        coerce(result, tree.type, target) :
                        null;
            } finally {
                pt = prevPt;
            }
        }

        Value coerce(Value sourceValue, Type sourceType, Type targetType) {
            if (sourceType.isReference() && targetType.isReference() &&
                    !types.isSubtype(types.erasure(sourceType), types.erasure(targetType))) {
                return append(CoreOps.cast(typeToTypeElement(targetType), sourceValue));
            } else {
                return convert(sourceValue, targetType);
            }
        }

        Value boxIfNeeded(Value exprVal) {
            Type source = typeElementToType(exprVal.type());
            return source.hasTag(NONE) ?
                    exprVal : convert(exprVal, types.boxedTypeOrType(source));
        }

        Value unboxIfNeeded(Value exprVal) {
            Type source = typeElementToType(exprVal.type());
            return source.hasTag(NONE) ?
                    exprVal : convert(exprVal, types.unboxedTypeOrType(source));
        }

        Value convert(Value exprVal, Type target) {
            Type source = typeElementToType(exprVal.type());
            boolean sourcePrimitive = source.isPrimitive();
            boolean targetPrimitive = target.isPrimitive();
            if (target.hasTag(NONE)) {
                return exprVal;
            } else if (sourcePrimitive == targetPrimitive) {
                if (!sourcePrimitive || types.isSameType(source, target)) {
                    return exprVal;
                } else {
                    // implicit primitive conversion
                    return append(CoreOps.conv(typeToTypeElement(target), exprVal));
                }
            } else if (sourcePrimitive) {
                // we need to box
                Type unboxedTarget = types.unboxedType(target);
                if (!unboxedTarget.hasTag(NONE)) {
                    // non-Object target
                    return box(exprVal, target);
                } else {
                    // Object target
                    return box(exprVal, types.boxedClass(source).type);
                }
            } else {
                // we need to unbox
                return unbox(exprVal, source, target, types.unboxedType(source));
            }
        }

        Value box(Value valueExpr, Type box) {
            MethodRef boxMethod = MethodRef.method(typeToTypeElement(box), names.valueOf.toString(),
                    FunctionType.functionType(typeToTypeElement(box), typeToTypeElement(types.unboxedType(box))));
            return append(CoreOps.invoke(boxMethod, valueExpr));
        }

        Value unbox(Value valueExpr, Type box, Type primitive, Type unboxedType) {
            if (unboxedType.hasTag(NONE)) {
                // Object target, first downcast to correct wrapper type
                unboxedType = primitive;
                valueExpr = append(CoreOps.cast(typeToTypeElement(types.boxedClass(unboxedType).type), valueExpr));
            }
            MethodRef unboxMethod = MethodRef.method(typeToTypeElement(box),
                    unboxedType.tsym.name.append(names.Value).toString(),
                    FunctionType.functionType(typeToTypeElement(unboxedType), typeToTypeElement(box)));
            return append(CoreOps.invoke(unboxMethod, valueExpr));
        }

        @Override
        void skip(JCTree tree) {
            // this method is called for unsupported AST nodes (see 'SUPPORTED_TAGS')
            throw unsupported(tree);
        }

        @Override
        public void visitVarDef(JCVariableDecl tree) {
            Value initOp;
            if (tree.init != null) {
                initOp = toValue(tree.init, tree.type);
            } else {
                initOp = append(defaultValue(tree.type));
            }
            result = append(CoreOps.var(tree.name.toString(), typeToTypeElement(tree.type), initOp));
            stack.localToOp.put(tree.sym, result);
        }

        @Override
        public void visitAssign(JCAssign tree) {
            // Consume top node that applies to write access
            JCTree lhs = TreeInfo.skipParens(tree.lhs);
            Type target = tree.lhs.type;
            switch (lhs.getTag()) {
                case IDENT: {
                    JCIdent assign = (JCIdent) lhs;

                    // Scan the rhs, the assign expression result is its input
                    result = toValue(tree.rhs, target);

                    Symbol sym = assign.sym;
                    switch (sym.getKind()) {
                        case LOCAL_VARIABLE, PARAMETER -> {
                            Value varOp = varOpValue(sym);
                            append(CoreOps.varStore(varOp, result));
                        }
                        case FIELD -> {
                            FieldRef fd = symbolToFieldRef(sym, symbolSiteType(sym));
                            if (sym.isStatic()) {
                                append(CoreOps.fieldStore(fd, result));
                            } else {
                                append(CoreOps.fieldStore(fd, thisValue(), result));
                            }
                        }
                        default -> {
                            // @@@ Cannot reach here?
                            throw unsupported(tree);
                        }
                    }
                    break;
                }
                case SELECT: {
                    JCFieldAccess assign = (JCFieldAccess) lhs;

                    Value receiver = toValue(assign.selected);

                    // Scan the rhs, the assign expression result is its input
                    result = toValue(tree.rhs, target);

                    Symbol sym = assign.sym;
                    FieldRef fr = symbolToFieldRef(sym, assign.selected.type);
                    if (sym.isStatic()) {
                        append(CoreOps.fieldStore(fr, result));
                    } else {
                        append(CoreOps.fieldStore(fr, receiver, result));
                    }
                    break;
                }
                case INDEXED: {
                    JCArrayAccess assign = (JCArrayAccess) lhs;

                    Value array = toValue(assign.indexed);
                    Value index = toValue(assign.index);

                    // Scan the rhs, the assign expression result is its input
                    result = toValue(tree.rhs, target);

                    append(CoreOps.arrayStoreOp(array, index, result));
                    break;
                }
                default:
                    throw unsupported(tree);
            }
        }

        @Override
        public void visitAssignop(JCTree.JCAssignOp tree) {
            // Capture applying rhs and operation
            Function<Value, Value> scanRhs = (lhs) -> {
                Type unboxedType = types.unboxedTypeOrType(tree.type);
                JavaType resultType = typeToTypeElement(unboxedType);
                Value rhs = toValue(tree.rhs, unboxedType);
                lhs = unboxIfNeeded(lhs);

                Value assignOpResult = switch (tree.getTag()) {

                    // Arithmetic operations
                    case PLUS_ASG -> append(CoreOps.add(lhs, rhs));
                    case MINUS_ASG -> append(CoreOps.sub(lhs, rhs));
                    case MUL_ASG -> append(CoreOps.mul(lhs, rhs));
                    case DIV_ASG -> append(CoreOps.div(lhs, rhs));
                    case MOD_ASG -> append(CoreOps.mod(lhs, rhs));

                    default -> throw unsupported(tree);
                };
                return result = convert(assignOpResult, tree.type);
            };

            applyCompoundAssign(tree.lhs, scanRhs);
        }

        void applyCompoundAssign(JCTree.JCExpression lhs, Function<Value, Value> scanRhs) {
            // Consume top node that applies to access
            lhs = TreeInfo.skipParens(lhs);
            switch (lhs.getTag()) {
                case IDENT -> {
                    JCIdent assign = (JCIdent) lhs;

                    Symbol sym = assign.sym;
                    switch (sym.getKind()) {
                        case LOCAL_VARIABLE, PARAMETER -> {
                            Value varOp = varOpValue(sym);

                            Op.Result lhsOpValue = append(CoreOps.varLoad(varOp));
                            // Scan the rhs
                            Value r = scanRhs.apply(lhsOpValue);

                            append(CoreOps.varStore(varOp, r));
                        }
                        case FIELD -> {
                            FieldRef fr = symbolToFieldRef(sym, symbolSiteType(sym));

                            Op.Result lhsOpValue;
                            TypeElement resultType = typeToTypeElement(sym.type);
                            if (sym.isStatic()) {
                                lhsOpValue = append(CoreOps.fieldLoad(resultType, fr));
                            } else {
                                lhsOpValue = append(CoreOps.fieldLoad(resultType, fr, thisValue()));
                            }
                            // Scan the rhs
                            Value r = scanRhs.apply(lhsOpValue);

                            if (sym.isStatic()) {
                                append(CoreOps.fieldStore(fr, r));
                            } else {
                                append(CoreOps.fieldStore(fr, thisValue(), r));
                            }
                        }
                        default -> {
                            // @@@ Cannot reach here?
                            throw unsupported(lhs);
                        }
                    }
                }
                case SELECT -> {
                    JCFieldAccess assign = (JCFieldAccess) lhs;

                    Value receiver = toValue(assign.selected);

                    Symbol sym = assign.sym;
                    FieldRef fr = symbolToFieldRef(sym, assign.selected.type);

                    Op.Result lhsOpValue;
                    TypeElement resultType = typeToTypeElement(sym.type);
                    if (sym.isStatic()) {
                        lhsOpValue = append(CoreOps.fieldLoad(resultType, fr));
                    } else {
                        lhsOpValue = append(CoreOps.fieldLoad(resultType, fr, receiver));
                    }
                    // Scan the rhs
                    Value r = scanRhs.apply(lhsOpValue);

                    if (sym.isStatic()) {
                        append(CoreOps.fieldStore(fr, r));
                    } else {
                        append(CoreOps.fieldStore(fr, receiver, r));
                    }
                }
                case INDEXED -> {
                    JCArrayAccess assign = (JCArrayAccess) lhs;

                    Value array = toValue(assign.indexed);
                    Value index = toValue(assign.index);

                    Op.Result lhsOpValue = append(CoreOps.arrayLoadOp(array, index));
                    // Scan the rhs
                    Value r = scanRhs.apply(lhsOpValue);

                    append(CoreOps.arrayStoreOp(array, index, r));
                }
                default -> throw unsupported(lhs);
            }
        }

        @Override
        public void visitIdent(JCIdent tree) {
            // Visited only for read access

            Symbol sym = tree.sym;
            switch (sym.getKind()) {
                case LOCAL_VARIABLE, RESOURCE_VARIABLE, BINDING_VARIABLE, PARAMETER, EXCEPTION_PARAMETER -> {
                    Value varOp = varOpValue(sym);
                    if (varOp.type() instanceof VarType) {
                        // regular var
                        result = append(CoreOps.varLoad(varOp));
                    } else {
                        // captured value
                        result = varOp;
                    }
                }
                case FIELD, ENUM_CONSTANT -> {
                    if (sym.name.equals(names._this)) {
                        result = thisValue();
                    } else {
                        FieldRef fr = symbolToFieldRef(sym, symbolSiteType(sym));
                        TypeElement resultType = typeToTypeElement(sym.type);
                        if (sym.isStatic()) {
                            result = append(CoreOps.fieldLoad(resultType, fr));
                        } else {
                            result = append(CoreOps.fieldLoad(resultType, fr, thisValue()));
                        }
                    }
                }
                case INTERFACE, CLASS, ENUM -> {
                    result = null;
                }
                default -> {
                    // @@@ Cannot reach here?
                    throw unsupported(tree);
                }
            }
        }

        @Override
        public void visitTypeIdent(JCTree.JCPrimitiveTypeTree tree) {
            result = null;
        }

        @Override
        public void visitTypeArray(JCTree.JCArrayTypeTree tree) {
            result = null; // MyType[].class is handled in visitSelect just as MyType.class
        }

        @Override
        public void visitSelect(JCFieldAccess tree) {
            // Visited only for read access

            Type qualifierTarget = qualifierTarget(tree);
            // @@@: might cause redundant load if accessed symbol is static but the qualifier is not a type
            Value receiver = toValue(tree.selected);

            if (tree.name.equals(names._class)) {
                result = append(CoreOps.constant(JavaType.J_L_CLASS, typeToTypeElement(tree.selected.type)));
            } else if (types.isArray(tree.selected.type)) {
                if (tree.sym.equals(syms.lengthVar)) {
                    result = append(CoreOps.arrayLength(receiver));
                } else {
                    // Should not reach here
                    throw unsupported(tree);
                }
            } else {
                Symbol sym = tree.sym;
                switch (sym.getKind()) {
                    case FIELD, ENUM_CONSTANT -> {
                        FieldRef fr = symbolToFieldRef(sym, qualifierTarget.hasTag(NONE) ?
                                tree.selected.type : qualifierTarget);
                        TypeElement resultType = typeToTypeElement(types.memberType(tree.selected.type, sym));
                        if (sym.isStatic()) {
                            result = append(CoreOps.fieldLoad(resultType, fr));
                        } else {
                            result = append(CoreOps.fieldLoad(resultType, fr, receiver));
                        }
                    }
                    case INTERFACE, CLASS, ENUM -> {
                        result = null;
                    }
                    default -> {
                        // @@@ Cannot reach here?
                        throw unsupported(tree);
                    }
                }
            }
        }

        @Override
        public void visitIndexed(JCArrayAccess tree) {
            // Visited only for read access

            Value array = toValue(tree.indexed);

            Value index = toValue(tree.index);

            result = append(CoreOps.arrayLoadOp(array, index));
        }

        @Override
        public void visitApply(JCTree.JCMethodInvocation tree) {
            // @@@ Symbol.externalType, for use with inner classes

            // @@@ this.xyz(...) calls in a constructor

            // @@@ super.xyz(...) calls
            // Modeling with a call operation would result in the receiver type differing from that
            // in the method reference, perhaps that is sufficient?

            JCTree meth = TreeInfo.skipParens(tree.meth);
            switch (meth.getTag()) {
                case IDENT: {
                    JCIdent access = (JCIdent) meth;

                    Symbol sym = access.sym;
                    List<Value> args = new ArrayList<>();
                    if (!sym.isStatic()) {
                        args.add(thisValue());
                    }

                    args.addAll(scanMethodArguments(tree.args, tree.meth.type, tree.varargsElement));

                    MethodRef mr = symbolToErasedMethodRef(sym, symbolSiteType(sym));
                    Value res = append(CoreOps.invoke(typeToTypeElement(meth.type.getReturnType()), mr, args));
                    if (sym.type.getReturnType().getTag() != TypeTag.VOID) {
                        result = res;
                    }
                    break;
                }
                case SELECT: {
                    JCFieldAccess access = (JCFieldAccess) meth;

                    Type qualifierTarget = qualifierTarget(access);
                    Value receiver = toValue(access.selected, qualifierTarget);

                    Symbol sym = access.sym;
                    List<Value> args = new ArrayList<>();
                    if (!sym.isStatic()) {
                        args.add(receiver);
                    }

                    args.addAll(scanMethodArguments(tree.args, tree.meth.type, tree.varargsElement));

                    MethodRef mr = symbolToErasedMethodRef(sym, qualifierTarget.hasTag(NONE) ?
                            access.selected.type : qualifierTarget);
                    Value res = append(CoreOps.invoke(typeToTypeElement(meth.type.getReturnType()), mr, args));
                    if (sym.type.getReturnType().getTag() != TypeTag.VOID) {
                        result = res;
                    }
                    break;
                }
                default:
                    unsupported(meth);
            }
        }

        List<Value> scanMethodArguments(List<JCExpression> args, Type methodType, Type varargsElement) {
            ListBuffer<Value> argValues = new ListBuffer<>();
            com.sun.tools.javac.util.List<Type> targetTypes = methodType.getParameterTypes();
            if (varargsElement != null) {
                targetTypes = targetTypes.reverse().tail;
                for (int i = 0 ; i < args.size() - (methodType.getParameterTypes().size() - 1) ; i++) {
                    targetTypes = targetTypes.prepend(varargsElement);
                }
                targetTypes = targetTypes.reverse();
            }

            for (JCTree.JCExpression arg : args) {
                argValues.add(toValue(arg, targetTypes.head));
                targetTypes = targetTypes.tail;
            }
            return argValues.toList();
        }

        @Override
        public void visitReference(JCTree.JCMemberReference tree) {
            MemberReferenceToLambda memberReferenceToLambda = new MemberReferenceToLambda(tree, currentClassSym);
            JCVariableDecl recv = memberReferenceToLambda.receiverVar();
            if (recv != null) {
                scan(recv);
            }
            scan(memberReferenceToLambda.lambda());
        }

        Type qualifierTarget(JCFieldAccess tree) {
            Type selectedType = types.skipTypeVars(tree.selected.type, true);
            return selectedType.isCompound() ?
                    tree.sym.owner.type :
                    Type.noType;
        }

        @Override
        public void visitTypeCast(JCTree.JCTypeCast tree) {
            Value v = toValue(tree.expr);

            Type expressionType = tree.expr.type;
            Type type = tree.type;
            if (expressionType.isPrimitive() && type.isPrimitive()) {
                if (expressionType.equals(type)) {
                    // Redundant cast
                    result = v;
                } else {
                    result = append(CoreOps.conv(typeToTypeElement(type), v));
                }
            } else if (expressionType.isPrimitive() || type.isPrimitive()) {
                result = convert(v, tree.type);
            } else if (!expressionType.hasTag(BOT) &&
                    types.isAssignable(expressionType, type)) {
                // Redundant cast
                result = v;
            } else {
                // Reference cast
                JavaType jt = typeToTypeElement(types.erasure(type));
                result = append(CoreOps.cast(typeToTypeElement(type), jt, v));
            }
        }

        @Override
        public void visitTypeTest(JCTree.JCInstanceOf tree) {
            Value target = toValue(tree.expr);

            if (tree.pattern.getTag() != Tag.IDENT) {
                result = scanPattern(tree.getPattern(), target);
            } else {
                result = append(CoreOps.instanceOf(typeToTypeElement(tree.pattern.type), target));
            }
        }

        Value scanPattern(JCTree.JCPattern pattern, Value target) {
            // Type of pattern
            JavaType patternType;
            if (pattern instanceof JCTree.JCBindingPattern p) {
                patternType = ExtendedOps.Pattern.bindingType(typeToTypeElement(p.type));
            } else if (pattern instanceof JCTree.JCRecordPattern p) {
                patternType = ExtendedOps.Pattern.recordType(typeToTypeElement(p.record.type));
            } else {
                throw unsupported(pattern);
            }

            // Push pattern body
            pushBody(pattern, FunctionType.functionType(patternType));

            // @@@ Assumes just pattern nodes, likely will change when method patterns are supported
            //     that have expressions for any arguments (which perhaps in turn may have pattern expressions)
            List<JCVariableDecl> variables = new ArrayList<>();
            class PatternScanner extends FilterScanner {

                private Value result;

                public PatternScanner() {
                    super(Set.of(Tag.BINDINGPATTERN, Tag.RECORDPATTERN));
                }

                @Override
                public void visitBindingPattern(JCTree.JCBindingPattern binding) {
                    JCVariableDecl var = binding.var;
                    variables.add(var);

                    result = append(ExtendedOps.bindingPattern(typeToTypeElement(var.type), var.name.toString()));
                }

                @Override
                public void visitRecordPattern(JCTree.JCRecordPattern record) {
                    // @@@ Is always Identifier to record?
                    // scan(record.deconstructor);

                    List<Value> nestedValues = new ArrayList<>();
                    for (JCTree.JCPattern jcPattern : record.nested) {
                        nestedValues.add(toValue(jcPattern));
                    }

                    result = append(ExtendedOps.recordPattern(symbolToRecordTypeRef(record.record), nestedValues));
                }

                Value toValue(JCTree tree) {
                    result = null;
                    scan(tree);
                    return result;
                }
            }
            // Scan pattern
            Value patternValue = new PatternScanner().toValue(pattern);
            append(CoreOps._yield(patternValue));
            Body.Builder patternBody = stack.body;

            // Pop body
            popBody();

            // Find nearest ancestor body stack element associated with a statement tree
            // @@@ Strengthen check of tree?
            BodyStack _variablesStack = stack;
            while (!(_variablesStack.tree instanceof JCTree.JCStatement)) {
                _variablesStack = _variablesStack.parent;
            }
            BodyStack variablesStack = _variablesStack;

            // Create pattern var ops for pattern variables using the
            // builder associated with the nearest statement tree
            for (JCVariableDecl jcVar : variables) {
                Value init = variablesStack.block.op(defaultValue(jcVar.type));
                Op.Result op = variablesStack.block.op(CoreOps.var(jcVar.name.toString(), typeToTypeElement(jcVar.type), init));
                variablesStack.localToOp.put(jcVar.sym, op);
            }

            // Create pattern descriptor
            List<JavaType> patternDescParams = variables.stream().map(var -> typeToTypeElement(var.type)).toList();
            FunctionType matchFuncType = FunctionType.functionType(JavaType.VOID, patternDescParams);

            // Create the match body, assigning pattern values to pattern variables
            Body.Builder matchBody = Body.Builder.of(patternBody.ancestorBody(), matchFuncType);
            Block.Builder matchBuilder = matchBody.entryBlock();
            for (int i = 0; i < variables.size(); i++) {
                Value v = matchBuilder.parameters().get(i);
                Value var = variablesStack.localToOp.get(variables.get(i).sym);
                matchBuilder.op(CoreOps.varStore(var, v));
            }
            matchBuilder.op(CoreOps._yield());

            // Create the match operation
            return append(ExtendedOps.match(target, patternBody, matchBody));
        }

        @Override
        public void visitNewClass(JCTree.JCNewClass tree) {
            // @@@ Support anonymous classes
            if (tree.def != null) {
                throw unsupported(tree);
            }

            Type type = tree.type;
            Type outer = type.getEnclosingType();
            List<Value> args = new ArrayList<>();
            if (!outer.hasTag(TypeTag.NONE)) {
                // Obtain outer value for inner class, and add as first argument
                JCTree.JCExpression encl = tree.encl;
                Value outerInstance;
                if (encl == null) {
                    outerInstance = thisValue();
                } else {
                    outerInstance = toValue(tree.encl);
                }
                args.add(outerInstance);
            }

            // Create erased method type reference for constructor, where
            // the return type declares the class to instantiate
            // @@@ require symbol site type?
            MethodRef methodRef = symbolToErasedMethodRef(tree.constructor);
            FunctionType constructorType = FunctionType.functionType(
                    symbolToErasedDesc(tree.constructor.owner),
                    methodRef.type().parameterTypes());

            args.addAll(scanMethodArguments(tree.args, tree.constructorType, tree.varargsElement));

            result = append(CoreOps._new(typeToTypeElement(type), constructorType, args));
        }

        @Override
        public void visitNewArray(JCTree.JCNewArray tree) {
            if (tree.elems != null) {
                int length = tree.elems.size();
                Op.Result a = append(CoreOps.newArray(
                        typeToTypeElement(tree.type),
                        append(CoreOps.constant(JavaType.INT, length))));
                int i = 0;
                for (JCExpression elem : tree.elems) {
                    Value element = toValue(elem, types.elemtype(tree.type));
                    append(CoreOps.arrayStoreOp(
                            a,
                            append(CoreOps.constant(JavaType.INT, i)),
                            element));
                    i++;
                }

                result = a;
            } else {
                List<Value> indexes = new ArrayList<>();
                for (JCTree.JCExpression dim : tree.dims) {
                    indexes.add(toValue(dim));
                }

                JavaType arrayType = typeToTypeElement(tree.type);
                FunctionType constructorType = FunctionType.functionType(arrayType,
                        indexes.stream().map(Value::type).toList());
                result = append(CoreOps._new(arrayType, constructorType, indexes));
            }
        }

        @Override
        public void visitLambda(JCTree.JCLambda tree) {
            FunctionalExpressionKind kind = functionalKind(tree);
            final FunctionType lambdaType = switch (kind) {
                case QUOTED_STRUCTURAL -> typeToFunctionType(tree.target);
                default -> typeToFunctionType(types.findDescriptorType(tree.target));
            };

            // Push quoted body
            // We can either be explicitly quoted or a structural quoted expression
            // within some larger reflected code
            if (isQuoted || kind == FunctionalExpressionKind.QUOTED_STRUCTURAL) {
                pushBody(tree.body, FunctionType.VOID);
            }

            // Push lambda body
            pushBody(tree.body, lambdaType);

            // Map lambda parameters to varOp values
            for (int i = 0; i < tree.params.size(); i++) {
                JCVariableDecl p = tree.params.get(i);
                Op.Result paramOp = append(CoreOps.var(
                        p.name.toString(),
                        stack.block.parameters().get(i)));
                stack.localToOp.put(p.sym, paramOp);
            }

            // Scan the lambda body
            if (tree.getBodyKind() == LambdaExpressionTree.BodyKind.EXPRESSION) {
                Value exprVal = toValue(tree.body, tree.getDescriptorType(types).getReturnType());
                if (!tree.body.type.hasTag(TypeTag.VOID)) {
                    append(CoreOps._return(exprVal));
                } else {
                    appendTerminating(CoreOps::_return);
                }
            } else {
                Type prevBodyTarget = bodyTarget;
                try {
                    bodyTarget = tree.getDescriptorType(types).getReturnType();
                    toValue(tree.body);
                    // @@@ Check if unreachable
                    appendTerminating(CoreOps::_return);
                } finally {
                    bodyTarget = prevBodyTarget;
                }
            }

            Op lambdaOp = switch (kind) {
                case QUOTED_STRUCTURAL -> {
                    yield CoreOps.closure(stack.body);
                }
                case QUOTABLE, NOT_QUOTED -> {
                    // Get the functional interface type
                    JavaType fiType = typeToTypeElement(tree.target);
                    // build functional lambda
                    yield CoreOps.lambda(fiType, stack.body);
                }
            };

            // Pop lambda body
            popBody();

            Value lambdaResult = append(lambdaOp);

            if (isQuoted || kind == FunctionalExpressionKind.QUOTED_STRUCTURAL) {
                append(CoreOps._yield(lambdaResult));
                CoreOps.QuotedOp quotedOp = CoreOps.quoted(stack.body);

                // Pop quoted body
                popBody();

                lambdaResult = append(quotedOp);
            }

            result = lambdaResult;
        }

        @Override
        public void visitIf(JCTree.JCIf tree) {
            List<Body.Builder> bodies = new ArrayList<>();

            boolean first = true;
            while (tree != null) {
                // @@@ cond.type can be boolean or Boolean
                JCTree.JCExpression cond = TreeInfo.skipParens(tree.cond);

                // Push if condition
                pushBody(cond,
                        FunctionType.functionType(JavaType.BOOLEAN));
                Value last = toValue(cond);
                // Yield the boolean result of the condition
                append(CoreOps._yield(last));
                bodies.add(stack.body);

                // Pop if condition
                popBody();

                // Push if body
                pushBody(tree.thenpart, FunctionType.VOID);

                scan(tree.thenpart);
                appendTerminating(CoreOps::_yield);
                bodies.add(stack.body);

                // Pop if body
                popBody();

                JCTree.JCStatement elsepart = tree.elsepart;
                if (elsepart == null) {
                    tree = null;
                }
                else if (elsepart.getTag() == Tag.BLOCK) {
                    // Push else body
                    pushBody(elsepart, FunctionType.VOID);

                    scan(elsepart);
                    appendTerminating(CoreOps::_yield);
                    bodies.add(stack.body);

                    // Pop else body
                    popBody();

                    tree = null;
                } else if (elsepart.getTag() == Tag.IF) {
                    tree = (JCTree.JCIf) elsepart;
                }
                first = false;
            }

            append(ExtendedOps._if(bodies));
            result = null;
        }

        @Override
        public void visitSwitchExpression(JCTree.JCSwitchExpression tree) {
            Value target = toValue(tree.selector);

            FunctionType caseLabelType = FunctionType.functionType(JavaType.BOOLEAN, target.type());
            Type switchType = adaptBottom(tree.type);
            FunctionType actionType = FunctionType.functionType(typeToTypeElement(switchType));
            List<Body.Builder> bodies = new ArrayList<>();
            for (JCTree.JCCase c : tree.cases) {
                // Labels body
                JCTree.JCCaseLabel headCl = c.labels.head;
                if (headCl instanceof JCTree.JCPatternCaseLabel pcl) {
                    if (c.labels.size() > 1) {
                        throw unsupported(c);
                    }

                    pushBody(pcl, caseLabelType);

                    Value localTarget = stack.block.parameters().get(0);
                    final Value localResult;
                    if (c.guard != null) {
                        List<Body.Builder> clBodies = new ArrayList<>();

                        pushBody(pcl.pat, FunctionType.functionType(JavaType.BOOLEAN));
                        Value patVal = scanPattern(pcl.pat, localTarget);
                        append(CoreOps._yield(patVal));
                        clBodies.add(stack.body);
                        popBody();

                        pushBody(c.guard, FunctionType.functionType(JavaType.BOOLEAN));
                        append(CoreOps._yield(toValue(c.guard)));
                        clBodies.add(stack.body);
                        popBody();

                        localResult = append(ExtendedOps.conditionalAnd(clBodies));
                    } else {
                        localResult = scanPattern(pcl.pat, localTarget);
                    }
                    // Yield the boolean result of the condition
                    append(CoreOps._yield(localResult));
                    bodies.add(stack.body);

                    // Pop label
                    popBody();
                } else if (headCl instanceof JCTree.JCConstantCaseLabel ccl) {
                    pushBody(headCl, caseLabelType);

                    Value localTarget = stack.block.parameters().get(0);
                    final Value localResult;
                    if (c.labels.size() == 1) {
                        Value expr = toValue(ccl.expr);
                        // @@@ Conversion of localTarget
                        if (ccl.expr.type.isPrimitive()) {
                            localResult = append(CoreOps.eq(localTarget, expr));
                        } else {
                            localResult = append(CoreOps.invoke(
                                    MethodRef.method(Objects.class, "equals", boolean.class, Object.class, Object.class),
                                    localTarget, expr));
                        }
                    } else {
                        List<Body.Builder> clBodies = new ArrayList<>();
                        for (JCTree.JCCaseLabel cl : c.labels) {
                            ccl = (JCTree.JCConstantCaseLabel) cl;
                            pushBody(ccl, FunctionType.functionType(JavaType.BOOLEAN));

                            Value expr = toValue(ccl.expr);
                            // @@@ Conversion of localTarget
                            final Value labelResult;
                            if (ccl.expr.type.isPrimitive()) {
                                labelResult = append(CoreOps.eq(localTarget, expr));
                            } else {
                                labelResult = append(CoreOps.invoke(
                                        MethodRef.method(Objects.class, "equals", boolean.class, Object.class, Object.class),
                                        localTarget, expr));
                            }

                            append(CoreOps._yield(labelResult));
                            clBodies.add(stack.body);

                            // Pop label
                            popBody();
                        }

                        localResult = append(ExtendedOps.conditionalOr(clBodies));
                    }

                    append(CoreOps._yield(localResult));
                    bodies.add(stack.body);

                    // Pop labels
                    popBody();
                } else if (headCl instanceof JCTree.JCDefaultCaseLabel) {
                    // @@@ Do we need to model the default label body?
                    pushBody(headCl, FunctionType.VOID);

                    append(CoreOps._yield());
                    bodies.add(stack.body);

                    // Pop label
                    popBody();
                } else {
                    throw unsupported(tree);
                }

                // Statements body
                switch (c.caseKind) {
                    case RULE -> {
                        pushBody(c.body, actionType);
                        Type yieldType = adaptBottom(tree.type);
                        if (c.body instanceof JCExpression) {
                            // Yield the boolean result of the condition
                            Value bodyVal = toValue(c.body, yieldType);
                            append(CoreOps._yield(bodyVal));
                        } else {
                            // Otherwise there is a yield statement
                            Type prevBodyTarget = bodyTarget;
                            try {
                                bodyTarget = yieldType;
                                Value bodyVal = toValue(c.body);
                            } finally {
                                bodyTarget = prevBodyTarget;
                            }
                        }
                        bodies.add(stack.body);

                        // Pop block
                        popBody();
                    }
                    case STATEMENT -> {
                        // @@@ Avoid nesting for a single block? Goes against "say what you see"
                        // boolean oneBlock = c.stats.size() == 1 && c.stats.head instanceof JCBlock;
                        pushBody(c, actionType);

                        scan(c.stats);

                        appendTerminating(c.completesNormally
                                ? ExtendedOps::switchFallthroughOp
                                : CoreOps::unreachable);

                        bodies.add(stack.body);

                        // Pop block
                        popBody();
                    }
                };
            }

            result = append(ExtendedOps.switchExpression(actionType.returnType(), target, bodies));
        }

        @Override
        public void visitYield(JCTree.JCYield tree) {
            Value retVal = toValue(tree.value, bodyTarget);
            if (retVal == null) {
                result = append(ExtendedOps.java_yield());
            } else {
                result = append(ExtendedOps.java_yield(retVal));
            }
        }

        @Override
        public void visitWhileLoop(JCTree.JCWhileLoop tree) {
            // @@@ Patterns
            // @@@ cond.type can be boolean or Boolean
            JCTree.JCExpression cond = TreeInfo.skipParens(tree.cond);

            // Push while condition
            pushBody(cond, FunctionType.functionType(JavaType.BOOLEAN));
            Value last = toValue(cond);
            // Yield the boolean result of the condition
            append(CoreOps._yield(last));
            Body.Builder condition = stack.body;

            // Pop while condition
            popBody();

            // Push while body
            pushBody(tree.body, FunctionType.VOID);
            scan(tree.body);
            appendTerminating(ExtendedOps::_continue);
            Body.Builder body = stack.body;

            // Pop while body
            popBody();

            append(ExtendedOps._while(condition, body));
            result = null;
        }

        @Override
        public void visitDoLoop(JCTree.JCDoWhileLoop tree) {
            // @@@ Patterns
            // @@@ cond.type can be boolean or Boolean
            JCTree.JCExpression cond = TreeInfo.skipParens(tree.cond);

            // Push while body
            pushBody(tree.body, FunctionType.VOID);
            scan(tree.body);
            appendTerminating(ExtendedOps::_continue);
            Body.Builder body = stack.body;

            // Pop while body
            popBody();

            // Push while condition
            pushBody(cond, FunctionType.functionType(JavaType.BOOLEAN));
            Value last = toValue(cond);
            // Yield the boolean result of the condition
            append(CoreOps._yield(last));
            Body.Builder condition = stack.body;

            // Pop while condition
            popBody();

            append(ExtendedOps.doWhile(body, condition));
            result = null;
        }

        @Override
        public void visitForeachLoop(JCTree.JCEnhancedForLoop tree) {
            // Push expression
            pushBody(tree.expr, FunctionType.functionType(typeToTypeElement(tree.expr.type)));
            Value last = toValue(tree.expr);
            // Yield the Iterable result of the expression
            append(CoreOps._yield(last));
            Body.Builder expression = stack.body;

            // Pop expression
            popBody();

            JCVariableDecl var = tree.getVariable();
            JavaType eType = typeToTypeElement(var.type);
            VarType varEType = VarType.varType(typeToTypeElement(var.type));

            // Push init
            // @@@ When lhs assignment is a pattern we embed the pattern match into the init body and
            // return the bound variables
            pushBody(var, FunctionType.functionType(varEType, eType));
            Op.Result varEResult = append(CoreOps.var(var.name.toString(), stack.block.parameters().get(0)));
            append(CoreOps._yield(varEResult));
            Body.Builder init = stack.body;
            // Pop init
            popBody();

            // Push body
            pushBody(tree.body, FunctionType.functionType(JavaType.VOID, varEType));
            stack.localToOp.put(var.sym, stack.block.parameters().get(0));

            scan(tree.body);
            appendTerminating(ExtendedOps::_continue);
            Body.Builder body = stack.body;
            // Pop body
            popBody();

            append(ExtendedOps.enhancedFor(expression, init, body));
            result = null;
        }

        @Override
        public void visitForLoop(JCTree.JCForLoop tree) {
            class VarDefScanner extends FilterScanner {
                final List<JCVariableDecl> decls;

                public VarDefScanner() {
                    super(Set.of(Tag.VARDEF));
                    this.decls = new ArrayList<>();
                }

                @Override
                public void visitVarDef(JCVariableDecl tree) {
                    decls.add(tree);
                }

                void mapVarsToBlockArguments() {
                    for (int i = 0; i < decls.size(); i++) {
                        stack.localToOp.put(decls.get(i).sym, stack.block.parameters().get(i));
                    }
                }

                List<VarType> varTypes() {
                    return decls.stream()
                            .map(t -> VarType.varType(typeToTypeElement(t.type)))
                            .toList();
                }

                List<Value> varValues() {
                    return decls.stream()
                            .map(t -> stack.localToOp.get(t.sym))
                            .toList();
                }
            }

            // Scan local variable declarations
            VarDefScanner vds = new VarDefScanner();
            vds.scan(tree.init);
            List<VarType> varTypes = vds.varTypes();

            // Push init
            if (varTypes.size() > 1) {
                pushBody(null, FunctionType.functionType(TupleType.tupleType(varTypes)));
                scan(tree.init);

                // Capture all local variable declarations in tuple
                append(CoreOps._yield(append(CoreOps.tuple(vds.varValues()))));
            } else if (varTypes.size() == 1) {
                pushBody(null, FunctionType.functionType(varTypes.get(0)));
                scan(tree.init);

                append(CoreOps._yield(vds.varValues().get(0)));
            } else {
                pushBody(null, FunctionType.VOID);
                scan(tree.init);

                append(CoreOps._yield());
            }
            Body.Builder init = stack.body;

            // Pop init
            popBody();

            // Push cond
            pushBody(tree.cond, FunctionType.functionType(JavaType.BOOLEAN, varTypes));
            if (tree.cond != null) {
                vds.mapVarsToBlockArguments();

                Value last = toValue(tree.cond);
                // Yield the boolean result of the condition
                append(CoreOps._yield(last));
            } else {
                append(CoreOps._yield(append(CoreOps.constant(JavaType.BOOLEAN, true))));
            }
            Body.Builder cond = stack.body;

            // Pop cond
            popBody();

            // Push update
            // @@@ tree.step is a List<JCStatement>
            pushBody(null, FunctionType.functionType(JavaType.VOID, varTypes));
            if (!tree.step.isEmpty()) {
                vds.mapVarsToBlockArguments();

                scan(tree.step);
            }
            append(CoreOps._yield());
            Body.Builder update = stack.body;

            // Pop update
            popBody();

            // Push body
            pushBody(tree.body, FunctionType.functionType(JavaType.VOID, varTypes));
            if (tree.body != null) {
                vds.mapVarsToBlockArguments();

                scan(tree.body);
            }
            appendTerminating(ExtendedOps::_continue);
            Body.Builder body = stack.body;

            // Pop update
            popBody();

            append(ExtendedOps._for(init, cond, update, body));
            result = null;
        }

        @Override
        public void visitConditional(JCTree.JCConditional tree) {
            List<Body.Builder> bodies = new ArrayList<>();

            JCTree.JCExpression cond = TreeInfo.skipParens(tree.cond);

            // Push condition
            pushBody(cond,
                    FunctionType.functionType(JavaType.BOOLEAN));
            Value condVal = toValue(cond);
            // Yield the boolean result of the condition
            append(CoreOps._yield(condVal));
            bodies.add(stack.body);

            // Pop condition
            popBody();

            JCTree.JCExpression truepart = TreeInfo.skipParens(tree.truepart);

            Type condType = adaptBottom(tree.type);

            // Push true body
            pushBody(truepart,
                    FunctionType.functionType(typeToTypeElement(condType)));

            Value trueVal = toValue(truepart, condType);
            // Yield the result
            append(CoreOps._yield(trueVal));
            bodies.add(stack.body);

            // Pop true body
            popBody();

            JCTree.JCExpression falsepart = TreeInfo.skipParens(tree.falsepart);

            // Push false body
            pushBody(falsepart,
                    FunctionType.functionType(typeToTypeElement(condType)));

            Value falseVal = toValue(falsepart, condType);
            // Yield the result
            append(CoreOps._yield(falseVal));
            bodies.add(stack.body);

            // Pop false body
            popBody();

            result = append(ExtendedOps.conditionalExpression(typeToTypeElement(condType), bodies));
        }

        private Type condType(JCExpression tree, Type type) {
            if (type.hasTag(BOT)) {
                return adaptBottom(tree.type);
            } else {
                return type;
            }
        }

        private Type adaptBottom(Type type) {
            return type.hasTag(BOT) ?
                    (pt.hasTag(NONE) ? syms.objectType : pt) :
                    type;
        }

        @Override
        public void visitAssert(JCAssert tree) {
            // assert <cond:body1> [detail:body2]

            List<Body.Builder> bodies = new ArrayList<>();
            JCTree.JCExpression cond = TreeInfo.skipParens(tree.cond);

            // Push condition
            pushBody(cond,
                    FunctionType.functionType(JavaType.BOOLEAN));
            Value condVal = toValue(cond);

            // Yield the boolean result of the condition
            append(CoreOps._yield(condVal));
            bodies.add(stack.body);

            // Pop condition
            popBody();

            if (tree.detail != null) {
                JCTree.JCExpression detail = TreeInfo.skipParens(tree.detail);

                pushBody(detail,
                         FunctionType.functionType(JavaType.J_L_STRING));
                Value detailVal = toValue(detail);

                append(CoreOps._yield(detailVal));
                bodies.add(stack.body);

                //Pop detail
                popBody();
            }

            result = append(CoreOps._assert(bodies));

        }

        @Override
        public void visitBlock(JCTree.JCBlock tree) {
            if (stack.tree == tree) {
                // Block is associated with the visit of a parent structure
                scan(tree.stats);
            } else {
                // Otherwise, independent block structure
                // @@@ Support synchronized blocks
                // Push block
                pushBody(tree, FunctionType.VOID);
                scan(tree.stats);
                appendTerminating(CoreOps::_yield);
                Body.Builder body = stack.body;

                // Pop block
                popBody();

                append(ExtendedOps.block(body));
            }
            result = null;
        }

        @Override
        public void visitLabelled(JCTree.JCLabeledStatement tree) {
            // Push block
            pushBody(tree, FunctionType.VOID);
            // Create constant for label
            String labelName = tree.label.toString();
            Op.Result label = append(CoreOps.constant(JavaType.J_L_STRING, labelName));
            // Set label on body stack
            stack.setLabel(labelName, label);
            scan(tree.body);
            appendTerminating(CoreOps::_yield);
            Body.Builder body = stack.body;

            // Pop block
            popBody();

            result = append(ExtendedOps.labeled(body));
        }

        @Override
        public void visitTry(JCTree.JCTry tree) {
            List<JCVariableDecl> rVariableDecls = new ArrayList<>();
            List<TypeElement> rTypes = new ArrayList<>();
            Body.Builder resources;
            if (!tree.resources.isEmpty()) {
                // Resources body returns a tuple that contains the resource variables/values
                // in order of declaration
                for (JCTree resource : tree.resources) {
                    if (resource instanceof JCVariableDecl vdecl) {
                        rVariableDecls.add(vdecl);
                        rTypes.add(VarType.varType(typeToTypeElement(vdecl.type)));
                    } else {
                        rTypes.add(typeToTypeElement(resource.type));
                    }
                }

                // Push resources body
                pushBody(null, FunctionType.functionType(TupleType.tupleType(rTypes)));

                List<Value> rValues = new ArrayList<>();
                for (JCTree resource : tree.resources) {
                    rValues.add(toValue(resource));
                }

                append(CoreOps._yield(append(CoreOps.tuple(rValues))));
                resources = stack.body;

                // Pop resources body
                popBody();
            } else {
                resources = null;
            }

            // Push body
            // Try body accepts the resource variables (in order of declaration).
            List<VarType> rVarTypes = rTypes.stream().<VarType>mapMulti((t, c) -> {
                if (t instanceof VarType vt) {
                    c.accept(vt);
                }
            }).toList();
            pushBody(tree.body, FunctionType.functionType(JavaType.VOID, rVarTypes));
            for (int i = 0; i < rVariableDecls.size(); i++) {
                stack.localToOp.put(rVariableDecls.get(i).sym, stack.block.parameters().get(i));
            }
            scan(tree.body);
            appendTerminating(CoreOps::_yield);
            Body.Builder body = stack.body;

            // Pop block
            popBody();

            List<Body.Builder> catchers = new ArrayList<>();
            for (JCTree.JCCatch catcher : tree.catchers) {
                // @@@ Support union types, how to express that plus lub?
                if (catcher.param.type.isUnion()) {
                    throw unsupported(tree);
                }

                // Push body
                pushBody(catcher.body, FunctionType.functionType(JavaType.VOID, typeToTypeElement(catcher.param.type)));
                Op.Result exVariable = append(CoreOps.var(
                        catcher.param.name.toString(),
                        stack.block.parameters().get(0)));
                stack.localToOp.put(catcher.param.sym, exVariable);
                scan(catcher.body);
                appendTerminating(CoreOps::_yield);
                catchers.add(stack.body);

                // Pop block
                popBody();
            }

            Body.Builder finalizer;
            if (tree.finalizer != null) {
                // Push body
                pushBody(tree.finalizer, FunctionType.VOID);
                scan(tree.finalizer);
                appendTerminating(CoreOps::_yield);
                finalizer = stack.body;

                // Pop block
                popBody();
            }
            else {
                finalizer = null;
            }

            result = append(ExtendedOps._try(resources, body, catchers, finalizer));
        }

        @Override
        public void visitUnary(JCTree.JCUnary tree) {
            Tag tag = tree.getTag();
            switch (tag) {
                case POSTINC, POSTDEC, PREINC, PREDEC -> {
                    // Capture applying rhs and operation
                    Function<Value, Value> scanRhs = (lhs) -> {
                        Value one = append(numericOneValue(tree.type));
                        Value unboxedLhs = unboxIfNeeded(lhs);

                        Value unboxedLhsPlusOne = switch (tree.getTag()) {
                            // Arithmetic operations
                            case POSTINC, PREINC -> append(CoreOps.add(unboxedLhs, one));
                            case POSTDEC, PREDEC -> append(CoreOps.sub(unboxedLhs, one));

                            default -> throw unsupported(tree);
                        };
                        Value lhsPlusOne = convert(unboxedLhsPlusOne, tree.type);

                        // Assign expression result
                        result =  switch (tree.getTag()) {
                            case POSTINC, POSTDEC -> lhs;
                            case PREINC, PREDEC -> lhsPlusOne;

                            default -> throw unsupported(tree);
                        };
                        return lhsPlusOne;
                    };

                    applyCompoundAssign(tree.arg, scanRhs);
                }
                case NEG -> {
                    Value rhs = toValue(tree.arg);
                    result = append(CoreOps.neg(rhs));
                }
                case NOT -> {
                    Value rhs = toValue(tree.arg);
                    result = append(CoreOps.not(rhs));
                }
                default -> throw unsupported(tree);
            }
        }

        @Override
        public void visitBinary(JCBinary tree) {
            Tag tag = tree.getTag();
            if (tag == Tag.AND || tag == Tag.OR) {
                // Logical operations
                // @@@ Flatten nested sequences

                // Push lhs
                pushBody(tree.lhs, FunctionType.functionType(JavaType.BOOLEAN));
                Value lhs = toValue(tree.lhs);
                // Yield the boolean result of the condition
                append(CoreOps._yield(lhs));
                Body.Builder bodyLhs = stack.body;

                // Pop lhs
                popBody();

                // Push rhs
                pushBody(tree.rhs, FunctionType.functionType(JavaType.BOOLEAN));
                Value rhs = toValue(tree.rhs);
                // Yield the boolean result of the condition
                append(CoreOps._yield(rhs));
                Body.Builder bodyRhs = stack.body;

                // Pop lhs
                popBody();

                List<Body.Builder> bodies = List.of(bodyLhs, bodyRhs);
                result = append(tag == Tag.AND
                        ? ExtendedOps.conditionalAnd(bodies)
                        : ExtendedOps.conditionalOr(bodies));
            } else {
                Type opType = tree.operator.type.getParameterTypes().get(0);
                Value lhs = toValue(tree.lhs, opType);
                Value rhs = toValue(tree.rhs, opType);

                result = switch (tag) {
                    // Arithmetic operations
                    case PLUS -> append(CoreOps.add(lhs, rhs));
                    case MINUS -> append(CoreOps.sub(lhs, rhs));
                    case MUL -> append(CoreOps.mul(lhs, rhs));
                    case DIV -> append(CoreOps.div(lhs, rhs));
                    case MOD -> append(CoreOps.mod(lhs, rhs));

                    // Test operations
                    case EQ -> append(CoreOps.eq(lhs, rhs));
                    case NE -> append(CoreOps.neq(lhs, rhs));
                    //
                    case LT -> append(CoreOps.lt(lhs, rhs));
                    case LE -> append(CoreOps.le(lhs, rhs));
                    case GT -> append(CoreOps.gt(lhs, rhs));
                    case GE -> append(CoreOps.ge(lhs, rhs));

                    // Bitwise operations (including their boolean variants)
                    case BITOR -> append(CoreOps.or(lhs, rhs));
                    case BITAND -> append(CoreOps.and(lhs, rhs));
                    case BITXOR -> append(CoreOps.xor(lhs, rhs));

                    default -> throw unsupported(tree);
                };
            }
        }

        @Override
        public void visitLiteral(JCLiteral tree) {
            Object value = switch (tree.type.getTag()) {
                case BOOLEAN -> tree.value instanceof Integer i && i == 1;
                case CHAR -> (char) (int) tree.value;
                default -> tree.value;
            };
            Type constantType = adaptBottom(tree.type);
            result = append(CoreOps.constant(typeToTypeElement(constantType), value));
        }

        @Override
        public void visitReturn(JCReturn tree) {
            Value retVal = toValue(tree.expr, bodyTarget);
            if (retVal == null) {
                result = append(CoreOps._return());
            } else {
                result = append(CoreOps._return(retVal));
            }
        }

        @Override
        public void visitThrow(JCTree.JCThrow tree) {
            Value throwVal = toValue(tree.expr);
            result = append(CoreOps._throw(throwVal));
        }

        @Override
        public void visitBreak(JCTree.JCBreak tree) {
            Value label = tree.label != null
                    ? getLabel(tree.label.toString())
                    : null;
            result = append(ExtendedOps._break(label));
        }

        @Override
        public void visitContinue(JCTree.JCContinue tree) {
            Value label = tree.label != null
                    ? getLabel(tree.label.toString())
                    : null;
            result = append(ExtendedOps._continue(label));
        }

        @Override
        public void visitClassDef(JCClassDecl tree) {
            // do nothing
        }


        @Override
        public void visitStringTemplate(JCTree.JCStringTemplate tree) {
            Value processor = toValue(tree.processor);
            List<Value> fragments = tree.fragments.map(f -> append(CoreOps.constant(JavaType.J_L_STRING, f)));

            List<Body.Builder> expressions = new ArrayList<>();
            tree.expressions.forEach(e -> {
                pushBody(e, FunctionType.functionType(typeToTypeElement(e.type)));
                append(CoreOps._yield(toValue(e)));
                expressions.add(stack.body);
                popBody();
            });
            result = append(ExtendedOps.stringTemplate(typeToTypeElement(tree.type), processor, fragments, expressions));
        }

        UnsupportedASTException unsupported(JCTree tree) {
            return new UnsupportedASTException(tree);
        }

        CoreOps.FuncOp scanMethod() {
            scan(body);
            // @@@ Check if unreachable
            appendTerminating(CoreOps::_return);
            return CoreOps.func(name.toString(), stack.body);
        }

        CoreOps.FuncOp scanLambda() {
            scan(body);
            append(CoreOps._return(result));
            return CoreOps.func(name.toString(), stack.body);
        }

        JavaType symbolToDesc(Symbol s) {
            return typeToTypeElement(s.type);
        }

        JavaType symbolToErasedDesc(Symbol s) {
            return typeToTypeElement(s.erasure(types));
        }

        JavaType typeToTypeElement(Type t) {
            t = normalizeType(t);
            return switch (t.getTag()) {
                case VOID -> JavaType.VOID;
                case CHAR -> JavaType.CHAR;
                case BOOLEAN -> JavaType.BOOLEAN;
                case BYTE -> JavaType.BYTE;
                case SHORT -> JavaType.SHORT;
                case INT -> JavaType.INT;
                case FLOAT -> JavaType.FLOAT;
                case LONG -> JavaType.LONG;
                case DOUBLE -> JavaType.DOUBLE;
                case ARRAY -> {
                    int dims = 1;
                    Type et = ((ArrayType)t).elemtype;
                    while (et.getKind() == TypeKind.ARRAY) {
                        et = ((ArrayType) et).elemtype;
                        dims++;
                    }

                    JavaType etd = typeToTypeElement(et);
                    yield new JavaTypeImpl(etd.rawType().toString(), dims, etd.typeArguments());
                }
                case CLASS -> {
                    // @@@ Need to clean this up, probably does not work inner generic classes
                    // whose enclosing class is also generic
                    List<JavaType> typeArguments;
                    if (t.getTypeArguments().nonEmpty()) {
                        typeArguments = new ArrayList<>();
                        for (Type ta : t.getTypeArguments()) {
                            typeArguments.add(typeToTypeElement(ta));
                        }
                    } else {
                        typeArguments = List.of();
                    }

                    // Use flat name to ensure demarcation of nested classes
                    yield new JavaTypeImpl(t.tsym.flatName().toString(), 0, typeArguments);
                }
                default -> {
                    throw new UnsupportedOperationException("Unsupported type: kind=" + t.getKind() + " type=" + t);
                }
            };
        }

        Type symbolSiteType(Symbol s) {
            boolean isMember = s.owner == syms.predefClass ||
                    s.isMemberOf(currentClassSym, types);
            return isMember ? currentClassSym.type : s.owner.type;
        }

        FieldRef symbolToFieldRef(Symbol s, Type site) {
            // @@@ Made Gen::binaryQualifier public, duplicate logic?
            // Ensure correct qualifying class is used in the reference, see JLS 13.1
            // https://docs.oracle.com/javase/specs/jls/se20/html/jls-13.html#jls-13.1
            return symbolToFieldRef(gen.binaryQualifier(s, site));
        }

        FieldRef symbolToFieldRef(Symbol s) {
            return FieldRef.field(
                    typeToTypeElement(s.owner.type),
                    s.name.toString(),
                    typeToTypeElement(s.type));
        }

        MethodRef symbolToMethodRef(Symbol s) {
            return MethodRef.method(
                    typeToTypeElement(s.owner.type),
                    s.name.toString(),
                    typeToTypeElement(s.type.getReturnType()),
                    s.type.getParameterTypes().stream().map(this::typeToTypeElement).toArray(TypeElement[]::new));
        }

        MethodRef symbolToErasedMethodRef(Symbol s, Type site) {
            // @@@ Made Gen::binaryQualifier public, duplicate logic?
            // Ensure correct qualifying class is used in the reference, see JLS 13.1
            // https://docs.oracle.com/javase/specs/jls/se20/html/jls-13.html#jls-13.1
            return symbolToErasedMethodRef(gen.binaryQualifier(s, site));
        }

        MethodRef symbolToErasedMethodRef(Symbol s) {
            Type erasedType = s.erasure(types);
            return MethodRef.method(
                    typeToTypeElement(s.owner.erasure(types)),
                    s.name.toString(),
                    typeToTypeElement(erasedType.getReturnType()),
                    erasedType.getParameterTypes().stream().map(this::typeToTypeElement).toArray(TypeElement[]::new));
        }

        FunctionType symbolToFunctionType(Symbol s) {
            return typeToFunctionType(s.type);
        }

        FunctionType typeToFunctionType(Type t) {
            return FunctionType.functionType(
                    typeToTypeElement(t.getReturnType()),
                    t.getParameterTypes().stream().map(this::typeToTypeElement).toArray(TypeElement[]::new));
        }

        RecordTypeRef symbolToRecordTypeRef(Symbol.ClassSymbol s) {
            TypeElement recordType = typeToTypeElement(s.type);
            List<RecordTypeRef.ComponentRef> components = s.getRecordComponents().stream()
                    .map(rc -> new RecordTypeRef.ComponentRef(typeToTypeElement(rc.type), rc.name.toString()))
                    .toList();
            return RecordTypeRef.recordType(recordType, components);
        }

        Op defaultValue(Type t) {
            return switch (t.getTag()) {
                case CHAR -> CoreOps.constant(typeToTypeElement(t), (char)0);
                case BOOLEAN -> CoreOps.constant(typeToTypeElement(t), false);
                case SHORT -> CoreOps.constant(typeToTypeElement(t), (short)0);
                case INT -> CoreOps.constant(typeToTypeElement(t), 0);
                case FLOAT -> CoreOps.constant(typeToTypeElement(t), 0f);
                case LONG -> CoreOps.constant(typeToTypeElement(t), 0L);
                case DOUBLE -> CoreOps.constant(typeToTypeElement(t), 0d);
                default -> CoreOps.constant(typeToTypeElement(t), null);
            };
        }

        Op numericOneValue(Type t) {
            return switch (t.getTag()) {
                case CHAR -> CoreOps.constant(typeToTypeElement(t), (char)1);
                case SHORT -> CoreOps.constant(typeToTypeElement(t), (short)1);
                case INT -> CoreOps.constant(typeToTypeElement(t), 1);
                case FLOAT -> CoreOps.constant(typeToTypeElement(t), 1f);
                case LONG -> CoreOps.constant(typeToTypeElement(t), 1L);
                case DOUBLE -> CoreOps.constant(typeToTypeElement(t), 1d);
                case CLASS -> numericOneValue(types.unboxedType(t));
                default -> throw new UnsupportedOperationException(t.toString());
            };
        }

        Type normalizeType(Type t) {
            return switch (t.getTag()) {
                case METHOD -> new MethodType(t.getParameterTypes().map(this::normalizeType),
                        normalizeType(t.getReturnType()), t.getThrownTypes().map(this::normalizeType),
                        syms.methodClass);
                default -> checkDenotableInTypeDesc(t) ? t : types.erasure(t);
            };
        }

        Type typeElementToType(TypeElement desc) {
            return primitiveAndBoxTypeMap().getOrDefault(desc, Type.noType);
        }

        public boolean checkDenotableInTypeDesc(Type t) {
            return denotableChecker.visit(t, null);
        }
        // where

        /**
         * A type visitor that descends into the given type looking for types that are non-denotable
         * in code model types. Examples of such types are: type-variables (regular or captured),
         * wildcard type argument, intersection types, union types. The visit methods return false
         * as soon as a non-denotable type is encountered and true otherwise. (see {@link Check#checkDenotable(Type)}.
         */
        private static final Types.SimpleVisitor<Boolean, Void> denotableChecker = new Types.SimpleVisitor<>() {
            @Override
            public Boolean visitType(Type t, Void s) {
                return true;
            }
            @Override
            public Boolean visitClassType(ClassType t, Void s) {
                if (t.isUnion() || t.isIntersection()) {
                    // union and intersections cannot be denoted in code model types
                    return false;
                }
                // @@@ What about enclosing types?
                for (Type targ : t.getTypeArguments()) {
                    // propagate into type arguments
                    if (!visit(targ, s)) {
                        return false;
                    }
                }
                return true;
            }

            @Override
            public Boolean visitTypeVar(TypeVar t, Void s) {
                // type variables cannot be denoted in code model types
                return false;
            }

            @Override
            public Boolean visitWildcardType(WildcardType t, Void s) {
                // wildcards cannot de denoted in code model types
                return false;
            }

            @Override
            public Boolean visitArrayType(ArrayType t, Void s) {
                // propagate into element type
                return visit(t.elemtype, s);
            }
        };

    }

    /**
     * An exception thrown when an unsupported AST node is found when building a method IR.
     */
    static class UnsupportedASTException extends RuntimeException {

        private static final long serialVersionUID = 0;
        transient final JCTree tree;

        public UnsupportedASTException(JCTree tree) {
            this.tree = tree;
        }
    }

    enum FunctionalExpressionKind {
        QUOTED_STRUCTURAL(true), // this is transitional
        QUOTABLE(true),
        NOT_QUOTED(false);

        final boolean isQuoted;

        FunctionalExpressionKind(boolean isQuoted) {
            this.isQuoted = isQuoted;
        }
    }

    FunctionalExpressionKind functionalKind(JCFunctionalExpression functionalExpression) {
        if (functionalExpression.target.hasTag(TypeTag.METHOD)) {
            return FunctionalExpressionKind.QUOTED_STRUCTURAL;
        } else if (types.asSuper(functionalExpression.target, syms.quotableType.tsym) != null) {
            return FunctionalExpressionKind.QUOTABLE;
        } else {
            return FunctionalExpressionKind.NOT_QUOTED;
        }
    }

    /*
     * Converts a method reference which cannot be used directly into a lambda.
     * This code has been derived from LambdaToMethod::MemberReferenceToLambda. The main
     * difference is that, while that code concerns with translation strategy, boxing
     * conversion and type erasure, this version does not and, as such, can remain
     * at a higher level. Note that this code needs to create a synthetic variable
     * declaration in case of a bounded method reference whose receiver expression
     * is other than 'this'/'super' (this is done to prevent the receiver expression
     * from being computed twice).
     */
    private class MemberReferenceToLambda {

        private final JCMemberReference tree;
        private final Symbol owner;
        private final ListBuffer<JCExpression> args = new ListBuffer<>();
        private final ListBuffer<JCVariableDecl> params = new ListBuffer<>();
        private JCVariableDecl receiverVar = null;

        MemberReferenceToLambda(JCMemberReference tree, Symbol currentClass) {
            this.tree = tree;
            this.owner = new MethodSymbol(0, names.lambda, tree.target, currentClass);
            if (tree.kind == ReferenceKind.BOUND && !isThisOrSuper(tree.getQualifierExpression())) {
                // true bound method reference, hoist receiver expression out
                Type recvType = types.asSuper(tree.getQualifierExpression().type, tree.sym.owner);
                VarSymbol vsym = makeSyntheticVar("rec$", recvType);
                receiverVar = make.VarDef(vsym, tree.getQualifierExpression());
            }
        }

        JCVariableDecl receiverVar() {
            return receiverVar;
        }

        JCLambda lambda() {
            int prevPos = make.pos;
            try {
                make.at(tree);

                //body generation - this can be either a method call or a
                //new instance creation expression, depending on the member reference kind
                VarSymbol rcvr = addParametersReturnReceiver();
                JCExpression expr = (tree.getMode() == ReferenceMode.INVOKE)
                        ? expressionInvoke(rcvr)
                        : expressionNew();

                JCLambda slam = make.Lambda(params.toList(), expr);
                slam.target = tree.target;
                slam.type = tree.type;
                slam.pos = tree.pos;
                return slam;
            } finally {
                make.at(prevPos);
            }
        }

        /**
         * Generate the parameter list for the converted member reference.
         *
         * @return The receiver variable symbol, if any
         */
        VarSymbol addParametersReturnReceiver() {
            com.sun.tools.javac.util.List<Type> descPTypes = tree.getDescriptorType(types).getParameterTypes();
            VarSymbol receiverParam = null;
            switch (tree.kind) {
                case BOUND:
                    if (receiverVar != null) {
                        receiverParam = receiverVar.sym;
                    }
                    break;
                case UNBOUND:
                    // The receiver is the first parameter, extract it and
                    // adjust the SAM and unerased type lists accordingly
                    receiverParam = addParameter("rec$", descPTypes.head, false);
                    descPTypes = descPTypes.tail;
                    break;
            }
            for (int i = 0; descPTypes.nonEmpty(); ++i) {
                // By default use the implementation method parameter type
                Type parmType = descPTypes.head;
                addParameter("x$" + i, parmType, true);

                // Advance to the next parameter
                descPTypes = descPTypes.tail;
            }

            return receiverParam;
        }

        /**
         * determine the receiver of the method call - the receiver can
         * be a type qualifier, the synthetic receiver parameter or 'super'.
         */
        private JCExpression expressionInvoke(VarSymbol receiverParam) {
            JCExpression qualifier = receiverParam != null ?
                    make.at(tree.pos).Ident(receiverParam) :
                    tree.getQualifierExpression();

            //create the qualifier expression
            JCFieldAccess select = make.Select(qualifier, tree.sym.name);
            select.sym = tree.sym;
            select.type = tree.referentType;

            //create the method call expression
            JCMethodInvocation apply = make.Apply(com.sun.tools.javac.util.List.nil(), select, args.toList()).
                    setType(tree.referentType.getReturnType());

            apply.varargsElement = tree.varargsElement;
            return apply;
        }

        /**
         * Lambda body to use for a 'new'.
         */
        private JCExpression expressionNew() {
            Type expectedType = tree.referentType.getReturnType().hasTag(TypeTag.VOID) ?
                    tree.expr.type : tree.referentType.getReturnType();
            if (tree.kind == ReferenceKind.ARRAY_CTOR) {
                //create the array creation expression
                JCNewArray newArr = make.NewArray(
                        make.Type(types.elemtype(expectedType)),
                        com.sun.tools.javac.util.List.of(make.Ident(params.first())),
                        null);
                newArr.type = tree.getQualifierExpression().type;
                return newArr;
            } else {
                //create the instance creation expression
                //note that method reference syntax does not allow an explicit
                //enclosing class (so the enclosing class is null)
                // but this may need to be patched up later with the proxy for the outer this
                JCExpression newType = make.Type(types.erasure(expectedType));
                if (expectedType.tsym.type.getTypeArguments().nonEmpty()) {
                    newType = make.TypeApply(newType, com.sun.tools.javac.util.List.nil());
                }
                JCNewClass newClass = make.NewClass(null,
                        com.sun.tools.javac.util.List.nil(),
                        newType,
                        args.toList(),
                        null);
                newClass.constructor = tree.sym;
                newClass.constructorType = tree.referentType;
                newClass.type = expectedType;
                newClass.varargsElement = tree.varargsElement;
                return newClass;
            }
        }

        private VarSymbol makeSyntheticVar(String name, Type type) {
            VarSymbol vsym = new VarSymbol(PARAMETER | SYNTHETIC, names.fromString(name), type, owner);
            vsym.pos = tree.pos;
            return vsym;
        }

        private VarSymbol addParameter(String name, Type type, boolean genArg) {
            VarSymbol vsym = makeSyntheticVar(name, type);
            params.append(make.VarDef(vsym, null));
            if (genArg) {
                args.append(make.Ident(vsym));
            }
            return vsym;
        }

        boolean isThisOrSuper(JCExpression expression) {
            return TreeInfo.isThisQualifier(expression) || TreeInfo.isSuperQualifier(tree);
        }
    }
}
