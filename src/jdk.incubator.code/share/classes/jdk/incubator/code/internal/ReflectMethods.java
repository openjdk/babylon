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

package jdk.incubator.code.internal;

import com.sun.source.tree.LambdaExpressionTree;
import com.sun.source.tree.MemberReferenceTree.ReferenceMode;
import com.sun.tools.javac.code.Kinds.Kind;
import com.sun.tools.javac.code.Symbol;
import com.sun.tools.javac.code.Symbol.ClassSymbol;
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
import com.sun.tools.javac.comp.CaptureScanner;
import com.sun.tools.javac.comp.DeferredAttr.FilterScanner;
import com.sun.tools.javac.comp.Flow;
import com.sun.tools.javac.comp.Lower;
import com.sun.tools.javac.comp.CodeReflectionTransformer;
import com.sun.tools.javac.comp.Resolve;
import com.sun.tools.javac.comp.TypeEnvs;
import com.sun.tools.javac.jvm.ByteCodes;
import com.sun.tools.javac.jvm.Gen;
import com.sun.tools.javac.tree.JCTree;
import com.sun.tools.javac.tree.JCTree.JCAnnotation;
import com.sun.tools.javac.tree.JCTree.JCArrayAccess;
import com.sun.tools.javac.tree.JCTree.JCAssign;
import com.sun.tools.javac.tree.JCTree.JCBinary;
import com.sun.tools.javac.tree.JCTree.JCBlock;
import com.sun.tools.javac.tree.JCTree.JCClassDecl;
import com.sun.tools.javac.tree.JCTree.JCExpression;
import com.sun.tools.javac.tree.JCTree.JCFieldAccess;
import com.sun.tools.javac.tree.JCTree.JCFunctionalExpression;
import com.sun.tools.javac.tree.JCTree.JCIdent;
import com.sun.tools.javac.tree.JCTree.JCLambda;
import com.sun.tools.javac.tree.JCTree.JCLiteral;
import com.sun.tools.javac.tree.JCTree.JCMemberReference;
import com.sun.tools.javac.tree.JCTree.JCMemberReference.ReferenceKind;
import com.sun.tools.javac.tree.JCTree.JCMethodDecl;
import com.sun.tools.javac.tree.JCTree.JCMethodInvocation;
import com.sun.tools.javac.tree.JCTree.JCModuleDecl;
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
import com.sun.tools.javac.util.JCDiagnostic.Note;
import com.sun.tools.javac.util.ListBuffer;
import com.sun.tools.javac.util.Log;
import com.sun.tools.javac.util.Name;
import com.sun.tools.javac.util.Names;
import com.sun.tools.javac.util.Options;
import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.core.TupleType;
import jdk.incubator.code.dialect.core.VarType;
import jdk.incubator.code.dialect.java.*;
import jdk.incubator.code.dialect.java.WildcardType.BoundKind;
import jdk.incubator.code.writer.OpBuilder;

import javax.lang.model.element.Modifier;
import javax.tools.JavaFileObject;
import java.lang.constant.ClassDesc;
import java.util.*;
import java.util.function.Function;
import java.util.function.Supplier;

import static com.sun.tools.javac.code.Flags.NOOUTERTHIS;
import static com.sun.tools.javac.code.Flags.PARAMETER;
import static com.sun.tools.javac.code.Flags.PUBLIC;
import static com.sun.tools.javac.code.Flags.STATIC;
import static com.sun.tools.javac.code.Flags.SYNTHETIC;
import static com.sun.tools.javac.code.Kinds.Kind.MTH;
import static com.sun.tools.javac.code.Kinds.Kind.TYP;
import static com.sun.tools.javac.code.Kinds.Kind.VAR;
import static com.sun.tools.javac.code.TypeTag.BOT;
import static com.sun.tools.javac.code.TypeTag.METHOD;
import static com.sun.tools.javac.code.TypeTag.NONE;
import static com.sun.tools.javac.main.Option.G_CUSTOM;

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
    private final Resolve resolve;
    private final Gen gen;
    private final Log log;
    private final Lower lower;
    private final TypeEnvs typeEnvs;
    private final Flow flow;
    private final CodeReflectionSymbols crSyms;
    private final boolean dumpIR;
    private final boolean lineDebugInfo;
    private final CodeModelStorageOption codeModelStorageOption;

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
        lineDebugInfo =
                options.isUnset(G_CUSTOM) ||
                        options.isSet(G_CUSTOM, "lines");
        codeModelStorageOption = CodeModelStorageOption.parse(options.get("codeModelStorageOption"));
        names = Names.instance(context);
        syms = Symtab.instance(context);
        resolve = Resolve.instance(context);
        types = Types.instance(context);
        gen = Gen.instance(context);
        log = Log.instance(context);
        lower = Lower.instance(context);
        typeEnvs = TypeEnvs.instance(context);
        flow = Flow.instance(context);
        crSyms = new CodeReflectionSymbols(context);
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
        if (tree.sym.attribute(crSyms.codeReflectionType.tsym) != null) {
            // if the method is annotated, scan it
            BodyScanner bodyScanner = new BodyScanner(tree);
            try {
                CoreOp.FuncOp funcOp = bodyScanner.scanMethod();
                if (dumpIR) {
                    // dump the method IR if requested
                    log.note(MethodIrDump(tree.sym.enclClass(), tree.sym, funcOp.toText()));
                }
                // create a static method that returns the op
                classOps.add(opMethodDecl(methodName(bodyScanner.symbolToErasedMethodRef(tree.sym)), funcOp, codeModelStorageOption));
            } catch (UnsupportedASTException ex) {
                // whoops, some AST node inside the method body were not supported. Log it and move on.
                log.note(ex.tree, MethodIrSkip(tree.sym.enclClass(), tree.sym, ex.tree.getTag().toString()));
            }
        }
        super.visitMethodDef(tree);
    }

    @Override
    public void visitModuleDef(JCModuleDecl that) {
        // do nothing
    }

    @Override
    public void visitClassDef(JCClassDecl tree) {
        ListBuffer<JCTree> prevClassOps = classOps;
        Symbol.ClassSymbol prevClassSym = currentClassSym;
        int prevLambdaCount = lambdaCount;
        JavaFileObject prev = log.useSource(tree.sym.sourcefile);
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
            log.useSource(prev);
        }
    }

    @Override
    public void visitLambda(JCLambda tree) {
        FunctionalExpressionKind kind = functionalKind(tree);
        if (kind.isQuoted) {
            // quoted lambda - scan it
            BodyScanner bodyScanner = new BodyScanner(tree, kind);
            try {
                CoreOp.FuncOp funcOp = bodyScanner.scanLambda();
                if (dumpIR) {
                    // dump the method IR if requested
                    log.note(QuotedIrDump(funcOp.toText()));
                }
                // create a static method that returns the FuncOp representing the lambda
                JCMethodDecl opMethod = opMethodDecl(lambdaName(), funcOp, codeModelStorageOption);
                classOps.add(opMethod);

                switch (kind) {
                    case QUOTED_STRUCTURAL -> {
                        // @@@ Consider replacing with invokedynamic to quoted bootstrap method
                        // Thereby we avoid certain dependencies and hide specific details
                        JCIdent opMethodId = make.Ident(opMethod.sym);
                        ListBuffer<JCExpression> interpreterArgs = new ListBuffer<>();
                        // Obtain MethodHandles.lookup()
                        // @@@ Could probably use MethodHandles.publicLookup()
                        JCMethodInvocation lookup = make.App(make.Ident(crSyms.methodHandlesLookup), com.sun.tools.javac.util.List.nil());
                        interpreterArgs.append(lookup);
                        // Get the func operation
                        JCFieldAccess opFactory = make.Select(make.Ident(crSyms.javaOpType.tsym),
                                crSyms.javaOpFactorySym);
                        JCFieldAccess typeFactory = make.Select(make.Ident(crSyms.coreTypeFactoryType.tsym),
                                crSyms.coreTypeFactorySym);
                        JCMethodInvocation op = make.App(opMethodId, com.sun.tools.javac.util.List.of(opFactory, typeFactory));
                        interpreterArgs.append(op);
                        // Append captured vars
                        ListBuffer<JCExpression> capturedArgs = quotedCapturedArgs(tree, bodyScanner);
                        interpreterArgs.appendList(capturedArgs.toList());

                        // Interpret the func operation to produce the quoted instance
                        JCMethodInvocation interpreterInvoke = make.App(make.Ident(crSyms.opInterpreterInvoke), interpreterArgs.toList());
                        interpreterInvoke.varargsElement = syms.objectType;
                        super.visitLambda(tree);
                        result = interpreterInvoke;
                    }
                    case QUOTABLE -> {
                        // leave the lambda in place, but also leave a trail for LambdaToMethod
                        tree.codeModel = opMethod.sym;
                        super.visitLambda(tree);
                    }
                }
            } catch (UnsupportedASTException ex) {
                // whoops, some AST node inside the quoted lambda body were not supported. Log it and move on.
                log.note(ex.tree, QuotedIrSkip(ex.tree.getTag().toString()));
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
                CoreOp.FuncOp funcOp = bodyScanner.scanLambda();
                if (dumpIR) {
                    // dump the method IR if requested
                    log.note(QuotedIrDump(funcOp.toText()));
                }
                // create a method that returns the FuncOp representing the lambda
                JCMethodDecl opMethod = opMethodDecl(lambdaName(), funcOp, codeModelStorageOption);
                classOps.add(opMethod);
                tree.codeModel = opMethod.sym;
                super.visitReference(tree);
                if (recvDecl != null) {
                    result = copyReferenceWithReceiverVar(tree, recvDecl);
                }
            } catch (UnsupportedASTException ex) {
                // whoops, some AST node inside the quoted lambda body were not supported. Log it and move on.
                log.note(ex.tree, QuotedIrSkip(ex.tree.getTag().toString()));
                result = tree;
            }
        } else {
            super.visitReference(tree);
        }
    }

    // @@@: Only used for quoted lambda, not quotable ones. Remove?
    ListBuffer<JCExpression> quotedCapturedArgs(DiagnosticPosition pos, BodyScanner bodyScanner) {
        ListBuffer<JCExpression> capturedArgs = new ListBuffer<>();
        for (Symbol capturedSym : bodyScanner.stack.localToOp.keySet()) {
            if (capturedSym.kind == Kind.VAR) {
                // captured var
                VarSymbol var = (VarSymbol)capturedSym;
                if (var.getConstValue() == null) {
                    capturedArgs.add(make.at(pos).Ident(capturedSym));
                }
            } else {
                throw new AssertionError("Unexpected captured symbol: " + capturedSym);
            }
        }
        if (capturedArgs.size() < bodyScanner.top.body.entryBlock().parameters().size()) {
            // needs to capture 'this'
            capturedArgs.prepend(make.at(pos).This(currentClassSym.type));
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
        newRef.codeModel = ref.codeModel;
        return make.at(ref).LetExpr(recvDecl, newRef).setType(newRef.type);
    }

    Name lambdaName() {
        return names.fromString("lambda").append('$', names.fromString(String.valueOf(lambdaCount++)));
    }

    Name methodName(MethodRef method) {
        char[] sigCh = method.toString().toCharArray();
        for (int i = 0; i < sigCh.length; i++) {
            switch (sigCh[i]) {
                case '.', ';', '[', '/' -> sigCh[i] = '$';
            }
        }
        return names.fromChars(sigCh, 0, sigCh.length);
    }

    private enum CodeModelStorageOption {
        TEXT, CODE_BUILDER;

        public static CodeModelStorageOption parse(String s) {
            if (s == null) {
                return CodeModelStorageOption.CODE_BUILDER;
            }
            return CodeModelStorageOption.valueOf(s);
        }
    }

    private JCMethodDecl opMethodDecl(Name methodName, CoreOp.FuncOp op, CodeModelStorageOption codeModelStorageOption) {
        switch (codeModelStorageOption) {
            case TEXT -> {
                var paramTypes = com.sun.tools.javac.util.List.of(crSyms.opFactoryType, crSyms.typeElementFactoryType);
                var mt = new MethodType(paramTypes, crSyms.opType,
                        com.sun.tools.javac.util.List.nil(), syms.methodClass);
                var ms = new MethodSymbol(PUBLIC | STATIC | SYNTHETIC, methodName, mt, currentClassSym);
                currentClassSym.members().enter(ms);
                var opFromStr = make.App(make.Ident(crSyms.opParserFromString),
                        com.sun.tools.javac.util.List.of(make.Literal(op.toText())));
                var ret = make.Return(opFromStr);
                var md = make.MethodDef(ms, make.Block(0, com.sun.tools.javac.util.List.of(ret)));
                return md;
            }
            case CODE_BUILDER -> {
                var opBuilder = OpBuilder.createBuilderFunction(op);
                var cmToASTTransformer = new CodeModelToAST(make, names, syms, resolve, types, typeEnvs.get(currentClassSym), crSyms);
                return cmToASTTransformer.transformFuncOpToAST(opBuilder, methodName);
            }
            case null, default ->
                    throw new IllegalStateException("unknown code model storage option: " + codeModelStorageOption);
        }
    }

    public JCTree translateTopLevelClass(JCTree cdef, TreeMaker make) {
        // note that this method does NOT support recursion.
        this.make = make;
        JCTree res = translate(cdef);
        return res;
    }

    public CoreOp.FuncOp getMethodBody(Symbol.ClassSymbol classSym, JCMethodDecl methodDecl, JCBlock attributedBody, TreeMaker make) {
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
        private final boolean isQuoted;
        private Type bodyTarget;
        private JCTree currentNode;
        private Map<Symbol, List<Symbol>> localCaptures = new HashMap<>();

        // unsupported tree nodes
        private static final EnumSet<JCTree.Tag> UNSUPPORTED_TAGS = EnumSet.of(
                // the nodes below are not as relevant, either because they have already
                // been handled by an earlier compiler pass, or because they are typically
                // not handled directly, but in the context of some enclosing statement.

                // modifiers (these are already turned into symbols by Attr and should not be dealt with directly)
                Tag.ANNOTATION, Tag.TYPE_ANNOTATION, Tag.MODIFIERS,
                // toplevel (likely outside the scope for code models)
                Tag.TOPLEVEL, Tag.PACKAGEDEF, Tag.IMPORT, Tag.METHODDEF,
                // modules (likely outside the scope for code models)
                Tag.MODULEDEF, Tag.EXPORTS, Tag.OPENS, Tag.PROVIDES, Tag.REQUIRES, Tag.USES,
                // classes, ignore local class definitions (allows access to but does not model the definition)
                // Tag.CLASSDEF,
                // switch labels (these are handled by the enclosing construct, SWITCH or SWITCH_EXPRESSION)
                Tag.CASE, Tag.DEFAULTCASELABEL, Tag.CONSTANTCASELABEL, Tag.PATTERNCASELABEL,
                // patterns (these are handled by the enclosing construct, like IF, SWITCH_EXPRESSION, TYPETEST)
                Tag.ANYPATTERN, Tag.BINDINGPATTERN, Tag.RECORDPATTERN,
                // catch (already handled as part of TRY)
                Tag.CATCH,
                // types (these are used to parse types and should not be dealt with directly)
                Tag.TYPEAPPLY, Tag.TYPEUNION, Tag.TYPEINTERSECTION, Tag.TYPEPARAMETER, Tag.WILDCARD,
                Tag.TYPEBOUNDKIND, Tag.ANNOTATED_TYPE,
                // internal (these are synthetic nodes generated by javac)
                Tag.NO_TAG, Tag.ERRONEOUS, Tag.NULLCHK, Tag.LETEXPR);

        private static final Set<JCTree.Tag> SUPPORTED_TAGS = EnumSet.complementOf(UNSUPPORTED_TAGS);

        BodyScanner(JCMethodDecl tree) {
            this(tree, tree.body);
        }

        BodyScanner(JCMethodDecl tree, JCBlock body) {
            super(SUPPORTED_TAGS);

            this.currentNode = tree;
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
                Op.Result paramOp = append(CoreOp.var(
                        tree.params.get(i).name.toString(),
                        top.block.parameters().get(blockArgOffset + i)));
                top.localToOp.put(tree.params.get(i).sym, paramOp);
            }

            bodyTarget = tree.sym.type.getReturnType();
        }

        BodyScanner(JCLambda tree, FunctionalExpressionKind kind) {
            super(SUPPORTED_TAGS);
            assert kind != FunctionalExpressionKind.NOT_QUOTED;

            this.currentNode = tree;
            this.body = tree;
            this.name = names.fromString("quotedLambda");
            this.isQuoted = true;

            QuotableLambdaCaptureScanner lambdaCaptureScanner =
                    new QuotableLambdaCaptureScanner(tree);

            List<VarSymbol> capturedSymbols = lambdaCaptureScanner.analyzeCaptures();
            int blockParamOffset = 0;

            ListBuffer<Type> capturedTypes = new ListBuffer<>();
            if (lambdaCaptureScanner.capturesThis) {
                capturedTypes.add(currentClassSym.type);
                blockParamOffset++;
            }
            for (Symbol s : capturedSymbols) {
                capturedTypes.add(s.type);
            }

            MethodType mtype = new MethodType(capturedTypes.toList(), crSyms.quotedType,
                    com.sun.tools.javac.util.List.nil(), syms.methodClass);
            FunctionType mtDesc = FunctionType.functionType(typeToTypeElement(mtype.restype),
                    mtype.getParameterTypes().map(this::typeToTypeElement));

            this.stack = this.top = new BodyStack(null, tree.body, mtDesc);

            // add captured variables mappings
            for (int i = 0 ; i < capturedSymbols.size() ; i++) {
                Symbol capturedSymbol = capturedSymbols.get(i);
                var capturedArg = top.block.parameters().get(blockParamOffset + i);
                top.localToOp.put(capturedSymbol,
                        append(CoreOp.var(capturedSymbol.name.toString(), capturedArg)));
            }

            // add captured constant mappings
            for (Map.Entry<Symbol, Object> constantCapture : lambdaCaptureScanner.constantCaptures.entrySet()) {
                Symbol capturedSymbol = constantCapture.getKey();
                var capturedArg = append(CoreOp.constant(typeToTypeElement(capturedSymbol.type),
                        constantCapture.getValue()));
                top.localToOp.put(capturedSymbol,
                        append(CoreOp.var(capturedSymbol.name.toString(), capturedArg)));
            }

            bodyTarget = tree.target.getReturnType();
        }

        /**
         * Compute the set of local variables captured by a quotable lambda expression.
         * Inspired from LambdaToMethod's LambdaCaptureScanner.
         */
        class QuotableLambdaCaptureScanner extends CaptureScanner {
            boolean capturesThis;
            Set<ClassSymbol> seenClasses = new HashSet<>();
            Map<Symbol, Object> constantCaptures = new HashMap<>();

            QuotableLambdaCaptureScanner(JCLambda ownerTree) {
                super(ownerTree);
            }

            @Override
            public void visitClassDef(JCClassDecl tree) {
                seenClasses.add(tree.sym);
                super.visitClassDef(tree);
            }

            @Override
            public void visitIdent(JCIdent tree) {
                if (!tree.sym.isStatic() &&
                        tree.sym.owner.kind == TYP &&
                        (tree.sym.kind == VAR || tree.sym.kind == MTH) &&
                        !seenClasses.contains(tree.sym.owner)) {
                    // a reference to an enclosing field or method, we need to capture 'this'
                    capturesThis = true;
                } else if (tree.sym.kind == VAR && ((VarSymbol)tree.sym).getConstValue() != null) {
                    // record the constant value associated with this
                    constantCaptures.put(tree.sym, ((VarSymbol)tree.sym).getConstValue());
                } else {
                    // might be a local capture
                    super.visitIdent(tree);
                }
            }

            @Override
            public void visitSelect(JCFieldAccess tree) {
                if (tree.sym.kind == VAR &&
                        (tree.sym.name == names._this ||
                                tree.sym.name == names._super) &&
                        !seenClasses.contains(tree.sym.type.tsym)) {
                    capturesThis = true;
                }
                super.visitSelect(tree);
            }

            @Override
            public void visitNewClass(JCNewClass tree) {
                if (tree.type.tsym.owner.kind == MTH &&
                        !seenClasses.contains(tree.type.tsym)) {
                    throw unsupported(tree);
                }
                super.visitNewClass(tree);
            }

            @Override
            public void visitAnnotation(JCAnnotation tree) {
                // do nothing (annotation values look like captured instance fields)
            }
        }

        @Override
        public void scan(JCTree tree) {
            JCTree prev = currentNode;
            currentNode = tree;
            try {
                super.scan(tree);
            } finally {
                currentNode = prev;
            }
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
            throw new NoSuchElementException(sym.toString());
        }

        Value thisValue() { // @@@: outer this?
            return top.block.parameters().get(0);
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
            return append(op, generateLocation(currentNode, false), stack);
        }

        private Op.Result append(Op op, Location l) {
            return append(op, l, stack);
        }

        private Op.Result append(Op op, Location l, BodyStack stack) {
            lastOp = op;
            op.setLocation(l);
            return stack.block.apply(op);
        }

        Location generateLocation(JCTree node, boolean includeSourceReference) {
            if (!lineDebugInfo) {
                return Location.NO_LOCATION;
            }

            int pos = node.getStartPosition();
            int line = log.currentSource().getLineNumber(pos);
            int col = log.currentSource().getColumnNumber(pos, false);
            String path;
            if (includeSourceReference) {
                path = log.currentSource().getFile().toUri().toString();
            } else {
                path = null;
            }
            return new Location(path, line, col);
        }

        private void appendReturnOrUnreachable(JCTree body) {
            // Append only if an existing terminating operation is not present
            if (lastOp == null || !(lastOp instanceof Op.Terminating)) {
                // If control can continue after the body append return.
                // Otherwise, append unreachable.
                if (isAliveAfter(body)) {
                    append(CoreOp._return());
                } else {
                    append(CoreOp.unreachable());
                }
            }
        }

        private boolean isAliveAfter(JCTree node) {
            return flow.aliveAfter(typeEnvs.get(currentClassSym), node, make);
        }

        private <O extends Op & Op.Terminating> void appendTerminating(Supplier<O> sop) {
            // Append only if an existing terminating operation is not present
            if (lastOp == null || !(lastOp instanceof Op.Terminating)) {
                append(sop.get());
            }
        }

        public Value toValue(JCExpression expression, Type targetType) {
            result = null; // reset
            Type prevPt = pt;
            try {
                pt = targetType;
                scan(expression);
                return result != null ?
                        coerce(result, expression.type, targetType) :
                        null;
            } finally {
                pt = prevPt;
            }
        }

        public Value toValue(JCExpression expression) {
            return toValue(expression, Type.noType);
        }

        public Value toValue(JCTree.JCStatement statement) {
            result = null; // reset
            scan(statement);
            return result;
        }

        Value coerce(Value sourceValue, Type sourceType, Type targetType) {
            if (sourceType.isReference() && targetType.isReference() &&
                    !types.isSubtype(types.erasure(sourceType), types.erasure(targetType))) {
                return append(JavaOp.cast(typeToTypeElement(targetType), sourceValue));
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
                    return append(JavaOp.conv(typeToTypeElement(target), exprVal));
                }
            } else if (sourcePrimitive) {
                // we need to box
                Type unboxedTarget = types.unboxedType(target);
                if (!unboxedTarget.hasTag(NONE)) {
                    // non-Object target
                    if (!types.isConvertible(source, unboxedTarget)) {
                        exprVal = convert(exprVal, unboxedTarget);
                    }
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
            // Boxing is a static method e.g., java.lang.Integer::valueOf(int)java.lang.Integer
            MethodRef boxMethod = MethodRef.method(typeToTypeElement(box), names.valueOf.toString(),
                    FunctionType.functionType(typeToTypeElement(box), typeToTypeElement(types.unboxedType(box))));
            return append(JavaOp.invoke(boxMethod, valueExpr));
        }

        Value unbox(Value valueExpr, Type box, Type primitive, Type unboxedType) {
            if (unboxedType.hasTag(NONE)) {
                // Object target, first downcast to correct wrapper type
                unboxedType = primitive;
                box = types.boxedClass(unboxedType).type;
                valueExpr = append(JavaOp.cast(typeToTypeElement(box), valueExpr));
            }
            // Unboxing is a virtual method e.g., java.lang.Integer::intValue()int
            MethodRef unboxMethod = MethodRef.method(typeToTypeElement(box),
                    unboxedType.tsym.name.append(names.Value).toString(),
                    FunctionType.functionType(typeToTypeElement(unboxedType)));
            return append(JavaOp.invoke(unboxMethod, valueExpr));
        }

        @Override
        protected void skip(JCTree tree) {
            // this method is called for unsupported AST nodes (see 'SUPPORTED_TAGS')
            throw unsupported(tree);
        }

        @Override
        public void visitVarDef(JCVariableDecl tree) {
            JavaType javaType = typeToTypeElement(tree.type);
            if (tree.init != null) {
                Value initOp = toValue(tree.init, tree.type);
                result = append(CoreOp.var(tree.name.toString(), javaType, initOp));
            } else {
                // Uninitialized
                result = append(CoreOp.var(tree.name.toString(), javaType));
            }
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
                            append(CoreOp.varStore(varOp, result));
                        }
                        case FIELD -> {
                            FieldRef fd = symbolToFieldRef(sym, symbolSiteType(sym));
                            if (sym.isStatic()) {
                                append(JavaOp.fieldStore(fd, result));
                            } else {
                                append(JavaOp.fieldStore(fd, thisValue(), result));
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
                        append(JavaOp.fieldStore(fr, result));
                    } else {
                        append(JavaOp.fieldStore(fr, receiver, result));
                    }
                    break;
                }
                case INDEXED: {
                    JCArrayAccess assign = (JCArrayAccess) lhs;

                    Value array = toValue(assign.indexed);
                    Value index = toValue(assign.index);

                    // Scan the rhs, the assign expression result is its input
                    result = toValue(tree.rhs, target);

                    append(JavaOp.arrayStoreOp(array, index, result));
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
                Value rhs;
                if (tree.operator.opcode == ByteCodes.string_add && tree.rhs.type.isPrimitive()) {
                    rhs = toValue(tree.rhs);
                } else {
                    rhs = toValue(tree.rhs, unboxedType);
                }
                lhs = unboxIfNeeded(lhs);

                Value assignOpResult = switch (tree.getTag()) {

                    // Arithmetic operations
                    case PLUS_ASG -> {
                        if (tree.operator.opcode == ByteCodes.string_add) {
                            yield append(JavaOp.concat(lhs, rhs));
                        } else {
                            yield append(JavaOp.add(lhs, rhs));
                        }
                    }
                    case MINUS_ASG -> append(JavaOp.sub(lhs, rhs));
                    case MUL_ASG -> append(JavaOp.mul(lhs, rhs));
                    case DIV_ASG -> append(JavaOp.div(lhs, rhs));
                    case MOD_ASG -> append(JavaOp.mod(lhs, rhs));

                    // Bitwise operations (including their boolean variants)
                    case BITOR_ASG -> append(JavaOp.or(lhs, rhs));
                    case BITAND_ASG -> append(JavaOp.and(lhs, rhs));
                    case BITXOR_ASG -> append(JavaOp.xor(lhs, rhs));

                    // Shift operations
                    case SL_ASG -> append(JavaOp.lshl(lhs, rhs));
                    case SR_ASG -> append(JavaOp.ashr(lhs, rhs));
                    case USR_ASG -> append(JavaOp.lshr(lhs, rhs));


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

                            Op.Result lhsOpValue = append(CoreOp.varLoad(varOp));
                            // Scan the rhs
                            Value r = scanRhs.apply(lhsOpValue);

                            append(CoreOp.varStore(varOp, r));
                        }
                        case FIELD -> {
                            FieldRef fr = symbolToFieldRef(sym, symbolSiteType(sym));

                            Op.Result lhsOpValue;
                            TypeElement resultType = typeToTypeElement(sym.type);
                            if (sym.isStatic()) {
                                lhsOpValue = append(JavaOp.fieldLoad(resultType, fr));
                            } else {
                                lhsOpValue = append(JavaOp.fieldLoad(resultType, fr, thisValue()));
                            }
                            // Scan the rhs
                            Value r = scanRhs.apply(lhsOpValue);

                            if (sym.isStatic()) {
                                append(JavaOp.fieldStore(fr, r));
                            } else {
                                append(JavaOp.fieldStore(fr, thisValue(), r));
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
                        lhsOpValue = append(JavaOp.fieldLoad(resultType, fr));
                    } else {
                        lhsOpValue = append(JavaOp.fieldLoad(resultType, fr, receiver));
                    }
                    // Scan the rhs
                    Value r = scanRhs.apply(lhsOpValue);

                    if (sym.isStatic()) {
                        append(JavaOp.fieldStore(fr, r));
                    } else {
                        append(JavaOp.fieldStore(fr, receiver, r));
                    }
                }
                case INDEXED -> {
                    JCArrayAccess assign = (JCArrayAccess) lhs;

                    Value array = toValue(assign.indexed);
                    Value index = toValue(assign.index);

                    Op.Result lhsOpValue = append(JavaOp.arrayLoadOp(array, index));
                    // Scan the rhs
                    Value r = scanRhs.apply(lhsOpValue);

                    append(JavaOp.arrayStoreOp(array, index, r));
                }
                default -> throw unsupported(lhs);
            }
        }

        @Override
        public void visitIdent(JCIdent tree) {
            // Visited only for read access

            Symbol sym = tree.sym;
            switch (sym.getKind()) {
                case LOCAL_VARIABLE, RESOURCE_VARIABLE, BINDING_VARIABLE, PARAMETER, EXCEPTION_PARAMETER ->
                        result = loadVar(sym);
                case FIELD, ENUM_CONSTANT -> {
                    if (sym.name.equals(names._this) || sym.name.equals(names._super)) {
                        result = thisValue();
                    } else {
                        FieldRef fr = symbolToFieldRef(sym, symbolSiteType(sym));
                        TypeElement resultType = typeToTypeElement(sym.type);
                        if (sym.isStatic()) {
                            result = append(JavaOp.fieldLoad(resultType, fr));
                        } else {
                            result = append(JavaOp.fieldLoad(resultType, fr, thisValue()));
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

        private Value loadVar(Symbol sym) {
            Value varOp = varOpValue(sym);
            return varOp.type() instanceof VarType ?
                    append(CoreOp.varLoad(varOp)) : // regular var
                    varOp;                          // captured value
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
                result = append(CoreOp.constant(JavaType.J_L_CLASS, typeToTypeElement(tree.selected.type)));
            } else if (types.isArray(tree.selected.type)) {
                if (tree.sym.equals(syms.lengthVar)) {
                    result = append(JavaOp.arrayLength(receiver));
                } else {
                    // Should not reach here
                    throw unsupported(tree);
                }
            } else {
                Symbol sym = tree.sym;
                switch (sym.getKind()) {
                    case FIELD, ENUM_CONSTANT -> {
                        if (sym.name.equals(names._this) || sym.name.equals(names._super)) {
                            result = thisValue();
                        } else {
                            FieldRef fr = symbolToFieldRef(sym, qualifierTarget.hasTag(NONE) ?
                                    tree.selected.type : qualifierTarget);
                            TypeElement resultType = typeToTypeElement(types.memberType(tree.selected.type, sym));
                            if (sym.isStatic()) {
                                result = append(JavaOp.fieldLoad(resultType, fr));
                            } else {
                                result = append(JavaOp.fieldLoad(resultType, fr, receiver));
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
        }

        @Override
        public void visitIndexed(JCArrayAccess tree) {
            // Visited only for read access

            Value array = toValue(tree.indexed);

            Value index = toValue(tree.index, typeElementToType(JavaType.INT));

            result = append(JavaOp.arrayLoadOp(array, index));
        }

        @Override
        public void visitApply(JCTree.JCMethodInvocation tree) {
            // @@@ Symbol.externalType, for use with inner classes

            // @@@ this.xyz(...) calls in a constructor

            JCTree meth = TreeInfo.skipParens(tree.meth);
            switch (meth.getTag()) {
                case IDENT: {
                    JCIdent access = (JCIdent) meth;

                    Symbol sym = access.sym;
                    List<Value> args = new ArrayList<>();
                    JavaOp.InvokeOp.InvokeKind ik;
                    if (!sym.isStatic()) {
                        ik = JavaOp.InvokeOp.InvokeKind.INSTANCE;
                        args.add(thisValue());
                    } else {
                        ik = JavaOp.InvokeOp.InvokeKind.STATIC;
                    }

                    args.addAll(scanMethodArguments(tree.args, tree.meth.type, tree.varargsElement));

                    MethodRef mr = symbolToErasedMethodRef(sym, symbolSiteType(sym));
                    Value res = append(JavaOp.invoke(ik, tree.varargsElement != null,
                            typeToTypeElement(meth.type.getReturnType()), mr, args));
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
                    JavaOp.InvokeOp.InvokeKind ik;
                    if (!sym.isStatic()) {
                        args.add(receiver);
                        // @@@ expr.super(...) for inner class super constructor calls
                        ik = switch (access.selected) {
                            case JCIdent i when i.sym.name.equals(names._super) -> JavaOp.InvokeOp.InvokeKind.SUPER;
                            case JCFieldAccess fa when fa.sym.name.equals(names._super) -> JavaOp.InvokeOp.InvokeKind.SUPER;
                            default -> JavaOp.InvokeOp.InvokeKind.INSTANCE;
                        };
                    } else {
                        ik = JavaOp.InvokeOp.InvokeKind.STATIC;
                    }

                    args.addAll(scanMethodArguments(tree.args, tree.meth.type, tree.varargsElement));

                    MethodRef mr = symbolToErasedMethodRef(sym, qualifierTarget.hasTag(NONE) ?
                            access.selected.type : qualifierTarget);
                    JavaType returnType = typeToTypeElement(meth.type.getReturnType());
                    JavaOp.InvokeOp iop = JavaOp.invoke(ik, tree.varargsElement != null,
                            returnType, mr, args);
                    Value res = append(iop);
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
                    result = append(JavaOp.conv(typeToTypeElement(type), v));
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
                result = append(JavaOp.cast(typeToTypeElement(type), jt, v));
            }
        }

        @Override
        public void visitTypeTest(JCTree.JCInstanceOf tree) {
            Value target = toValue(tree.expr);

            if (tree.pattern.getTag() != Tag.IDENT) {
                result = scanPattern(tree.getPattern(), target);
            } else {
                result = append(JavaOp.instanceOf(typeToTypeElement(tree.pattern.type), target));
            }
        }

        Value scanPattern(JCTree.JCPattern pattern, Value target) {
            // Type of pattern
            JavaType patternType;
            if (pattern instanceof JCTree.JCBindingPattern p) {
                patternType = JavaOp.Pattern.bindingType(typeToTypeElement(p.type));
            } else if (pattern instanceof JCTree.JCRecordPattern p) {
                patternType = JavaOp.Pattern.recordType(typeToTypeElement(p.record.type));
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
                    super(Set.of(Tag.BINDINGPATTERN, Tag.RECORDPATTERN, Tag.ANYPATTERN));
                }

                @Override
                public void visitBindingPattern(JCTree.JCBindingPattern binding) {
                    JCVariableDecl var = binding.var;
                    variables.add(var);
                    boolean unnamedPatternVariable = var.name.isEmpty();
                    String bindingName = unnamedPatternVariable ? null : var.name.toString();
                    result = append(JavaOp.typePattern(typeToTypeElement(var.type), bindingName));
                }

                @Override
                public void visitRecordPattern(JCTree.JCRecordPattern record) {
                    // @@@ Is always Identifier to record?
                    // scan(record.deconstructor);

                    List<Value> nestedValues = new ArrayList<>();
                    for (JCTree.JCPattern jcPattern : record.nested) {
                        // @@@ when we support ANYPATTERN, we must add result of toValue only if it's non-null
                        // because passing null to recordPattern methods will cause an error
                        nestedValues.add(toValue(jcPattern));
                    }

                    result = append(JavaOp.recordPattern(symbolToRecordTypeRef(record.record), nestedValues));
                }

                @Override
                public void visitAnyPattern(JCTree.JCAnyPattern anyPattern) {
                    result = append(JavaOp.matchAllPattern());
                }

                Value toValue(JCTree tree) {
                    result = null;
                    scan(tree);
                    return result;
                }
            }
            // Scan pattern
            Value patternValue = new PatternScanner().toValue(pattern);
            append(CoreOp._yield(patternValue));
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
                // @@@ use uninitialized variable
                Value defaultValue = variablesStack.block.op(defaultValue(jcVar.type));
                Value init = convert(defaultValue, jcVar.type);
                Op.Result op = variablesStack.block.op(CoreOp.var(jcVar.name.toString(), typeToTypeElement(jcVar.type), init));
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
                matchBuilder.op(CoreOp.varStore(var, v));
            }
            matchBuilder.op(CoreOp._yield());

            // Create the match operation
            return append(JavaOp.match(target, patternBody, matchBody));
        }

        @Override
        public void visitNewClass(JCTree.JCNewClass tree) {
            if (tree.def != null) {
                scan(tree.def);
            }

            // @@@ Support local classes in pre-construction contexts
            if (tree.type.tsym.isDirectlyOrIndirectlyLocal() && (tree.type.tsym.flags() & NOOUTERTHIS) != 0) {
                throw unsupported(tree);
            }

            List<TypeElement> argtypes = new ArrayList<>();
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
                argtypes.add(outerInstance.type());
            }
            if (tree.type.tsym.isDirectlyOrIndirectlyLocal()) {
                for (Symbol c : localCaptures.get(tree.type.tsym)) {
                    args.add(loadVar(c));
                    argtypes.add(symbolToErasedDesc(c));
                }
            }

            // Create erased method type reference for constructor, where
            // the return type declares the class to instantiate
            // We need to manually construct the constructor reference,
            // as the signature of the constructor symbol is not augmented
            // with enclosing this and captured params.
            MethodRef methodRef = symbolToErasedMethodRef(tree.constructor);
            argtypes.addAll(methodRef.type().parameterTypes());
            FunctionType constructorType = FunctionType.functionType(
                    symbolToErasedDesc(tree.constructor.owner),
                    argtypes);
            ConstructorRef constructorRef = ConstructorRef.constructor(constructorType);

            args.addAll(scanMethodArguments(tree.args, tree.constructorType, tree.varargsElement));

            result = append(JavaOp._new(tree.varargsElement != null, typeToTypeElement(type), constructorRef, args));
        }

        @Override
        public void visitNewArray(JCTree.JCNewArray tree) {
            if (tree.elems != null) {
                int length = tree.elems.size();
                Op.Result a = append(JavaOp.newArray(
                        typeToTypeElement(tree.type),
                        append(CoreOp.constant(JavaType.INT, length))));
                int i = 0;
                for (JCExpression elem : tree.elems) {
                    Value element = toValue(elem, types.elemtype(tree.type));
                    append(JavaOp.arrayStoreOp(
                            a,
                            append(CoreOp.constant(JavaType.INT, i)),
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
                ConstructorRef constructorRef = ConstructorRef.constructor(arrayType,
                        indexes.stream().map(Value::type).toList());
                result = append(JavaOp._new(constructorRef, indexes));
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
                Op.Result paramOp = append(CoreOp.var(
                        p.name.toString(),
                        stack.block.parameters().get(i)));
                stack.localToOp.put(p.sym, paramOp);
            }

            // Scan the lambda body
            if (tree.getBodyKind() == LambdaExpressionTree.BodyKind.EXPRESSION) {
                Value exprVal = toValue(((JCExpression) tree.body), tree.getDescriptorType(types).getReturnType());
                if (!tree.body.type.hasTag(TypeTag.VOID)) {
                    append(CoreOp._return(exprVal));
                } else {
                    appendTerminating(CoreOp::_return);
                }
            } else {
                Type prevBodyTarget = bodyTarget;
                try {
                    bodyTarget = tree.getDescriptorType(types).getReturnType();
                    toValue(((JCTree.JCStatement) tree.body));
                    appendReturnOrUnreachable(tree.body);
                } finally {
                    bodyTarget = prevBodyTarget;
                }
            }

            Op lambdaOp = switch (kind) {
                case QUOTED_STRUCTURAL -> {
                    yield CoreOp.closure(stack.body);
                }
                case QUOTABLE, NOT_QUOTED -> {
                    // Get the functional interface type
                    JavaType fiType = typeToTypeElement(tree.target);
                    // build functional lambda
                    yield JavaOp.lambda(fiType, stack.body);
                }
            };

            // Pop lambda body
            popBody();

            Value lambdaResult;
            if (isQuoted) {
                lambdaResult = append(lambdaOp, generateLocation(tree, true));
            } else {
                lambdaResult = append(lambdaOp);
            }

            if (isQuoted || kind == FunctionalExpressionKind.QUOTED_STRUCTURAL) {
                append(CoreOp._yield(lambdaResult));
                CoreOp.QuotedOp quotedOp = CoreOp.quoted(stack.body);

                // Pop quoted body
                popBody();

                lambdaResult = append(quotedOp);
            }

            result = lambdaResult;
        }

        @Override
        public void visitIf(JCTree.JCIf tree) {
            List<Body.Builder> bodies = new ArrayList<>();

            while (tree != null) {
                JCTree.JCExpression cond = TreeInfo.skipParens(tree.cond);

                // Push if condition
                pushBody(cond,
                        FunctionType.functionType(JavaType.BOOLEAN));
                Value last = toValue(cond);
                last = convert(last, typeElementToType(JavaType.BOOLEAN));
                // Yield the boolean result of the condition
                append(CoreOp._yield(last));
                bodies.add(stack.body);

                // Pop if condition
                popBody();

                // Push if body
                pushBody(tree.thenpart, FunctionType.VOID);

                scan(tree.thenpart);
                appendTerminating(CoreOp::_yield);
                bodies.add(stack.body);

                // Pop if body
                popBody();

                JCTree.JCStatement elsepart = tree.elsepart;
                if (elsepart == null) {
                    tree = null;
                } else if (elsepart.getTag() == Tag.IF) {
                    tree = (JCTree.JCIf) elsepart;
                } else {
                    // Push else body
                    pushBody(elsepart, FunctionType.VOID);

                    scan(elsepart);
                    appendTerminating(CoreOp::_yield);
                    bodies.add(stack.body);

                    // Pop else body
                    popBody();

                    tree = null;
                }
            }

            append(JavaOp._if(bodies));
            result = null;
        }

        @Override
        public void visitSwitchExpression(JCTree.JCSwitchExpression tree) {
            Value target = toValue(tree.selector);

            Type switchType = adaptBottom(tree.type);
            FunctionType caseBodyType = FunctionType.functionType(typeToTypeElement(switchType));

            List<Body.Builder> bodies = visitSwitchStatAndExpr(tree, tree.selector, target, tree.cases, caseBodyType,
                    !tree.hasUnconditionalPattern);

            result = append(JavaOp.switchExpression(caseBodyType.returnType(), target, bodies));
        }

        @Override
        public void visitSwitch(JCTree.JCSwitch tree) {
            Value target = toValue(tree.selector);

            FunctionType actionType = FunctionType.VOID;

            List<Body.Builder> bodies = visitSwitchStatAndExpr(tree, tree.selector, target, tree.cases, actionType,
                    tree.patternSwitch && !tree.hasUnconditionalPattern);

            result = append(JavaOp.switchStatement(target, bodies));
        }

        private List<Body.Builder> visitSwitchStatAndExpr(JCTree tree, JCExpression selector, Value target,
                                                          List<JCTree.JCCase> cases, FunctionType caseBodyType,
                                                          boolean isDefaultCaseNeeded) {
            List<Body.Builder> bodies = new ArrayList<>();
            Body.Builder defaultLabel = null;
            Body.Builder defaultBody = null;

            for (JCTree.JCCase c : cases) {
                Body.Builder caseLabel = visitCaseLabel(tree, selector, target, c);
                Body.Builder caseBody = visitCaseBody(tree, c, caseBodyType);

                if (c.labels.head instanceof JCTree.JCDefaultCaseLabel) {
                    defaultLabel = caseLabel;
                    defaultBody = caseBody;
                } else {
                    bodies.add(caseLabel);
                    bodies.add(caseBody);
                }
            }

            if (defaultLabel != null) {
                bodies.add(defaultLabel);
                bodies.add(defaultBody);
            } else if (isDefaultCaseNeeded) {
                // label
                pushBody(tree, FunctionType.functionType(JavaType.BOOLEAN));
                append(CoreOp._yield(append(CoreOp.constant(JavaType.BOOLEAN, true))));
                bodies.add(stack.body);
                popBody();

                // body
                pushBody(tree, caseBodyType);
                append(JavaOp._throw(
                        append(JavaOp._new(ConstructorRef.constructor(MatchException.class)))
                ));
                bodies.add(stack.body);
                popBody();
            }

            return bodies;
        }

        private Body.Builder visitCaseLabel(JCTree tree, JCExpression selector, Value target, JCTree.JCCase c) {
            Body.Builder body;
            FunctionType caseLabelType = FunctionType.functionType(JavaType.BOOLEAN, target.type());

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
                    append(CoreOp._yield(patVal));
                    clBodies.add(stack.body);
                    popBody();

                    pushBody(c.guard, FunctionType.functionType(JavaType.BOOLEAN));
                    append(CoreOp._yield(toValue(c.guard)));
                    clBodies.add(stack.body);
                    popBody();

                    localResult = append(JavaOp.conditionalAnd(clBodies));
                } else {
                    localResult = scanPattern(pcl.pat, localTarget);
                }
                // Yield the boolean result of the condition
                append(CoreOp._yield(localResult));
                body = stack.body;

                // Pop label
                popBody();
            } else if (headCl instanceof JCTree.JCConstantCaseLabel ccl) {
                pushBody(headCl, caseLabelType);

                Value localTarget = stack.block.parameters().get(0);
                final Value localResult;
                if (c.labels.size() == 1) {
                    Value expr = toValue(ccl.expr);
                    // per java spec, constant type is compatible with the type of the selector expression
                    // so, we convert constant to the type of the selector expression
                    expr = convert(expr, selector.type);
                    if (selector.type.isPrimitive()) {
                        localResult = append(JavaOp.eq(localTarget, expr));
                    } else {
                        localResult = append(JavaOp.invoke(
                                MethodRef.method(Objects.class, "equals", boolean.class, Object.class, Object.class),
                                localTarget, expr));
                    }
                } else {
                    List<Body.Builder> clBodies = new ArrayList<>();
                    for (JCTree.JCCaseLabel cl : c.labels) {
                        ccl = (JCTree.JCConstantCaseLabel) cl;
                        pushBody(ccl, FunctionType.functionType(JavaType.BOOLEAN));

                        Value expr = toValue(ccl.expr);
                        expr = convert(expr, selector.type);
                        final Value labelResult;
                        if (selector.type.isPrimitive()) {
                            labelResult = append(JavaOp.eq(localTarget, expr));
                        } else {
                            labelResult = append(JavaOp.invoke(
                                    MethodRef.method(Objects.class, "equals", boolean.class, Object.class, Object.class),
                                    localTarget, expr));
                        }

                        append(CoreOp._yield(labelResult));
                        clBodies.add(stack.body);

                        // Pop label
                        popBody();
                    }

                    localResult = append(JavaOp.conditionalOr(clBodies));
                }

                append(CoreOp._yield(localResult));
                body = stack.body;

                // Pop labels
                popBody();
            } else if (headCl instanceof JCTree.JCDefaultCaseLabel) {
                // @@@ Do we need to model the default label body?
                pushBody(headCl, FunctionType.functionType(JavaType.BOOLEAN));

                append(CoreOp._yield(append(CoreOp.constant(JavaType.BOOLEAN, true))));
                body = stack.body;

                // Pop label
                popBody();
            } else {
                throw unsupported(tree);
            }

            return body;
        }

        private Body.Builder visitCaseBody(JCTree tree, JCTree.JCCase c, FunctionType caseBodyType) {
            Body.Builder body = null;
            Type yieldType = tree.type != null ? adaptBottom(tree.type) : null;

            JCTree.JCCaseLabel headCl = c.labels.head;
            switch (c.caseKind) {
                case RULE -> {
                    pushBody(c.body, caseBodyType);

                    if (c.body instanceof JCTree.JCExpression e) {
                        Value bodyVal = toValue(e, yieldType);
                        append(CoreOp._yield(bodyVal));
                    } else if (c.body instanceof JCTree.JCStatement s){ // this includes Block
                        // Otherwise there is a yield statement
                        Type prevBodyTarget = bodyTarget;
                        try {
                            bodyTarget = yieldType;
                            toValue(s);
                        } finally {
                            bodyTarget = prevBodyTarget;
                        }
                        appendTerminating(c.completesNormally ? CoreOp::_yield : CoreOp::unreachable);
                    }
                    body = stack.body;

                    // Pop block
                    popBody();
                }
                case STATEMENT -> {
                    // @@@ Avoid nesting for a single block? Goes against "say what you see"
                    // boolean oneBlock = c.stats.size() == 1 && c.stats.head instanceof JCBlock;
                    pushBody(c, caseBodyType);

                    scan(c.stats);

                    appendTerminating(c.completesNormally ?
                            headCl instanceof JCTree.JCDefaultCaseLabel ? CoreOp::_yield : JavaOp::switchFallthroughOp
                            : CoreOp::unreachable);

                    body = stack.body;

                    // Pop block
                    popBody();
                }
            }
            return body;
        }

        @Override
        public void visitYield(JCTree.JCYield tree) {
            Value retVal = toValue(tree.value, bodyTarget);
            if (retVal == null) {
                result = append(JavaOp.java_yield());
            } else {
                result = append(JavaOp.java_yield(retVal));
            }
        }

        @Override
        public void visitWhileLoop(JCTree.JCWhileLoop tree) {
            // @@@ Patterns
            JCTree.JCExpression cond = TreeInfo.skipParens(tree.cond);

            // Push while condition
            pushBody(cond, FunctionType.functionType(JavaType.BOOLEAN));
            Value last = toValue(cond);
            // Yield the boolean result of the condition
            last = convert(last, typeElementToType(JavaType.BOOLEAN));
            append(CoreOp._yield(last));
            Body.Builder condition = stack.body;

            // Pop while condition
            popBody();

            // Push while body
            pushBody(tree.body, FunctionType.VOID);
            scan(tree.body);
            appendTerminating(JavaOp::_continue);
            Body.Builder body = stack.body;

            // Pop while body
            popBody();

            append(JavaOp._while(condition, body));
            result = null;
        }

        @Override
        public void visitDoLoop(JCTree.JCDoWhileLoop tree) {
            // @@@ Patterns
            JCTree.JCExpression cond = TreeInfo.skipParens(tree.cond);

            // Push while body
            pushBody(tree.body, FunctionType.VOID);
            scan(tree.body);
            appendTerminating(JavaOp::_continue);
            Body.Builder body = stack.body;

            // Pop while body
            popBody();

            // Push while condition
            pushBody(cond, FunctionType.functionType(JavaType.BOOLEAN));
            Value last = toValue(cond);
            last = convert(last, typeElementToType(JavaType.BOOLEAN));
            // Yield the boolean result of the condition
            append(CoreOp._yield(last));
            Body.Builder condition = stack.body;

            // Pop while condition
            popBody();

            append(JavaOp.doWhile(body, condition));
            result = null;
        }

        @Override
        public void visitForeachLoop(JCTree.JCEnhancedForLoop tree) {
            // Push expression
            pushBody(tree.expr, FunctionType.functionType(typeToTypeElement(tree.expr.type)));
            Value last = toValue(tree.expr);
            // Yield the Iterable result of the expression
            append(CoreOp._yield(last));
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
            Op.Result varEResult = append(CoreOp.var(var.name.toString(), stack.block.parameters().get(0)));
            append(CoreOp._yield(varEResult));
            Body.Builder init = stack.body;
            // Pop init
            popBody();

            // Push body
            pushBody(tree.body, FunctionType.functionType(JavaType.VOID, varEType));
            stack.localToOp.put(var.sym, stack.block.parameters().get(0));

            scan(tree.body);
            appendTerminating(JavaOp::_continue);
            Body.Builder body = stack.body;
            // Pop body
            popBody();

            append(JavaOp.enhancedFor(expression, init, body));
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
                append(CoreOp._yield(append(CoreOp.tuple(vds.varValues()))));
            } else if (varTypes.size() == 1) {
                pushBody(null, FunctionType.functionType(varTypes.get(0)));
                scan(tree.init);

                append(CoreOp._yield(vds.varValues().get(0)));
            } else {
                pushBody(null, FunctionType.VOID);
                scan(tree.init);

                append(CoreOp._yield());
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
                append(CoreOp._yield(last));
            } else {
                append(CoreOp._yield(append(CoreOp.constant(JavaType.BOOLEAN, true))));
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
            append(CoreOp._yield());
            Body.Builder update = stack.body;

            // Pop update
            popBody();

            // Push body
            pushBody(tree.body, FunctionType.functionType(JavaType.VOID, varTypes));
            if (tree.body != null) {
                vds.mapVarsToBlockArguments();

                scan(tree.body);
            }
            appendTerminating(JavaOp::_continue);
            Body.Builder body = stack.body;

            // Pop update
            popBody();

            append(JavaOp._for(init, cond, update, body));
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
            append(CoreOp._yield(condVal));
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
            append(CoreOp._yield(trueVal));
            bodies.add(stack.body);

            // Pop true body
            popBody();

            JCTree.JCExpression falsepart = TreeInfo.skipParens(tree.falsepart);

            // Push false body
            pushBody(falsepart,
                    FunctionType.functionType(typeToTypeElement(condType)));

            Value falseVal = toValue(falsepart, condType);
            // Yield the result
            append(CoreOp._yield(falseVal));
            bodies.add(stack.body);

            // Pop false body
            popBody();

            result = append(JavaOp.conditionalExpression(typeToTypeElement(condType), bodies));
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
            append(CoreOp._yield(condVal));
            bodies.add(stack.body);

            // Pop condition
            popBody();

            if (tree.detail != null) {
                JCTree.JCExpression detail = TreeInfo.skipParens(tree.detail);

                pushBody(detail,
                        FunctionType.functionType(typeToTypeElement(tree.detail.type)));
                Value detailVal = toValue(detail);

                append(CoreOp._yield(detailVal));
                bodies.add(stack.body);

                //Pop detail
                popBody();
            }

            result = append(JavaOp._assert(bodies));

        }

        @Override
        public void visitBlock(JCTree.JCBlock tree) {
            if (stack.tree == tree) {
                // Block is associated with the visit of a parent structure
                scan(tree.stats);
            } else {
                // Otherwise, independent block structure
                // Push block
                pushBody(tree, FunctionType.VOID);
                scan(tree.stats);
                appendTerminating(CoreOp::_yield);
                Body.Builder body = stack.body;

                // Pop block
                popBody();

                append(JavaOp.block(body));
            }
            result = null;
        }

        @Override
        public void visitSynchronized(JCTree.JCSynchronized tree) {
            // Push expr
            pushBody(tree.lock, FunctionType.functionType(typeToTypeElement(tree.lock.type)));
            Value last = toValue(tree.lock);
            append(CoreOp._yield(last));
            Body.Builder expr = stack.body;

            // Pop expr
            popBody();

            // Push body block
            pushBody(tree.body, FunctionType.VOID);
            // Scan body block statements
            scan(tree.body.stats);
            appendTerminating(CoreOp::_yield);
            Body.Builder blockBody = stack.body;

            // Pop body block
            popBody();

            append(JavaOp.synchronized_(expr, blockBody));
        }

        @Override
        public void visitLabelled(JCTree.JCLabeledStatement tree) {
            // Push block
            pushBody(tree, FunctionType.VOID);
            // Create constant for label
            String labelName = tree.label.toString();
            Op.Result label = append(CoreOp.constant(JavaType.J_L_STRING, labelName));
            // Set label on body stack
            stack.setLabel(labelName, label);
            scan(tree.body);
            appendTerminating(CoreOp::_yield);
            Body.Builder body = stack.body;

            // Pop block
            popBody();

            result = append(JavaOp.labeled(body));
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
                    if (resource instanceof JCTree.JCExpression e) {
                        rValues.add(toValue(e));
                    } else if (resource instanceof JCTree.JCStatement s) {
                        rValues.add(toValue(s));
                    }
                }

                append(CoreOp._yield(append(CoreOp.tuple(rValues))));
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
            appendTerminating(CoreOp::_yield);
            Body.Builder body = stack.body;

            // Pop block
            popBody();

            List<Body.Builder> catchers = new ArrayList<>();
            for (JCTree.JCCatch catcher : tree.catchers) {
                // Push body
                pushBody(catcher.body, FunctionType.functionType(JavaType.VOID, typeToTypeElement(catcher.param.type)));
                Op.Result exVariable = append(CoreOp.var(
                        catcher.param.name.toString(),
                        stack.block.parameters().get(0)));
                stack.localToOp.put(catcher.param.sym, exVariable);
                scan(catcher.body);
                appendTerminating(CoreOp::_yield);
                catchers.add(stack.body);

                // Pop block
                popBody();
            }

            Body.Builder finalizer;
            if (tree.finalizer != null) {
                // Push body
                pushBody(tree.finalizer, FunctionType.VOID);
                scan(tree.finalizer);
                appendTerminating(CoreOp::_yield);
                finalizer = stack.body;

                // Pop block
                popBody();
            }
            else {
                finalizer = null;
            }

            result = append(JavaOp._try(resources, body, catchers, finalizer));
        }

        @Override
        public void visitUnary(JCTree.JCUnary tree) {
            Tag tag = tree.getTag();
            switch (tag) {
                case POSTINC, POSTDEC, PREINC, PREDEC -> {
                    // Capture applying rhs and operation
                    Function<Value, Value> scanRhs = (lhs) -> {
                        Type unboxedType = types.unboxedTypeOrType(tree.type);
                        Value one = convert(append(numericOneValue(unboxedType)), unboxedType);
                        Value unboxedLhs = unboxIfNeeded(lhs);

                        Value unboxedLhsPlusOne = switch (tree.getTag()) {
                            // Arithmetic operations
                            case POSTINC, PREINC -> append(JavaOp.add(unboxedLhs, one));
                            case POSTDEC, PREDEC -> append(JavaOp.sub(unboxedLhs, one));

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
                    Value rhs = toValue(tree.arg, tree.type);
                    result = append(JavaOp.neg(rhs));
                }
                case NOT -> {
                    Value rhs = toValue(tree.arg, tree.type);
                    result = append(JavaOp.not(rhs));
                }
                case COMPL -> {
                    Value rhs = toValue(tree.arg, tree.type);
                    result = append(JavaOp.compl(rhs));
                }
                case POS -> {
                    // Result is value of the operand
                    result = toValue(tree.arg, tree.type);
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
                append(CoreOp._yield(lhs));
                Body.Builder bodyLhs = stack.body;

                // Pop lhs
                popBody();

                // Push rhs
                pushBody(tree.rhs, FunctionType.functionType(JavaType.BOOLEAN));
                Value rhs = toValue(tree.rhs);
                // Yield the boolean result of the condition
                append(CoreOp._yield(rhs));
                Body.Builder bodyRhs = stack.body;

                // Pop lhs
                popBody();

                List<Body.Builder> bodies = List.of(bodyLhs, bodyRhs);
                result = append(tag == Tag.AND
                        ? JavaOp.conditionalAnd(bodies)
                        : JavaOp.conditionalOr(bodies));
            } else if (tag == Tag.PLUS && tree.operator.opcode == ByteCodes.string_add) {
                //Ignore the operator and query both subexpressions for their type with concats
                Type lhsType = tree.lhs.type;
                Type rhsType = tree.rhs.type;

                Value lhs = toValue(tree.lhs, lhsType);
                Value rhs = toValue(tree.rhs, rhsType);

                result = append(JavaOp.concat(lhs, rhs));
            }
            else {
                Type opType = tree.operator.type.getParameterTypes().getFirst();
                // @@@ potentially handle shift input conversion like other binary ops
                boolean isShift = tag == Tag.SL || tag == Tag.SR || tag == Tag.USR;
                Value lhs = toValue(tree.lhs, opType);
                Value rhs = toValue(tree.rhs, isShift ? tree.operator.type.getParameterTypes().getLast() : opType);

                result = switch (tag) {
                    // Arithmetic operations
                    case PLUS -> append(JavaOp.add(lhs, rhs));
                    case MINUS -> append(JavaOp.sub(lhs, rhs));
                    case MUL -> append(JavaOp.mul(lhs, rhs));
                    case DIV -> append(JavaOp.div(lhs, rhs));
                    case MOD -> append(JavaOp.mod(lhs, rhs));

                    // Test operations
                    case EQ -> append(JavaOp.eq(lhs, rhs));
                    case NE -> append(JavaOp.neq(lhs, rhs));
                    //
                    case LT -> append(JavaOp.lt(lhs, rhs));
                    case LE -> append(JavaOp.le(lhs, rhs));
                    case GT -> append(JavaOp.gt(lhs, rhs));
                    case GE -> append(JavaOp.ge(lhs, rhs));

                    // Bitwise operations (including their boolean variants)
                    case BITOR -> append(JavaOp.or(lhs, rhs));
                    case BITAND -> append(JavaOp.and(lhs, rhs));
                    case BITXOR -> append(JavaOp.xor(lhs, rhs));

                    // Shift operations
                    case SL -> append(JavaOp.lshl(lhs, rhs));
                    case SR -> append(JavaOp.ashr(lhs, rhs));
                    case USR -> append(JavaOp.lshr(lhs, rhs));

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
            result = append(CoreOp.constant(typeToTypeElement(constantType), value));
        }

        @Override
        public void visitReturn(JCReturn tree) {
            Value retVal = toValue(tree.expr, bodyTarget);
            if (retVal == null) {
                result = append(CoreOp._return());
            } else {
                result = append(CoreOp._return(retVal));
            }
        }

        @Override
        public void visitThrow(JCTree.JCThrow tree) {
            Value throwVal = toValue(tree.expr);
            result = append(JavaOp._throw(throwVal));
        }

        @Override
        public void visitBreak(JCTree.JCBreak tree) {
            Value label = tree.label != null
                    ? getLabel(tree.label.toString())
                    : null;
            result = append(JavaOp._break(label));
        }

        @Override
        public void visitContinue(JCTree.JCContinue tree) {
            Value label = tree.label != null
                    ? getLabel(tree.label.toString())
                    : null;
            result = append(JavaOp._continue(label));
        }

        @Override
        public void visitClassDef(JCClassDecl tree) {
            if (tree.sym.isDirectlyOrIndirectlyLocal()) {
                // we need to keep track of captured locals using same strategy as Lower
                class FreeVarScanner extends Lower.FreeVarCollector {
                    FreeVarScanner() {
                        lower.super(tree);
                    }

                    @Override
                    protected void addFreeVars(ClassSymbol c) {
                        localCaptures.getOrDefault(c, List.of())
                                .forEach(s -> addFreeVar((VarSymbol)s));
                    }
                }
                FreeVarScanner fvs = new FreeVarScanner();
                localCaptures.put(tree.sym, List.copyOf(fvs.analyzeCaptures()));
            }
        }

        UnsupportedASTException unsupported(JCTree tree) {
            return new UnsupportedASTException(tree);
        }

        CoreOp.FuncOp scanMethod() {
            scan(body);
            appendReturnOrUnreachable(body);
            CoreOp.FuncOp func = CoreOp.func(name.toString(), stack.body);
            func.setLocation(generateLocation(currentNode, true));
            return func;
        }

        CoreOp.FuncOp scanLambda() {
            scan(body);
            // Return the quoted result
            append(CoreOp._return(result));
            return CoreOp.func(name.toString(), stack.body);
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
                    Type et = ((ArrayType)t).elemtype;
                    yield JavaType.array(typeToTypeElement(et));
                }
                case WILDCARD -> {
                    Type.WildcardType wt = (Type.WildcardType)t;
                    yield wt.isUnbound() ?
                            JavaType.wildcard() :
                            JavaType.wildcard(wt.isExtendsBound() ? BoundKind.EXTENDS : BoundKind.SUPER, typeToTypeElement(wt.type));
                }
                case TYPEVAR -> t.tsym.owner.kind == Kind.MTH ?
                        JavaType.typeVarRef(t.tsym.name.toString(), symbolToErasedMethodRef(t.tsym.owner),
                                typeToTypeElement(t.getUpperBound())) :
                        JavaType.typeVarRef(t.tsym.name.toString(),
                                (jdk.incubator.code.dialect.java.ClassType)symbolToErasedDesc(t.tsym.owner),
                                typeToTypeElement(t.getUpperBound()));
                case CLASS -> {
                    Assert.check(!t.isIntersection() && !t.isUnion());
                    JavaType typ;
                    if (t.getEnclosingType() != Type.noType) {
                        Name innerName = t.tsym.flatName().subName(t.getEnclosingType().tsym.flatName().length() + 1);
                        typ = JavaType.qualified(typeToTypeElement(t.getEnclosingType()), innerName.toString());
                    } else {
                        typ = JavaType.type(ClassDesc.of(t.tsym.flatName().toString()));
                    }

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
                    yield JavaType.parameterized(typ, typeArguments);
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
            return symbolToErasedFieldRef(gen.binaryQualifier(s, types.erasure(site)));
        }

        FieldRef symbolToErasedFieldRef(Symbol s) {
            Type erasedType = s.erasure(types);
            return FieldRef.field(
                    typeToTypeElement(s.owner.erasure(types)),
                    s.name.toString(),
                    typeToTypeElement(erasedType));
        }

        MethodRef symbolToErasedMethodRef(Symbol s, Type site) {
            // @@@ Made Gen::binaryQualifier public, duplicate logic?
            // Ensure correct qualifying class is used in the reference, see JLS 13.1
            // https://docs.oracle.com/javase/specs/jls/se20/html/jls-13.html#jls-13.1
            return symbolToErasedMethodRef(gen.binaryQualifier(s, types.erasure(site)));
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
                case BYTE, SHORT, INT -> CoreOp.constant(JavaType.INT, 0);
                case CHAR -> CoreOp.constant(typeToTypeElement(t), (char)0);
                case BOOLEAN -> CoreOp.constant(typeToTypeElement(t), false);
                case FLOAT -> CoreOp.constant(typeToTypeElement(t), 0f);
                case LONG -> CoreOp.constant(typeToTypeElement(t), 0L);
                case DOUBLE -> CoreOp.constant(typeToTypeElement(t), 0d);
                default -> CoreOp.constant(typeToTypeElement(t), null);
            };
        }

        Op numericOneValue(Type t) {
            return switch (t.getTag()) {
                case BYTE, SHORT, INT -> CoreOp.constant(JavaType.INT, 1);
                case CHAR -> CoreOp.constant(typeToTypeElement(t), (char)1);
                case FLOAT -> CoreOp.constant(typeToTypeElement(t), 1f);
                case LONG -> CoreOp.constant(typeToTypeElement(t), 1L);
                case DOUBLE -> CoreOp.constant(typeToTypeElement(t), 1d);
                default -> throw new UnsupportedOperationException(t.toString());
            };
        }

        Type normalizeType(Type t) {
            Assert.check(!t.hasTag(METHOD));
            return types.upward(t, false, types.captures(t));
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
        } else if (types.asSuper(functionalExpression.target, crSyms.quotableType.tsym) != null) {
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

    /**
     * compiler.note.quoted.ir.dump=\
     *    code reflection enabled for method quoted lambda\n\
     *    {0}
     */
    public static Note QuotedIrDump(String arg0) {
        return new Note("compiler", "quoted.ir.dump", arg0);
    }

    /**
     * compiler.note.quoted.ir.skip=\
     *    unsupported code reflection node {0} found in quoted lambda
     */
    public static Note QuotedIrSkip(String arg0) {
        return new Note("compiler", "quoted.ir.skip", arg0);
    }

    /**
     * compiler.note.method.ir.dump=\
     *    code reflection enabled for method {0}.{1}\n\
     *    {2}
     */
    public static Note MethodIrDump(Symbol arg0, Symbol arg1, String arg2) {
        return new Note("compiler", "method.ir.dump", arg0, arg1, arg2);
    }

    /**
     * compiler.note.method.ir.skip=\
     *    unsupported code reflection node {2} found in method {0}.{1}
     */
    public static Note MethodIrSkip(Symbol arg0, Symbol arg1, String arg2) {
        return new Note("compiler", "method.ir.skip", arg0, arg1, arg2);
    }

    public static class Provider implements CodeReflectionTransformer {
        @Override
        public JCTree translateTopLevelClass(Context context, JCTree tree, TreeMaker make) {
            return ReflectMethods.instance(context).translateTopLevelClass(tree, make);
        }
    }
}
