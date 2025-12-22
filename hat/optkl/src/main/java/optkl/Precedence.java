package optkl;

import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

public interface Precedence {
    interface LoadOrConv extends Precedence {}
    interface Multiplicative extends Precedence{};
    interface Additive extends Precedence{}
    interface Store extends Precedence{}

    static boolean needsParenthesis(Op parent, Op child) {
        return precedenceOf(parent) <= precedenceOf(child);
    }

    static int precedenceOf(Op op) {
        return switch (op) {
            case Precedence.LoadOrConv _,
                    CoreOp.YieldOp _,
                 JavaOp.InvokeOp _,
                 CoreOp.FuncCallOp _ ,
                 JavaOp.FieldAccessOp _,
                 CoreOp.VarAccessOp.VarLoadOp _,
                 CoreOp.ConstantOp _,
                 JavaOp.LambdaOp _,
                 CoreOp.TupleOp _,
                 JavaOp.WhileOp _
                    -> 0;   //  ()[ ] .
            case JavaOp.ConvOp _,
                 JavaOp.NegOp  _
                    -> 1; //  ++ --+ -! ~ (type) *(deref) &(addressof) sizeof
            case Precedence.Multiplicative _,
                    JavaOp.ModOp _,
                 JavaOp.MulOp _,
                 JavaOp.DivOp _,
                 JavaOp.NotOp _
                    -> 2; //  * / %
            case Precedence.Additive _,
                 JavaOp.AddOp _,
                 JavaOp.SubOp _
                    -> 3; //  = + -
            case JavaOp.AshrOp _,
                 JavaOp.LshlOp _,
                 JavaOp.LshrOp _
                    -> 4; // 4 = << >>
            case JavaOp.LtOp _,
                 JavaOp.GtOp _,
                 JavaOp.LeOp _,
                 JavaOp.GeOp _
                    -> 5; //  < <= > >=
            case JavaOp.EqOp _,
                 JavaOp.NeqOp _
                    -> 6;  // == !=
            case JavaOp.AndOp _
                    -> 7; //  &
            case JavaOp.XorOp _
                    -> 8; // ^
            case JavaOp.OrOp _
                    -> 9; // |
            case JavaOp.ConditionalAndOp _
                    -> 10;//&&
            case JavaOp.ConditionalOrOp _
                    -> 11;// ||
            case JavaOp.ConditionalExpressionOp _
                    -> 12;// ()?:
            case Precedence.Store _,
                 CoreOp.VarOp _,
                 CoreOp.VarAccessOp.VarStoreOp _
                    -> 13;  // = += -= *= /= %= &= ^= |= <<= >>=
            case CoreOp.ReturnOp _-> 14;
            default -> throw new IllegalStateException("[Illegal] Precedence Op not registered: " + op.getClass().getSimpleName());

        };
    }

}
