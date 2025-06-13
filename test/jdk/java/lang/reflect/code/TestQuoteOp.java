import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.Quotable;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.Value;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.parser.OpParser;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.lang.reflect.Method;
import java.util.LinkedHashSet;
import java.util.SequencedSet;
import java.util.function.IntUnaryOperator;

/*
 * @test
 * @modules jdk.incubator.code
 * @run testng TestQuoteOp
 */
public class TestQuoteOp {

    @CodeReflection
    public void f(int i) {
        String s = "abc";
        Runnable r = () -> {
            System.out.println(i + s + hashCode());
        };
    }

    @Test
    void testQuoteOpThatHasCaptures() throws NoSuchMethodException {
        Method f = getClass().getDeclaredMethod("f", int.class);
        CoreOp.FuncOp fm = Op.ofMethod(f).orElseThrow();
        Op lop = fm.body().entryBlock().ops().stream().filter(op -> op instanceof CoreOp.LambdaOp).findFirst().orElseThrow();

//        fm.writeTo(System.out);

        CoreOp.FuncOp funcOp = CoreOp.quoteOp(lop);
//        funcOp.writeTo(System.out);

        CoreOp.OpAndValues opAndValues = CoreOp.quotedOp(funcOp);
        // op must have the same structure as lop
        // for the moment, we don't have utility to check that
        Op op = opAndValues.op();

        Assert.assertTrue(lop.getClass().isInstance(op));

        SequencedSet<Value> e = new LinkedHashSet<>();
        e.addAll(op.operands());
        e.addAll(op.capturedValues());
        Assert.assertEquals(opAndValues.operandsAndCaptures(), e);
    }

    @CodeReflection
    static void g(String s) {
        boolean b = s.startsWith("a");
    }

    @Test
    void testQuoteOpThatHasOperands() throws NoSuchMethodException { // op with operands
        Method g = getClass().getDeclaredMethod("g", String.class);
        CoreOp.FuncOp gm = Op.ofMethod(g).orElseThrow();
        Op invOp = gm.body().entryBlock().ops().stream().filter(o -> o instanceof CoreOp.InvokeOp).findFirst().orElseThrow();

//        gm.writeTo(System.out);

        CoreOp.FuncOp funcOp = CoreOp.quoteOp(invOp);
//        funcOp.writeTo(System.out);

        CoreOp.OpAndValues opAndValues = CoreOp.quotedOp(funcOp);
        Op op = opAndValues.op();

        Assert.assertTrue(invOp.getClass().isInstance(op));

        SequencedSet<Value> e = new LinkedHashSet<>();
        e.addAll(op.operands());
        e.addAll(op.capturedValues());
        Assert.assertEquals(opAndValues.operandsAndCaptures(), e);
    }

    @Test
    void testWithJavacModel() {
        final int y = 88;
        int z = 99;
        Quotable q = (IntUnaryOperator & Quotable) x -> x + y + z + hashCode();

        Quoted quoted = Op.ofQuotable(q).orElseThrow();
        Op op = quoted.op();
        CoreOp.QuotedOp qop = ((CoreOp.QuotedOp) op.ancestorBody().parentOp());
        CoreOp.FuncOp fop = ((CoreOp.FuncOp) qop.ancestorBody().parentOp());
//        fop.writeTo(System.out);

        CoreOp.OpAndValues opAndValues = CoreOp.quotedOp(fop);

        SequencedSet<Value> e = new LinkedHashSet<>();
        e.addAll(op.operands());
        e.addAll(op.capturedValues());
        Assert.assertEquals(opAndValues.operandsAndCaptures(), e);
    }

    @DataProvider
    Object[] invalidCases() {
      return new Object[] {
              // TODO describe error in a comment
              // func op must have one block
              """
func @"q" ()jdk.incubator.code.Quoted -> {
    branch ^block_1;

  ^block_1:
    %5 : jdk.incubator.code.Quoted = quoted ()void -> {
        %6 : java.lang.Runnable = lambda ()void -> {
            return;
        };
        yield %6;
    };
    return %5;
};
""",
              // before last op must be QuotedOp
              """
func @"q" ()jdk.incubator.code.Quoted -> {
    %5 : jdk.incubator.code.Quoted = quoted ()void -> {
      %6 : java.lang.Runnable = lambda ()void -> {
          return;
      };
      yield %6;
    };
    %0 : boolean = constant @"false";
    return %5;
};
""",
              // last op must be ReturnOp
              """
func @"q" ()jdk.incubator.code.Quoted -> {
    %5 : jdk.incubator.code.Quoted = quoted ()void -> {
      %6 : java.lang.Runnable = lambda ()void -> {
          return;
      };
      yield %6;
    };
    yield %5;
};
""",
              // the result of QuotedOp must be returned
              """
func @"q" ()jdk.incubator.code.Quoted -> {
    %5 : jdk.incubator.code.Quoted = quoted ()void -> {
      %6 : java.lang.Runnable = lambda ()void -> {
          return;
      };
      yield %6;
    };
    return;
};
""",
              // the result of QuotedOp must be returned
              """
func @"q" ()jdk.incubator.code.Quoted -> {
    %0 : int = constant @"1";
    %5 : jdk.incubator.code.Quoted = quoted ()void -> {
      %6 : java.lang.Runnable = lambda ()void -> {
          return;
      };
      yield %6;
    };
    return %0;
};
""",
              // param must be used
              """
func @"q" (%0 : Object)jdk.incubator.code.Quoted -> {
    %5 : jdk.incubator.code.Quoted = quoted ()void -> {
      %6 : java.lang.Runnable = lambda ()void -> {
          return;
      };
      yield %6;
    };
    return %5;
};
""",
              // param used more than once, all uses must be as operand or capture of quoted op
              """
func @"q" (%0 : int)jdk.incubator.code.Quoted -> {
    %2 : Var<int> = var %0 @"y";
    %5 : jdk.incubator.code.Quoted = quoted ()void -> {
        %6 : java.util.function.IntSupplier = lambda ()int -> {
            return %0;
        };
        yield %6;
    };
    return %5;
};
""",
              // operations before quoted op must be ConstantOp or VarOp
              """
func @"q" ()jdk.incubator.code.Quoted -> {
    %0 : java.lang.String = new @"java.lang.String::<new>()";
    %5 : jdk.incubator.code.Quoted = quoted ()void -> {
      %6 : java.util.function.IntSupplier = lambda ()int -> {
           %7 : int = invoke %0 @"java.lang.String::length()int";
           return %7;
      };
      yield %6;
    };
    return %5;
};
""",
              // constant op must be used
              """
func @"q" ()jdk.incubator.code.Quoted -> {
    %0 : int = constant @"1";
    %5 : jdk.incubator.code.Quoted = quoted ()void -> {
      %6 : java.lang.Runnable = lambda ()void -> {
          return;
      };
      yield %6;
    };
    return %5;
};
""",
              // constant used more than once, all its uses must be as operand or capture of quoted op
              """
func @"q" ()jdk.incubator.code.Quoted -> {
    %0 : int = constant @"1";
    %1 : Var<int> = var %0 @"y";
    %5 : jdk.incubator.code.Quoted = quoted ()void -> {
      %6 : java.util.function.IntSupplier = lambda ()int -> {
            %7 : int = var.load %1;
            %8 : int = add %0 %7;
            return %8;
      };
      yield %6;
    };
    return %5;
};
""",
              // var op must be initialized with param or result of constant op
              """
func @"q" ()jdk.incubator.code.Quoted -> {
    %1 : Var<int> = var @"y";
    %5 : jdk.incubator.code.Quoted = quoted ()void -> {
      %6 : java.util.function.IntSupplier = lambda ()int -> {
            %7 : int = var.load %1;
            return %7;
      };
      yield %6;
    };
    return %5;
};
""",
              // var op must be used as operand or capture of quoted op
              """
func @"q" ()jdk.incubator.code.Quoted -> {
    %0 : int = constant @"1";
    %1 : Var<int> = var %0 @"y";
    %5 : jdk.incubator.code.Quoted = quoted ()void -> {
      %6 : java.lang.Runnable = lambda ()void -> {
          return;
      };
      yield %6;
    };
    return %5;
};
"""
      };
}


    @Test(dataProvider = "invalidCases")
    void testInvalidCases(String model) {
        CoreOp.FuncOp fop = ((CoreOp.FuncOp) OpParser.fromStringOfFuncOp(model));
        Assert.assertThrows(IllegalArgumentException.class, () -> CoreOp.quotedOp(fop));
    }

    @DataProvider
    Object[] validCases() {
        return new Object[] {
                """
func @"q" ()jdk.incubator.code.Quoted -> {
    %5 : jdk.incubator.code.Quoted = quoted ()void -> {
      %6 : java.lang.Runnable = lambda ()void -> {
          return;
      };
      yield %6;
    };
    return %5;
};
""",
                """
func @"q" (%0 : int)jdk.incubator.code.Quoted -> {
    %5 : jdk.incubator.code.Quoted = quoted ()void -> {
      %6 : java.util.function.IntSupplier = lambda ()int -> {
            return %0;
      };
      yield %6;
    };
    return %5;
};
""",
                """
func @"q" (%0 : int)jdk.incubator.code.Quoted -> {
    %1 : Var<int> = var %0;
    %5 : jdk.incubator.code.Quoted = quoted ()void -> {
      %6 : java.util.function.IntSupplier = lambda ()int -> {
            %7 : int = var.load %1;
            return %7;
      };
      yield %6;
    };
    return %5;
};
""",
                """
func @"q" (%0 : int)jdk.incubator.code.Quoted -> {
    %5 : jdk.incubator.code.Quoted = quoted ()void -> {
      %6 : java.util.function.IntSupplier = lambda ()int -> {
            %7 : int = add %0 %0;
            %8 : int = mul %0 %0;
            %9 : int = sub %8 %7;
            return %9;
      };
      yield %6;
    };
    return %5;
};
""",
                """
func @"q" ()jdk.incubator.code.Quoted -> {
    %0 : int = constant @"1";
    %5 : jdk.incubator.code.Quoted = quoted ()void -> {
      %6 : java.util.function.IntSupplier = lambda ()int -> {
            return %0;
      };
      yield %6;
    };
    return %5;
};
""",
                """
func @"q" ()jdk.incubator.code.Quoted -> {
    %0 : int = constant @"1";
    %5 : jdk.incubator.code.Quoted = quoted ()void -> {
      %6 : java.util.function.IntSupplier = lambda ()int -> {
            %7 : int = add %0 %0;
            %8 : int = mul %0 %0;
            %9 : int = sub %8 %7;
            return %9;
      };
      yield %6;
    };
    return %5;
};
""",
                """
func @"q" ()jdk.incubator.code.Quoted -> {
    %0 : int = constant @"1";
    %1 : Var<int> = var %0;
    %5 : jdk.incubator.code.Quoted = quoted ()void -> {
      %6 : java.util.function.IntSupplier = lambda ()int -> {
            %7 : int = var.load %1;
            %8 : int = mul %7 %7;
            return %8;
      };
      yield %6;
    };
    return %5;
};
""",
        };
    }

    @Test(dataProvider = "validCases")
    void testValidCases(String model) {
        CoreOp.FuncOp fop = ((CoreOp.FuncOp) OpParser.fromStringOfFuncOp(model));
        CoreOp.quotedOp(fop);
    }
}
