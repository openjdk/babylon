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
import java.util.Iterator;
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

        CoreOp.FuncOp funcOp = CoreOp.quoteOp(lop);

        Object[] args = new Object[]{1, "a", this};
        Quoted quoted = CoreOp.quotedOp(funcOp, args);
        // op must have the same structure as lop
        // for the moment, we don't have utility to check that

        Assert.assertTrue(lop.getClass().isInstance(quoted.op()));

        Iterator<Object> iterator = quoted.capturedValues().values().iterator();

        Assert.assertEquals(((CoreOp.Var) iterator.next()).value(), args[0]);
        Assert.assertEquals(((CoreOp.Var) iterator.next()).value(), args[1]);
        Assert.assertEquals(iterator.next(), args[2]);
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

        CoreOp.FuncOp funcOp = CoreOp.quoteOp(invOp);

        Object[] args = {"abc", "b"};
        Quoted quoted = CoreOp.quotedOp(funcOp, args);

        Assert.assertTrue(invOp.getClass().isInstance(quoted.op()));

        Iterator<Object> iterator = quoted.capturedValues().values().iterator();

        Assert.assertEquals(iterator.next(), args[0]);
        Assert.assertEquals(iterator.next(), args[1]);
    }

    @Test
    void testWithJavacModel() {
        final int y = 88;
        int z = 99;
        Quotable q = (IntUnaryOperator & Quotable) x -> x + y + z + hashCode();

        // access FuncOp created by javac
        Quoted quoted = Op.ofQuotable(q).orElseThrow();
        Op op = quoted.op();
        CoreOp.QuotedOp qop = ((CoreOp.QuotedOp) op.ancestorBody().parentOp());
        CoreOp.FuncOp fop = ((CoreOp.FuncOp) qop.ancestorBody().parentOp());

        Object[] args = {this, 111};
        Quoted quoted2 = CoreOp.quotedOp(fop, args);

        Iterator<Object> iterator = quoted2.capturedValues().values().iterator();

        Assert.assertEquals(((CoreOp.Var) iterator.next()).value(), y);
        Assert.assertEquals(((CoreOp.Var) iterator.next()).value(), args[1]);
        Assert.assertEquals(iterator.next(), args[0]);
    }

    @DataProvider
    Object[][] invalidCases() {
        return new Object[][]{
              // TODO describe error in a comment
                {
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
""", new Object[]{}
                },
                {
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
""", new Object[]{}
                },
                {
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
""", new Object[]{}
                },
                {
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
""", new Object[]{}
                },
                {
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
""", new Object[]{}
                },
                {
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
""", new Object[]{"s"}
                },
                {
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
""", new Object[]{1}
                },
                {
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
""", new Object[]{}
                },
                {
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
""", new Object[]{}
                },
                {
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
""", new Object[]{}
                },
                {
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
""", new Object[]{}
                },
                {
                        // model must contains at least two operations
                        """
func @"q" (%5 : jdk.incubator.code.Quoted)jdk.incubator.code.Quoted -> {
    return %5;
};
""", new Object[]{null}
                }
      };
}


    @Test(dataProvider = "invalidCases")
    void testInvalidCases(String model, Object[] args) {
        CoreOp.FuncOp fop = ((CoreOp.FuncOp) OpParser.fromStringOfFuncOp(model));
        Assert.assertThrows(IllegalArgumentException.class, () -> CoreOp.quotedOp(fop, args));
    }

    @DataProvider
    Object[][] validCases() {
        return new Object[][] {
                {
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
""", new Object[] {}
                },
                {
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
""", new Object[] {1}
                },
                {
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
""", new Object[] {2}
                },
                {
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
""", new Object[] {3}
                },
                {
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
""", new Object[] {}
                },
                {
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
""", new Object[] {}
                },
                {
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
""", new Object[] {}
                },
        };
    }

    @Test(dataProvider = "validCases")
    void testValidCases(String model, Object[] args) {
        CoreOp.FuncOp fop = ((CoreOp.FuncOp) OpParser.fromStringOfFuncOp(model));
        CoreOp.quotedOp(fop, args);
    }
}
