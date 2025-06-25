import jdk.incubator.code.Block;
import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.Quotable;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.parser.OpParser;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.lang.reflect.Method;
import java.util.Iterator;
import java.util.Map;
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
        Op lop = fm.body().entryBlock().ops().stream().filter(op -> op instanceof JavaOp.LambdaOp).findFirst().orElseThrow();

        CoreOp.FuncOp funcOp = Quoted.quoteOp(lop);

        Object[] args = new Object[]{1, "a", this};
        Quoted quoted = Quoted.quotedOp(funcOp, args);
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
        Op invOp = gm.body().entryBlock().ops().stream().filter(o -> o instanceof JavaOp.InvokeOp).findFirst().orElseThrow();

        CoreOp.FuncOp funcOp = Quoted.quoteOp(invOp);

        Object[] args = {"abc", "b"};
        Quoted quoted = Quoted.quotedOp(funcOp, args);

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
        Quoted quoted2 = Quoted.quotedOp(fop, args);

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
func @"q" ()java.type:"jdk.incubator.code.Quoted" -> {
    branch ^block_1;

  ^block_1:
    %5 : java.type:"jdk.incubator.code.Quoted" = quoted ()java.type:"void" -> {
        %6 : java.type:"java.lang.Runnable" = lambda ()java.type:"void" -> {
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
func @"q" ()java.type:"jdk.incubator.code.Quoted" -> {
    %5 : java.type:"jdk.incubator.code.Quoted" = quoted ()java.type:"void" -> {
      %6 : java.type:"java.lang.Runnable" = lambda ()java.type:"void" -> {
          return;
      };
      yield %6;
    };
    %0 : java.type:"boolean" = constant @false;
    return %5;
};
""", new Object[]{}
                },
                {
                        // last op must be ReturnOp
                        """
func @"q" ()java.type:"jdk.incubator.code.Quoted" -> {
    %5 : java.type:"jdk.incubator.code.Quoted" = quoted ()java.type:"void" -> {
      %6 : java.type:"java.lang.Runnable" = lambda ()java.type:"void" -> {
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
func @"q" ()java.type:"jdk.incubator.code.Quoted" -> {
    %5 : java.type:"jdk.incubator.code.Quoted" = quoted ()java.type:"void" -> {
      %6 : java.type:"java.lang.Runnable" = lambda ()java.type:"void" -> {
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
func @"q" ()java.type:"jdk.incubator.code.Quoted" -> {
    %0 : java.type:"int" = constant @1;
    %5 : java.type:"jdk.incubator.code.Quoted" = quoted ()java.type:"void" -> {
      %6 : java.type:"java.lang.Runnable" = lambda ()java.type:"void" -> {
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
func @"q" (%0 : java.type:"Object")java.type:"jdk.incubator.code.Quoted" -> {
    %5 : java.type:"jdk.incubator.code.Quoted" = quoted ()java.type:"void" -> {
      %6 : java.type:"java.lang.Runnable" = lambda ()java.type:"void" -> {
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
func @"q" (%0 : java.type:"int")java.type:"jdk.incubator.code.Quoted" -> {
    %2 : Var<java.type:"int"> = var %0 @"y";
    %5 : java.type:"jdk.incubator.code.Quoted" = quoted ()java.type:"void" -> {
        %6 : java.type:"java.util.function.IntSupplier" = lambda ()java.type:"int" -> {
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
func @"q" ()java.type:"jdk.incubator.code.Quoted" -> {
    %0 : java.type:"java.lang.String" = new @java.ref:"java.lang.String::()";
    %5 : java.type:"jdk.incubator.code.Quoted" = quoted ()java.type:"void" -> {
      %6 : java.type:"java.util.function.IntSupplier" = lambda ()java.type:"int" -> {
           %7 : java.type:"int" = invoke %0 @java.ref:"java.lang.String::length():int";
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
func @"q" ()java.type:"jdk.incubator.code.Quoted" -> {
    %0 : java.type:"int" = constant @1;
    %5 : java.type:"jdk.incubator.code.Quoted" = quoted ()java.type:"void" -> {
      %6 : java.type:"java.lang.Runnable" = lambda ()java.type:"void" -> {
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
func @"q" ()java.type:"jdk.incubator.code.Quoted" -> {
    %0 : java.type:"int" = constant @1;
    %1 : Var<java.type:"int"> = var %0 @"y";
    %5 : java.type:"jdk.incubator.code.Quoted" = quoted ()java.type:"void" -> {
      %6 : java.type:"java.util.function.IntSupplier" = lambda ()java.type:"int" -> {
            %7 : java.type:"int" = var.load %1;
            %8 : java.type:"int" = add %0 %7;
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
func @"q" ()java.type:"jdk.incubator.code.Quoted" -> {
    %1 : Var<java.type:"int"> = var @"y";
    %5 : java.type:"jdk.incubator.code.Quoted" = quoted ()java.type:"void" -> {
      %6 : java.type:"java.util.function.IntSupplier" = lambda ()java.type:"int" -> {
            %7 : java.type:"int" = var.load %1;
            return %7;
      };
      yield %6;
    };
    return %5;
};
""", new Object[]{}
                },
                {
                        // model must contain at least two operations
                        """
func @"q" (%5 : java.type:"jdk.incubator.code.Quoted")java.type:"jdk.incubator.code.Quoted" -> {
    return %5;
};
""", new Object[]{null}
                },
                // args length must be equal to params size
                {
                        """
func @"q" (%0 : java.type:"int")java.type:"jdk.incubator.code.Quoted" -> {
    %5 : java.type:"jdk.incubator.code.Quoted" = quoted ()java.type:"void" -> {
      %6 : java.type:"java.util.function.IntSupplier" = lambda ()java.type:"int" -> {
            return %0;
      };
      yield %6;
    };
    return %5;
};
""", new Object[]{1, 2}
                }
      };
}


    @Test(dataProvider = "invalidCases")
    void testInvalidCases(String model, Object[] args) {
        CoreOp.FuncOp fop = ((CoreOp.FuncOp) OpParser.fromStringOfJavaCodeModel(model));
        Assert.assertThrows(RuntimeException.class, () -> Quoted.quotedOp(fop, args));
    }

    @DataProvider
    Object[][] validCases() {
        return new Object[][] {
                {
                        """
func @"q" ()java.type:"jdk.incubator.code.Quoted" -> {
    %5 : java.type:"jdk.incubator.code.Quoted" = quoted ()java.type:"void" -> {
      %6 : java.type:"java.lang.Runnable" = lambda ()java.type:"void" -> {
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
func @"q" (%0 : java.type:"int")java.type:"jdk.incubator.code.Quoted" -> {
    %5 : java.type:"jdk.incubator.code.Quoted" = quoted ()java.type:"void" -> {
      %6 : java.type:"java.util.function.IntSupplier" = lambda ()java.type:"int" -> {
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
func @"q" (%0 : java.type:"int")java.type:"jdk.incubator.code.Quoted" -> {
    %1 : Var<java.type:"int"> = var %0;
    %5 : java.type:"jdk.incubator.code.Quoted" = quoted ()java.type:"void" -> {
      %6 : java.type:"java.util.function.IntSupplier" = lambda ()java.type:"int" -> {
            %7 : java.type:"int" = var.load %1;
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
func @"q" (%0 : java.type:"int")java.type:"jdk.incubator.code.Quoted" -> {
    %5 : java.type:"jdk.incubator.code.Quoted" = quoted ()java.type:"void" -> {
      %6 : java.type:"java.util.function.IntSupplier" = lambda ()java.type:"int" -> {
            %7 : java.type:"int" = add %0 %0;
            %8 : java.type:"int" = mul %0 %0;
            %9 : java.type:"int" = sub %8 %7;
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
func @"q" ()java.type:"jdk.incubator.code.Quoted" -> {
    %0 : java.type:"int" = constant @1;
    %5 : java.type:"jdk.incubator.code.Quoted" = quoted ()java.type:"void" -> {
      %6 : java.type:"java.util.function.IntSupplier" = lambda ()java.type:"int" -> {
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
func @"q" ()java.type:"jdk.incubator.code.Quoted" -> {
    %0 : java.type:"int" = constant @1;
    %5 : java.type:"jdk.incubator.code.Quoted" = quoted ()java.type:"void" -> {
      %6 : java.type:"java.util.function.IntSupplier" = lambda ()java.type:"int" -> {
            %7 : java.type:"int" = add %0 %0;
            %8 : java.type:"int" = mul %0 %0;
            %9 : java.type:"int" = sub %8 %7;
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
func @"q" ()java.type:"jdk.incubator.code.Quoted" -> {
    %0 : java.type:"int" = constant @1;
    %1 : Var<java.type:"int"> = var %0;
    %5 : java.type:"jdk.incubator.code.Quoted" = quoted ()java.type:"void" -> {
      %6 : java.type:"java.util.function.IntSupplier" = lambda ()java.type:"int" -> {
            %7 : java.type:"int" = var.load %1;
            %8 : java.type:"int" = mul %7 %7;
            return %8;
      };
      yield %6;
    };
    return %5;
};
""", new Object[] {}
                },
                {
                        """
func @"q" (%0 : java.type:"int", %2 : java.type:"int")java.type:"jdk.incubator.code.Quoted" -> {
    %1 : Var<java.type:"int"> = var %0;
    %5 : java.type:"jdk.incubator.code.Quoted" = quoted ()java.type:"void" -> {
      %6 : java.type:"java.util.function.IntSupplier" = lambda ()java.type:"int" -> {
            %7 : java.type:"int" = var.load %1;
            %8 : java.type:"int" = add %7 %2;
            return %8;
      };
      yield %6;
    };
    return %5;
};
""", new Object[]{8, 9}
                }
        };
    }

    @Test(dataProvider = "validCases")
    void testValidCases(String model, Object[] args) {
        CoreOp.FuncOp fop = ((CoreOp.FuncOp) OpParser.fromStringOfJavaCodeModel(model));
        Quoted quoted = Quoted.quotedOp(fop, args);

        for (Map.Entry<Value, Object> e : quoted.capturedValues().entrySet()) {
            Value sv = e.getKey();
            Object rv = e.getValue();
            // assert only when captured value is block param, or result of VarOp initialized with block param
            if (sv instanceof Op.Result opr && opr.op() instanceof CoreOp.VarOp vop
                    && vop.initOperand() instanceof Block.Parameter p) {
                Assert.assertEquals(((CoreOp.Var) rv).value(), args[p.index()]);
            } else if (sv instanceof Block.Parameter p) {
                Assert.assertEquals(rv, args[p.index()]);
            }
        }
    }
}
