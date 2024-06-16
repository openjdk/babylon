package hat;

import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.Value;
import java.util.List;

public class HatOps {
    public abstract sealed static class HatOp extends Op permits HatPtrOp, HatKernelContextOp{
        private final TypeElement type;

        HatOp(String opName, TypeElement type, List<Value> operands) {
            super(opName, operands);
            this.type = type;
        }

        HatOp(HatOp that, CopyContext cc) {
            super(that, cc);
            this.type = that.type;
        }

        @Override
        public TypeElement resultType() {
            return type;
        }
    }
    public final static class HatKernelContextOp extends HatOp {
        public final static String NAME="hat.kc.op";
        public HatKernelContextOp(TypeElement typeElement, List<Value> operands) {
            super(NAME, typeElement, operands);
        }

        public HatKernelContextOp(HatOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public Op transform(CopyContext cc, OpTransformer ot) {
            return new HatKernelContextOp(this, cc);
        }
    }


    public abstract static sealed class HatPtrOp extends HatOp permits HatPtrStoreOp,HatPtrLoadOp {

        public HatPtrOp(String name, TypeElement typeElement, List<Value> operands) {
            super(name, typeElement, operands);
        }

        public HatPtrOp(HatOp that, CopyContext cc) {
            super(that, cc);
        }


    }
    public final static class HatPtrStoreOp extends HatPtrOp {
        public final static String NAME="hat.ptr.store";
        public HatPtrStoreOp(TypeElement typeElement, List<Value> operands) {
            super(NAME, typeElement, operands);
        }

        public HatPtrStoreOp(HatOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public Op transform(CopyContext cc, OpTransformer ot) {
            return new HatPtrStoreOp(this, cc);
        }
    }
    public final static class HatPtrLoadOp extends HatPtrOp {
        public final static String NAME="hat.ptr.load";
        public HatPtrLoadOp(TypeElement typeElement, List<Value> operands) {
            super(NAME, typeElement, operands);
        }

        public HatPtrLoadOp(HatOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public Op transform(CopyContext cc, OpTransformer ot) {
            return new HatPtrStoreOp(this, cc);
        }
    }

}
