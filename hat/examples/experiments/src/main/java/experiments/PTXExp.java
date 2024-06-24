package experiments;

import hat.buffer.Buffer;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.StructLayout;

import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public class PTXExp {

    interface Tree extends Buffer.StructChild{
        StructLayout layout = MemoryLayout.structLayout(
                JAVA_INT.withName("id"),
                JAVA_SHORT.withName("firstFeatureId"),
                JAVA_SHORT.withName("featureCount")
        ).withName(PTXExp.Tree.class.getSimpleName());

        void id(int id);

        void firstFeatureId(short firstFeatureId);

        void featureCount(short featureCount);

        int id();

        short firstFeatureId();

        short featureCount();

//        Schema schema = Schema.of(PTXExp.Tree.class, b->
//                b.primitive("id").primitive("featureCount").primitive("firstFeatureId")
//        );
    }

    public static void main (String[] args) {
//        Tree.schema.field.toText(0, t->System.out.print(t));
        System.out.println(Tree.layout.memberLayouts());
    }
}
