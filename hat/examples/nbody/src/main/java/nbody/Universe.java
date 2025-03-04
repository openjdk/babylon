package nbody;

import hat.Accelerator;
import hat.buffer.Buffer;
import hat.ifacemapper.Schema;

public interface Universe extends Buffer {
    int length();

    interface Body extends Struct {
        float x();

        float y();

        float z();

        float vx();

        float vy();

        float vz();

        void x(float x);

        void y(float y);

        void z(float z);

        void vx(float vx);

        void vy(float vy);

        void vz(float vz);
    }

    Body body(long idx);

    /*
    typedef struct Body_s{
        float x;
        float y;
        float y;
        float vx;
        float vy;
        float y;
    } Body_t;

    typedef struct Universe_s{
       int length;
       Body_t body[1];
    }Universe_t;

     */
    Schema<Universe> schema = Schema.of(Universe.class, resultTable -> resultTable

            .arrayLen("length").array("body", array -> array
                    .fields("x", "y", "z", "vx", "vy", "vz")
            )
    );

    static Universe create(Accelerator accelerator, int length) {
        return schema.allocate(accelerator, length);
    }

}
