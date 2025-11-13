package view.f32.pool;

import view.f32.F32;
import view.f32.F32x2;
import view.f32.F32x2Triangle;
import view.f32.F32x3;
import view.f32.F32x3Triangle;
import view.f32.F32x4x4;

public record F32PoolBased(F32x4x4.Factory f32x4x4Factory,
                           F32x3.Factory f32x3Factory,
                           F32x2.Factory f32x2Factory,
                           F32x3Triangle.Factory f32x3TriangleFactory,
                           F32x2Triangle.Factory f32x2TriangleFactory

) implements F32 {
}
