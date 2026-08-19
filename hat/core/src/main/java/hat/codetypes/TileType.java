package hat.codetypes;

import jdk.incubator.code.CodeType;

public sealed interface TileType extends CodeType
        permits ConstantType, IndexType,
        PtrType, ShapeType, TensorType {
}
