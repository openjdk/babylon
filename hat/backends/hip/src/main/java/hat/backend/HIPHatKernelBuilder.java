package hat.backend;

import hat.backend.c99codebuilders.C99HatBuildContext;
import hat.backend.c99codebuilders.C99HatKernelBuilder;
import hat.optools.OpWrapper;

import jdk.incubator.code.Op;
import jdk.incubator.code.type.JavaType;

public class HIPHatKernelBuilder extends C99HatKernelBuilder<HIPHatKernelBuilder> {

    @Override
    public HIPHatKernelBuilder defines() {
        return this
                .hashDefine("NDRANGE_HIP")
                .hashDefine("__global")
                .hashDefine("NULL", "nullptr");
    }

    @Override
    public HIPHatKernelBuilder pragmas() {
        return self();
    }

    public HIPHatKernelBuilder globalId() {
        return identifier("blockIdx").dot().identifier("x")
                .asterisk()
                .identifier("blockDim").dot().identifier("x")
                .plus()
                .identifier("threadIdx").dot().identifier("x");
    }

    @Override
    public HIPHatKernelBuilder globalSize() {
        return identifier("gridDim").dot().identifier("x")
                .asterisk()
                .identifier("blockDim").dot().identifier("x");
    }


    @Override
    public HIPHatKernelBuilder kernelDeclaration(String name) {
        return externC().space().keyword("__global__").space().voidType().space().identifier(name);
    }

    @Override
    public HIPHatKernelBuilder functionDeclaration(JavaType javaType, String name) {
        return externC().space().keyword("__device__").space().keyword("inline").space().type(javaType).space().identifier(name);
    }

    @Override
    public HIPHatKernelBuilder globalPtrPrefix() {
        return self();
    }


    @Override
    public HIPHatKernelBuilder atomicInc(C99HatBuildContext buildContext, Op.Result instanceResult, String name){
        return identifier("atomicAdd").paren(_ -> {
             ampersand().recurse(buildContext, OpWrapper.wrap(instanceResult.op()));
             rarrow().identifier(name).comma().literal(1);
        });
    }
}
