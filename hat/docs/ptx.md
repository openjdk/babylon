Here is the Java source code for a kernel
```Java
public static void mandel(KernelContext kc, S32Array2D s32Array2D, S32Array pallette, float offsetx, float offsety, float scale) {
        if (kc.x < kc.maxX) {
            float width = s32Array2D.width();
            float height = s32Array2D.height();
            float x = ((kc.x % s32Array2D.width()) * scale - (scale / 2f * width)) / width + offsetx;
            float y = ((kc.x / s32Array2D.width()) * scale - (scale / 2f * height)) / height + offsety;
            float zx = x;
            float zy = y;
            float new_zx;
            int colorIdx = 0;
            while ((colorIdx < pallette.length()) && (((zx * zx) + (zy * zy)) < 4f)) {
                new_zx = ((zx * zx) - (zy * zy)) + x;
                zy = (2f * zx * zy) + y;
                zx = new_zx;
                colorIdx++;
            }
            int color = colorIdx < pallette.length() ? pallette.array(colorIdx) : 0;
            s32Array2D.array(kc.x, color);
        }
    }

```

And here is the babylon code model

```
func @"mandel" @loc="39:5:file:/Users/grfrost/orahub/hat/examples/mandel/src/java/mandel/MandelCompute.java"
 (%0 : hat.KernelContext, %1 : hat.buffer.S32Array2D, %2 : hat.buffer.S32Array, %3 : float, %4 : float, %5 : float)void -> {
    %6 : Var<hat.KernelContext> = var %0 @"kc" ;
    %7 : Var<hat.buffer.S32Array2D> = var %1 @"s32Array2D" ;
    %8 : Var<hat.buffer.S32Array> = var %2 @"pallette" ;
    %9 : Var<float> = var %3 @"offsetx" ;
    %10 : Var<float> = var %4 @"offsety" ;
    %11 : Var<float> = var %5 @"scale" ;
    java.if
        ()boolean -> {
            %12 : hat.KernelContext = var.load %6 ;
            %13 : int = field.load %12 @"hat.KernelContext::x()int" ;
            %14 : hat.KernelContext = var.load %6 ;
            %15 : int = field.load %14 @"hat.KernelContext::maxX()int" ;
            %16 : boolean = lt %13 %15 ;
            yield %16 ;
        }
        ()void -> {
            %17 : hat.buffer.S32Array2D = var.load %7 ;
            %18 : int = invoke %17 @"hat.buffer.S32Array2D::width()int" ;
            %19 : float = conv %18 ;
            %20 : Var<float> = var %19 @"width" ;
            %21 : hat.buffer.S32Array2D = var.load %7 ;
            %22 : int = invoke %21 @"hat.buffer.S32Array2D::height()int" ;
            %23 : float = conv %22 ;
            %24 : Var<float> = var %23 @"height" ;
            %25 : hat.KernelContext = var.load %6 ;
            %26 : int = field.load %25 @"hat.KernelContext::x()int" ;
            %27 : hat.buffer.S32Array2D = var.load %7 ;
            %28 : int = invoke %27 @"hat.buffer.S32Array2D::width()int" ;
            %29 : int = mod %26 %28 ;
            %30 : float = conv %29 ;
            %31 : float = var.load %11 ;
            %32 : float = mul %30 %31 ;
            %33 : float = var.load %11 ;
            %34 : float = constant @"2.0" ;
            %35 : float = div %33 %34 ;
            %36 : float = var.load %20 ;
            %37 : float = mul %35 %36 ;
            %38 : float = sub %32 %37 ;
            %39 : float = var.load %20 ;
            %40 : float = div %38 %39 ;
            %41 : float = var.load %9 ;
            %42 : float = add %40 %41 ;
            %43 : Var<float> = var %42 @"x" ;
            %44 : hat.KernelContext = var.load %6 ;
            %45 : int = field.load %44 @"hat.KernelContext::x()int" ;
            %46 : hat.buffer.S32Array2D = var.load %7 ;
            %47 : int = invoke %46 @"hat.buffer.S32Array2D::width()int" ;
            %48 : int = div %45 %47 ;
            %49 : float = conv %48 ;
            %50 : float = var.load %11 ;
            %51 : float = mul %49 %50 ;
            %52 : float = var.load %11 ;
            %53 : float = constant @"2.0" ;
            %54 : float = div %52 %53 ;
            %55 : float = var.load %24 ;
            %56 : float = mul %54 %55 ;
            %57 : float = sub %51 %56 ;
            %58 : float = var.load %24 ;
            %59 : float = div %57 %58 ;
            %60 : float = var.load %10 ;
            %61 : float = add %59 %60 ;
            %62 : Var<float> = var %61 @"y" ;
            %63 : float = var.load %43 ;
            %64 : Var<float> = var %63 @"zx" ;
            %65 : float = var.load %62 ;
            %66 : Var<float> = var %65 @"zy" ;
            %67 : float = constant @"0.0" ;
            %68 : Var<float> = var %67 @"new_zx" ;
            %69 : int = constant @"0" ;
            %70 : Var<int> = var %69 @"colorIdx" ;
            java.while
                ()boolean -> {
                    %71 : boolean = java.cand
                        ()boolean -> {
                            %72 : int = var.load %70 ;
                            %73 : hat.buffer.S32Array = var.load %8 ;
                            %74 : int = invoke %73 @"hat.buffer.S32Array::length()int" ;
                            %75 : boolean = lt %72 %74 ;
                            yield %75 ;
                        }
                        ()boolean -> {
                            %76 : float = var.load %64 ;
                            %77 : float = var.load %64 ;
                            %78 : float = mul %76 %77 ;
                            %79 : float = var.load %66 ;
                            %80 : float = var.load %66 ;
                            %81 : float = mul %79 %80 ;
                            %82 : float = add %78 %81 ;
                            %83 : float = constant @"4.0" ;
                            %84 : boolean = lt %82 %83 ;
                            yield %84 ;
                        };
                    yield %71 ;
                }
                ()void -> {
                    %85 : float = var.load %64 ;
                    %86 : float = var.load %64 ;
                    %87 : float = mul %85 %86 ;
                    %88 : float = var.load %66 ;
                    %89 : float = var.load %66 ;
                    %90 : float = mul %88 %89 ;
                    %91 : float = sub %87 %90 ;
                    %92 : float = var.load %43 ;
                    %93 : float = add %91 %92 ;
                    var.store %68 %93 ;
                    %94 : float = constant @"2.0" ;
                    %95 : float = var.load %64 ;
                    %96 : float = mul %94 %95 ;
                    %97 : float = var.load %66 ;
                    %98 : float = mul %96 %97 ;
                    %99 : float = var.load %62 ;
                    %100 : float = add %98 %99 ;
                    var.store %66 %100 ;
                    %101 : float = var.load %68 ;
                    var.store %64 %101 ;
                    %102 : int = var.load %70 ;
                    %103 : int = constant @"1" ;
                    %104 : int = add %102 %103 ;
                    var.store %70 %104 ;
                    java.continue ;
                };
            %105 : int = java.cexpression
                ()boolean -> {
                    %106 : int = var.load %70 ;
                    %107 : hat.buffer.S32Array = var.load %8 ;
                    %108 : int = invoke %107 @"hat.buffer.S32Array::length()int" ;
                    %109 : boolean = lt %106 %108 ;
                    yield %109 ;
                }
                ()int -> {
                    %110 : hat.buffer.S32Array = var.load %8 ;
                    %111 : int = var.load %70 ;
                    %112 : long = conv %111 ;
                    %113 : int = invoke %110 %112 @"hat.buffer.S32Array::array(long)int" ;
                    yield %113 ;
                }
                ()int -> {
                    %114 : int = constant @"0" ;
                    yield %114 ;
                };
            %115 : Var<int> = var %105 @"color" ;
            %116 : hat.buffer.S32Array2D = var.load %7 ;
            %117 : hat.KernelContext = var.load %6 ;
            %118 : int = field.load %117 @"hat.KernelContext::x()int" ;
            %119 : long = conv %118 ;
            %120 : int = var.load %115 ;
            invoke %116 %119 %120 @"hat.buffer.S32Array2D::array(long, int)void" ;
            yield ;
        }
        ()void -> {
            yield;
        };
    return ;
};

```
From the above we can generate C99 style CUDA code


Here is the Cuda C99 code generated

```C
typedef struct KernelContext_s{
    int x;
    int maxX;
}KernelContext_t;

typedef struct S32Array2D_s{
    int width;
    int height;
    int array[0];
}S32Array2D_t;

typedef struct S32Array_s{
    int length;
    int array[0];
}S32Array_t;


extern "C" __global__ void mandel(
     S32Array2D_t* s32Array2D,  S32Array_t* pallette, float offsetx, float offsety, float scale
){
    KernelContext_t kc;
    kc.x=blockIdx.x*blockDim.x+threadIdx.x;
    kc.maxX=gridDim.x*blockDim.x;
    if(kc.x<kc.maxX){
        float width = (float)s32Array2D->width;
        float height = (float)s32Array2D->height;
        float x = ((float)(kc.x%s32Array2D->width)*scale-scale/2.0*width)/width+offsetx;
        float y = ((float)(kc.x/s32Array2D->width)*scale-scale/2.0*height)/height+offsety;
        float zx = x;
        float zy = y;
        float new_zx = 0.0;
        int colorIdx = 0;
        while(colorIdx<pallette->length && zx*zx+zy*zy<4.0){
            new_zx=zx*zx-zy*zy+x;
            zy=2.0*zx*zy+y;
            zx=new_zx;
            colorIdx=colorIdx+1;
        }
        int color = colorIdx<pallette->length?pallette->array[(long)colorIdx]:0;
        s32Array2D->array[(long)kc.x]=color;
    }
    return;
}
```

Which we can convert to ptx using the nvcc compiler (on a NVDIA platform)

`nvcc -ptx mandel.cu -o mandel.ptx`

```
//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-33191640
// Cuda compilation tools, release 12.2, V12.2.140
// Based on NVVM 7.0.1
//

.version 8.2
.target sm_52
.address_size 64

        // .globl       mandel

.visible .entry mandel(
        .param .u64 mandel_param_0,
        .param .u64 mandel_param_1,
        .param .f32 mandel_param_2,
        .param .f32 mandel_param_3,
        .param .f32 mandel_param_4
)
{
        .reg .pred      %p<9>;
        .reg .f32       %f<29>;
        .reg .b32       %r<24>;
        .reg .f64       %fd<22>;
        .reg .b64       %rd<9>;


        ld.param.u64    %rd3, [mandel_param_0];
        ld.param.u64    %rd4, [mandel_param_1];
        ld.param.f32    %f13, [mandel_param_2];
        ld.param.f32    %f14, [mandel_param_3];
        ld.param.f32    %f15, [mandel_param_4];
        cvta.to.global.u64      %rd1, %rd4;
        cvta.to.global.u64      %rd2, %rd3;
        mov.u32         %r8, %ntid.x;
        mov.u32         %r9, %ctaid.x;
        mov.u32         %r10, %tid.x;
        mad.lo.s32      %r1, %r9, %r8, %r10;
        mov.u32         %r11, %nctaid.x;
        mul.lo.s32      %r12, %r11, %r8;
        setp.ge.s32     %p1, %r1, %r12;
        @%p1 bra        $L__BB0_7;

        ld.global.u32   %r14, [%rd2];
        cvt.rn.f32.s32  %f16, %r14;
        ld.global.u32   %r15, [%rd2+4];
        cvt.rn.f32.s32  %f17, %r15;
        div.s32         %r16, %r1, %r14;
        mul.lo.s32      %r17, %r16, %r14;
        sub.s32         %r18, %r1, %r17;
        cvt.rn.f32.s32  %f18, %r18;
        mul.f32         %f19, %f18, %f15;
        cvt.f64.f32     %fd2, %f19;
        cvt.f64.f32     %fd3, %f15;
        mul.f64         %fd4, %fd3, 0d3FE0000000000000;
        cvt.f64.f32     %fd5, %f16;
        mul.f64         %fd6, %fd4, %fd5;
        sub.f64         %fd7, %fd2, %fd6;
        div.rn.f64      %fd8, %fd7, %fd5;
        cvt.f64.f32     %fd9, %f13;
        add.f64         %fd10, %fd8, %fd9;
        cvt.rn.f32.f64  %f1, %fd10;
        cvt.rn.f32.s32  %f20, %r16;
        mul.f32         %f21, %f20, %f15;
        cvt.f64.f32     %fd11, %f21;
        cvt.f64.f32     %fd12, %f17;
        mul.f64         %fd13, %fd4, %fd12;
        sub.f64         %fd14, %fd11, %fd13;
        div.rn.f64      %fd15, %fd14, %fd12;
        cvt.f64.f32     %fd16, %f14;
        add.f64         %fd17, %fd15, %fd16;
        cvt.rn.f32.f64  %f27, %fd17;
        ld.global.u32   %r2, [%rd1];
        setp.lt.s32     %p2, %r2, 1;
        mov.u32         %r23, 0;
        mul.f32         %f26, %f1, %f1;
        mul.f32         %f25, %f27, %f27;
        add.f32         %f22, %f26, %f25;
        setp.geu.f32    %p3, %f22, 0f40800000;
        or.pred         %p4, %p2, %p3;
        mov.u32         %r22, %r23;
        @%p4 bra        $L__BB0_4;

        cvt.f64.f32     %fd1, %f27;
        mov.f32         %f28, %f1;

$L__BB0_3:
        sub.f32         %f23, %f26, %f25;
        add.f32         %f9, %f23, %f1;
        cvt.f64.f32     %fd18, %f28;
        add.f64         %fd19, %fd18, %fd18;
        cvt.f64.f32     %fd20, %f27;
        fma.rn.f64      %fd21, %fd19, %fd20, %fd1;
        cvt.rn.f32.f64  %f27, %fd21;
        add.s32         %r22, %r22, 1;
        setp.lt.s32     %p5, %r22, %r2;
        mul.f32         %f26, %f9, %f9;
        mul.f32         %f25, %f27, %f27;
        add.f32         %f24, %f26, %f25;
        setp.lt.f32     %p6, %f24, 0f40800000;
        and.pred        %p7, %p5, %p6;
        mov.f32         %f28, %f9;
        @%p7 bra        $L__BB0_3;

$L__BB0_4:
        setp.ge.s32     %p8, %r22, %r2;
        @%p8 bra        $L__BB0_6;

        mul.wide.s32    %rd5, %r22, 4;
        add.s64         %rd6, %rd1, %rd5;
        ld.global.u32   %r23, [%rd6+4];

$L__BB0_6:
        mul.wide.s32    %rd7, %r1, 4;
        add.s64         %rd8, %rd2, %rd7;
        st.global.u32   [%rd8+8], %r23;

$L__BB0_7:
        ret;

}
```
But we would like to create the ptx directly from the babylon model, without being on an NVidia platform.  Probably the lowered model

We will probably use the lowered babylon model

```
func @"mandel" @loc="39:5:file:/Users/grfrost/orahub/hat/examples/mandel/src/java/mandel/MandelCompute.java"
  (%0 : hat.KernelContext, %1 : hat.buffer.S32Array2D, %2 : hat.buffer.S32Array, %3 : float, %4 : float, %5 : float)void -> {
    %6 : Var<hat.KernelContext> = var %0 @"kc" ;
    %7 : Var<hat.buffer.S32Array2D> = var %1 @"s32Array2D" ;
    %8 : Var<hat.buffer.S32Array> = var %2 @"pallette" ;
    %9 : Var<float> = var %3 @"offsetx" ;
    %10 : Var<float> = var %4 @"offsety" ;
    %11 : Var<float> = var %5 @"scale" ;
    %12 : hat.KernelContext = var.load %6 ;
    %13 : int = field.load %12 @"hat.KernelContext::x()int" ;
    %14 : hat.KernelContext = var.load %6 ;
    %15 : int = field.load %14 @"hat.KernelContext::maxX()int" ;
    %16 : boolean = lt %13 %15 ;
    cbranch %16 ^block_0 ^block_1;

  ^block_0:
    %17 : hat.buffer.S32Array2D = var.load %7 ;
    %18 : int = invoke %17 @"hat.buffer.S32Array2D::width()int" ;
    %19 : float = conv %18 ;
    %20 : Var<float> = var %19 @"width" ;
    %21 : hat.buffer.S32Array2D = var.load %7 ;
    %22 : int = invoke %21 @"hat.buffer.S32Array2D::height()int" ;
    %23 : float = conv %22 ;
    %24 : Var<float> = var %23 @"height" ;
    %25 : hat.KernelContext = var.load %6 ;
    %26 : int = field.load %25 @"hat.KernelContext::x()int" ;
    %27 : hat.buffer.S32Array2D = var.load %7 ;
    %28 : int = invoke %27 @"hat.buffer.S32Array2D::width()int" ;
    %29 : int = mod %26 %28 ;
    %30 : float = conv %29 ;
    %31 : float = var.load %11 ;
    %32 : float = mul %30 %31 ;
    %33 : float = var.load %11 ;
    %34 : float = constant @"2.0" ;
    %35 : float = div %33 %34 ;
    %36 : float = var.load %20 ;
    %37 : float = mul %35 %36 ;
    %38 : float = sub %32 %37 ;
    %39 : float = var.load %20 ;
    %40 : float = div %38 %39 ;
    %41 : float = var.load %9 ;
    %42 : float = add %40 %41 ;
    %43 : Var<float> = var %42 @"x" ;
    %44 : hat.KernelContext = var.load %6 ;
    %45 : int = field.load %44 @"hat.KernelContext::x()int" ;
    %46 : hat.buffer.S32Array2D = var.load %7 ;
    %47 : int = invoke %46 @"hat.buffer.S32Array2D::width()int" ;
    %48 : int = div %45 %47 ;
    %49 : float = conv %48 ;
    %50 : float = var.load %11 ;
    %51 : float = mul %49 %50 ;
    %52 : float = var.load %11 ;
    %53 : float = constant @"2.0" ;
    %54 : float = div %52 %53 ;
    %55 : float = var.load %24 ;
    %56 : float = mul %54 %55 ;
    %57 : float = sub %51 %56 ;
    %58 : float = var.load %24 ;
    %59 : float = div %57 %58 ;
    %60 : float = var.load %10 ;
    %61 : float = add %59 %60 ;
    %62 : Var<float> = var %61 @"y" ;
    %63 : float = var.load %43 ;
    %64 : Var<float> = var %63 @"zx" ;
    %65 : float = var.load %62 ;
    %66 : Var<float> = var %65 @"zy" ;
    %67 : float = constant @"0.0" ;
    %68 : Var<float> = var %67 @"new_zx" ;
    %69 : int = constant @"0" ;
    %70 : Var<int> = var %69 @"colorIdx" ;
    branch ^block_2;

  ^block_2:
    %71 : int = var.load %70 ;
    %72 : hat.buffer.S32Array = var.load %8 ;
    %73 : int = invoke %72 @"hat.buffer.S32Array::length()int" ;
    %74 : boolean = lt %71 %73 ;
    cbranch %74 ^block_3 ^block_4(%74);

  ^block_3:
    %75 : float = var.load %64 ;
    %76 : float = var.load %64 ;
    %77 : float = mul %75 %76 ;
    %78 : float = var.load %66 ;
    %79 : float = var.load %66 ;
    %80 : float = mul %78 %79 ;
    %81 : float = add %77 %80 ;
    %82 : float = constant @"4.0" ;
    %83 : boolean = lt %81 %82 ;
    branch ^block_4(%83);

  ^block_4(%84 : boolean):
    cbranch %84 ^block_5 ^block_6;

  ^block_5:
    %85 : float = var.load %64 ;
    %86 : float = var.load %64 ;
    %87 : float = mul %85 %86 ;
    %88 : float = var.load %66 ;
    %89 : float = var.load %66 ;
    %90 : float = mul %88 %89 ;
    %91 : float = sub %87 %90 ;
    %92 : float = var.load %43 ;
    %93 : float = add %91 %92 ;
    var.store %68 %93 ;
    %94 : float = constant @"2.0" ;
    %95 : float = var.load %64 ;
    %96 : float = mul %94 %95 ;
    %97 : float = var.load %66 ;
    %98 : float = mul %96 %97 ;
    %99 : float = var.load %62 ;
    %100 : float = add %98 %99 ;
    var.store %66 %100 ;
    %101 : float = var.load %68 ;
    var.store %64 %101 ;
    %102 : int = var.load %70 ;
    %103 : int = constant @"1" ;
    %104 : int = add %102 %103 ;
    var.store %70 %104 ;
    branch ^block_2;

  ^block_6:
    %105 : int = var.load %70 ;
    %106 : hat.buffer.S32Array = var.load %8 ;
    %107 : int = invoke %106 @"hat.buffer.S32Array::length()int" ;
    %108 : boolean = lt %105 %107 ;
    cbranch %108 ^block_7 ^block_8;

  ^block_7:
    %109 : hat.buffer.S32Array = var.load %8 ;
    %110 : int = var.load %70 ;
    %111 : long = conv %110 ;
    %112 : int = invoke %109 %111 @"hat.buffer.S32Array::array(long)int" ;
    branch ^block_9(%112);

  ^block_8:
    %113 : int = constant @"0" ;
    branch ^block_9(%113);

  ^block_9(%114 : int):
    %115 : Var<int> = var %114 @"color" ;
    %116 : hat.buffer.S32Array2D = var.load %7 ;
    %117 : hat.KernelContext = var.load %6 ;
    %118 : int = field.load %117 @"hat.KernelContext::x()int" ;
    %119 : long = conv %118 ;
    %120 : int = var.load %115 ;
    invoke %116 %119 %120 @"hat.buffer.S32Array2D::array(long, int)void" ;
    branch ^block_10;

  ^block_1:
    branch ^block_10;

  ^block_10:
    return ;
};
```

