
# Cascade Issue 
I think I am fighting an alignment issue with nested layouts.

Ultimately I need to pass this struct to the GPU.

```C
typedef struct Cascade_s{
    int width;
    int height;
    int featureCount;
    struct Feature_t{
       int id;
       float threshold;
       struct LinkOrValue_t{
          boolean hasValue;
          union Anon_t{
             int featureId;
             float value;
          } anon;
       } left, right;
       struct Rect_t{
          byte x;
          byte y;
          byte width;
          byte height;
          float weight;
       } rect[3];
    } feature[2913];
    int stageCount;
    struct Stage_t{
       int id;
       float threshold;
       short firstTreeId;
       short treeCount;
    } stage[25];
    int treeCount;
    struct Tree_t{
       int id;
       short firstFeatureId;
       short featureCount;
    } tree[291
} Cascade_t;
```

I will expand this to a more readable form 

```C
typedef union Anon_s{
    int featureId;
    float value;
}Anon_t;

typedef struct LinkOrValue_s{
    boolean hasValue;
    Anon_t anon;
}LinkOrValue_t;

typedef struct Rect_s{
    byte x;
    byte y;
    byte width;
    byte height;
    float weight;
}Rect_t;

typedef struct Feature_s{
    int id;
    float threshold;
    LinkOrValue_t left;
    LinkOrValue_t right;
    Rect_t rect[3];
}Feature_t;

typedef struct Stage_s{
    int id;
    float threshold;
    short firstTreeId;
    short treeCount;
}Stage_t;

typedef struct Tree_s{
    int id;
    short firstFeatureId;
    short featureCount;
}Tree_t;

typedef struct Cascade_s{
    int width;
    int height;
    int featureCount;
    Feature_t feature[2913];
    int stageCount;
    Stage_t stage[25];
    int treeCount;
    Tree_t tree[291
} Cascade_t;
```

I use nested mapped segment interfaces, for this `Cascade_t` and the 
interface `nest` can be found here
[Cascade.java](https://orahub.oci.oraclecorp.com/gary.frost/hat/-/blame/main/examples/violajones/src/java/violajones/ifaces/Cascade.java#L43)
(I also added as stripped down version
at the end of this doc).

To dispatch a kernel, I pass a pointer to a `Cascade_t` to native code, 
which is then handed off to the GPU. I have observed this alignment issue on 
Both the GPU and the CPU. It's just harder to debug on the GPU.   

The first observation is that `sizeof(Cascade_t)` is smaller than the `MemorySegment.sizeInBytes`
of the allocated segment.

So
```c++
Cascade_t *cascade =  (Cascade_t *)arg->value.buffer.memorySegment;
std::cout << "sizeof(Cascade_t)       ="<< sizeof(Cascade_t) <<std::endl;
std::cout << "Cascade_t segment bytes ="<< (size_t)arg->value.buffer.sizeInBytes<< std::endl;
```
Produces
```
sizeof(Cascade_t)       =163448
Cascade_t segment bytes =186752
```

Also I find that whilst  `cascade->width`, `cascade->height` and
`cascade->featureCount` are all as expected (first few fields of the struct), the value returned for `cascade->stageCount` is
messed up. 

To me this indicates that the issue is with the array of `Feature_t` following the `featureCount` field.

```
std::cout << "cascade->width          ="<< cascade->width <<std::endl;
std::cout << "cascade->height         ="<< cascade->height <<std::endl;
std::cout << "cascade->featureCount   ="<< cascade->featureCount <<std::endl;
std::cout << "cascade->stageCount     ="<< cascade->stageCount <<std::endl;
```
Producing
```
cascade->width          =24
cascade->height         =24
cascade->featureCount   =2913
cascade->stageCount     =?????????? <-- nonsense
```

So digging deeper ;), if I loop over the `Feature_t`'s on the native side
to extract the `id` and hexdump the `Feature_t` using this loop.  
```C
for(int i=0; i<3; i++){ // first 3 of cascade->featureCount
   Feature_t *feature = cascade->feature + i;
   Feature_t *feature = &cascade->feature[(long)i];
   std::cout << " feature->id "<< std::hex << feature->id <<std::dec <<std::endl;
   hexdump(feature, sizeof(Feature_t));
}
```
I find that instead of the `id`'s monotonically increasing
in value (0,1,2,3,4...) I get nonsense (after the first). 

```
     feature->id 0         <-- expect 0
000000: 00 00 00 00 ba 12 01 bd 01 00 00 00 00 00 00 00  ....�..�........
000010: 39 9a 05 40 01 00 00 00 00 00 00 00 c5 e6 0d c0  9..@........��.�
000020: 06 04 0c 09 00 00 80 bf 06 07 0c 03 00 00 40 40  .......�......@@
     feature->id 0         <-- expect 1
000000: 00 00 00 00 00 00 00 00<01>00 00 00 98 18 4b 3c  ..............K<
000010: 01 00 00 00 00 00 00 00 b2 83 ee bf 01 00 00 00  ........�.�....
000020: 00 00 00 00 da e1 a9 3f 06 04 0c 07 00 00 80 bf  ....��?.......�
     feature->id 704040a   <-- expect 2
000000: 0a 04 04 07 00 00 40 40 00 00 00 00 00 00 00 00  ......@@........
000010:<02> 00 00 00 59 a2 b3 3c 01 00 00 00 00 00 00 00  ....Y��<........
000020: e2 58 c1 bf 01 00 00 00 00 00 00 00 64 02 88 3f  �X��........d..?
```
Also note the highlighted values `<01>` and  `<02>`.  
Which I take to be the `id`'s inside the misaligned structs.
To me this indicates that we have a discrepancy in the size of my Typedef  and the layout representing `Feature_t` is 8 bytes longer than the typedef. 

Indeed if I hack my [Typedef builder code](https://orahub.oci.oraclecorp.com/gary.frost/hat/-/blame/main/hat/src/java/hat/backend/c99codebuilders/C99HatBuilder.java#L437)  for this one typedef

```java
//Around Line 444 of C99HatBuilder  
if (typeDef.name().equals("Feature")){
   nl().append("char padme[8]").semicolon();
}
```
Yielding
```java
typedef struct Feature_s{
    int id;
    float threshold;
    LinkOrValue_t left;
    LinkOrValue_t right;
    Rect_t rect[3];
    char padme[8];   //<---  added by the hack above 
}Feature_t;
```
Then rebuild ;) and rerun.  Some sanity starts to return to the debug code I added

```
sizeof(Cascade_t)       =186752  
Cascade_t segment bytes =186752   <-- matches above
cascade->width          =24
cascade->height         =24
cascade->featureCount   =2913
cascade->stageCount     =25       <--  correct now
     feature->id 0
000000: 00 00 00 00 ba 12 01 bd 01 00 00 00 00 00 00 00  ....�..�........
000010: 39 9a 05 40 01 00 00 00 00 00 00 00 c5 e6 0d c0  9..@........��.�
000020: 06 04 0c 09 00 00 80 bf 06 07 0c 03 00 00 40 40  .......�......@@
000030: 00 00 00 00 00 00 00 00                          ........
     feature->id 1
000000: 01 00 00 00 98 18 4b 3c 01 00 00 00 00 00 00 00  ......K<........
000010: b2 83 ee bf 01 00 00 00 00 00 00 00 da e1 a9 3f  �.�........��?
000020: 06 04 0c 07 00 00 80 bf 0a 04 04 07 00 00 40 40  .......�......@@
000030: 00 00 00 00 00 00 00 00                          ........
     feature->id 2
000000: 02 00 00 00 59 a2 b3 3c 01 00 00 00 00 00 00 00  ....Y��<........
000010: e2 58 c1 bf 01 00 00 00 00 00 00 00 64 02 88 3f  �X��........d..?
000020: 03 09 12 09 00 00 80 bf 03 0c 12 03 00 00 40 40  .......�......@@
000030: 00 00 00 00 00 00 00 00                          ........
     feature->id 3
000000: 03 00 00 00 a9 83 bc 3b 01 00 00 00 00 00 00 00  ....�.�;........
000010: 57 e8 5f bf 01 00 00 00 00 00 00 00 48 88 96 3f  W�_�........H..?
000020: 08 12 09 06 00 00 80 bf 08 14 09 02 00 00 40 40  .......�......@@
000030: 00 00 00 00 00 00 00 00                          ........
```

So what is going wrong here?  

I think that the issue is here 
```java
StructLayout featureLayout = MemoryLayout.structLayout(
    JAVA_INT.withName("id"),
    JAVA_FLOAT.withName("threshold"),
    Feature.LinkOrValue.layout.withName("left"),
    Feature.LinkOrValue.layout.withName("right"),
    MemoryLayout.sequenceLayout(3, Feature.Rect.layout).withName("rect")
);

```
Which is used to build an array of `Feature_t`'s here 
```java
   JAVA_INT.withName("width"),
   JAVA_INT.withName("height"), 
   JAVA_INT.withName("featureCount"),
   sequenceLayout(haarCascade.features.size(), Feature.layout.withName("feature"),
   JAVA_INT.withName("stageCount"),
   sequenceLayout(haarCascade.stages.size(), Stage.layout.withName("stage"),
   JAVA_INT.withName("treeCount"),
   sequenceLayout(haarCascade.trees.size(), Tree.layout.withName("tree")
```
I think that the sequenceLayout is `unnecessarily` padding 8 bytes to each  `Feature_t`... as it builds the array. 

Let's dump the sizes of typedefs and layouts ;)  
```C
std::cout << "cascade->featureCount"<< cascade->featureCount <<std::endl;
std::cout << "cascade->stageCount"<< cascade->stageCount <<std::endl;
std::cout << "sizeof(Feature_t) "<< sizeof(Feature_t) <<std::endl;
std::cout << "sizeof(cascade->feature) "<< sizeof(cascade->feature) <<std::endl;
std::cout << "sizeof(cascade->feature[0]) "<< sizeof(cascade->feature[0]) <<std::endl;
```
Yielding the following for the 'padded feature'
```
cascade->featureCount =2913
sizeof(Feature_t) = 56
sizeof(cascade->feature) = 163128
sizeof(cascade->feature[0]) = 56
```
And sure enough 2913*56 = 163128
vs the following for the unpadded
```
cascade->featureCount2913
sizeof(Feature_t) 48
sizeof(cascade->feature) 139824
sizeof(cascade->feature[0]) 48
```
And sure enough 2913*48 = 139824

Now the layouts...  
```java
var featureSequenceLayout = sequenceLayout(haarCascade.features.size(), Feature.layout.withName(Feature.class.getSimpleName())).withName("feature");
System.out.println("Feature.layout.byteSize() "+Feature.layout.byteSize());
System.out.println("FeatureSequence.layout.byteSize() "+featureSequenceLayout.byteSize());
```
Yielding
```
Feature.layout.byteSize() 56
FeatureSequence.layout.byteSize() 163128
```

OK I found this
```java
  StructLayout layout = MemoryLayout.structLayout(
                    JAVA_BOOLEAN.withName("hasValue"),
                    MemoryLayout.paddingLayout(3), //<-- was 7 
                    Feature.LinkOrValue.Anon.layout.withName("anon")
                ).withName("LinkOrValue");
```
-----
###  Here are layouts for the Cascade.  

```java
public interface Cascade extends CompleteBuffer {
    interface Feature {
        interface Rect {
            StructLayout layout = MemoryLayout.structLayout(
               JAVA_BYTE.withName("x"),
               JAVA_BYTE.withName("y"),
               JAVA_BYTE.withName("width"),
               JAVA_BYTE.withName("height"),
               JAVA_FLOAT.withName("weight")
            ).withName("Rect");
        }
        interface LinkOrValue {
            interface Anon {
                MemoryLayout layout = MemoryLayout.unionLayout(
                        JAVA_INT.withName("featureId"),
                        JAVA_FLOAT.withName("value")
                ).withName("Anon");
            }

            StructLayout layout = MemoryLayout.structLayout(
                JAVA_BOOLEAN.withName("hasValue"),
                MemoryLayout.paddingLayout(7),
                Anon.layout.withName("anon")
            ).withName("LinkOrValue");
        }

        StructLayout layout = MemoryLayout.structLayout(
            JAVA_INT.withName("id"),
            JAVA_FLOAT.withName("threshold"),
            Feature.LinkOrValue.layout.withName("left"),
            Feature.LinkOrValue.layout.withName("right"),
            MemoryLayout.sequenceLayout(3, Feature.Rect.layout).withName("rect")
        );
    }

    interface Stage {
        StructLayout layout = MemoryLayout.structLayout(
                JAVA_INT.withName("id"),
                JAVA_FLOAT.withName("threshold"),
                JAVA_SHORT.withName("firstTreeId"),
                JAVA_SHORT.withName("treeCount")
        ).withName("Stage");
    }
    interface Tree {
        StructLayout layout = MemoryLayout.structLayout(
           JAVA_INT.withName("id"),
           JAVA_SHORT.withName("firstFeatureId"),
           JAVA_SHORT.withName("featureCount")
        ).withName("Tree");
    }

    static Cascade create(Accelerator accelerator, XMLHaarCascadeModel haarCascade) {
        return SegmentMapper.of(accelerator.lookup, Cascade.class,
          JAVA_INT.withName("width"),
          JAVA_INT.withName("height"),
          JAVA_INT.withName("featureCount"),
          sequenceLayout(haarCascade.features.size(), Feature.layout.withName("feature"),
          JAVA_INT.withName("stageCount"),
          sequenceLayout(haarCascade.stages.size(), Stage.layout.withName("stage"),
          JAVA_INT.withName("treeCount"),
          sequenceLayout(haarCascade.trees.size(), Tree.layout.withName("tree")
        ).allocate(accelerator.arena());
    }
}
```