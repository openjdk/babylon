
# More Complex Interface Mapping Example - The Cascade

----

* [Contents](hat-00.md)
* House Keeping
    * [Project Layout](hat-01-01-project-layout.md)
    * [Building Babylon](hat-01-02-building-babylon.md)
    * [Maven and CMake](hat-01-03-maven-cmake.md)
* Programming Model
    * [Programming Model](hat-03-programming-model.md)
* Interface Mapping
    * [Interface Mapping Overview](hat-04-01-interface-mapping.md)
    * [Cascade Interface Mapping](hat-04-02-cascade-interface-mapping.md)
* Implementation Detail
    * [Walkthrough Of Accelerator.compute()](hat-accelerator-compute.md)

----

# More Complex Interface Mapping Example - The Cascade

Previously we showed probably the minimal useful mapping with S32Array

The HaarCascade example has multiple nested interfaces representing data
structures involving various nested structs and unions

```java
public interface Cascade extends Buffer {
    int width();
    void width(int width);
    int height();
    void height(int height);
    interface Feature extends Buffer.StructChild {
        int id();
        float threshold();
        void id(int id);
        void threshold(float threshold);
        interface LinkOrValue extends Buffer.StructChild {
            interface Anon extends Buffer.UnionChild {
                int featureId();
                void featureId(int featureId);
                float value();
                void value(float value);
            }
            boolean hasValue();
            void hasValue(boolean hasValue);
            Anon anon();
        }
        LinkOrValue left();
        LinkOrValue right();
        interface Rect extends Buffer.StructChild {
            byte x();
            byte y();
            byte width();
            byte height();
            float weight();
            void x(byte x);
            void y(byte y);
            void width(byte width);
            void height(byte height);
            void weight(float height);
        }
        Rect rect(long idx);
    }
    int featureCount();
    void featureCount(int featureCount);
    Feature feature(long idx);
    interface Stage extends Buffer.StructChild {
        float threshold();
        short firstTreeId();
        short treeCount();
        int id();
        void id(int id);
        void threshold(float threshold);
        void firstTreeId(short firstTreeId);
        void treeCount(short treeCount);
    }

    int stageCount();
    void stageCount(int stageCount);
    Stage stage(long idx);
    interface Tree extends Buffer.StructChild {
        void id(int id);
        void firstFeatureId(short firstFeatureId);
        void featureCount(short featureCount);
        int id();
        short firstFeatureId();
        short featureCount();
    }
    int treeCount();
    void treeCount(int treeCount);
    Tree tree(long idx);
    Schema<Cascade> schema = Schema.of(Cascade.class, cascade -> cascade
            .fields("width", "height")
            .arrayLen("featureCount")
                .array("feature", feature -> feature
                    .field("id")
                    .field("threshold")
                    .fields("left", "right", linkOrValue -> linkOrValue
                            .field("hasValue")
                            .pad(3)
                            .field("anon", anon -> anon
                               .field("featureId")
                               .field("value")
                            )
                    )
                    .array("rect", 3, rect -> rect
                       .field("x")
                       .field("y")
                       .field("width")
                       .field("height").
                       .field("weight"))
            )
            .arrayLen("treeCount")
                .array("tree", tree -> tree
                   .field("id")
                   .field("featureCount")
                   .field("firstFeatureId")
                )
            .arrayLen("stageCount")
                .array("stage", stage -> stage
                   .field("id")
                   .field("threshold")
                   .field("treeCount")
                   .field("firstTreeId"))
    );
}
```

Another great advantage of using interfaces is that we can choose
to re implement the interface in any was we see fit.

For example we load  HaarCascades from XML files.
We therefore can create an implementation of the Cascade interface which just
loads the XML DOM... stores them in arrays and the interface methods just delegate to
the appropriately wrapped  w3c.Node tree nodes ;)

If we know we are using Java backend we can actually
just pass the XMLCascade implementation directly to the backend...

Actually the Cascade `create` method takes an existing
implementation of a Cascade and clones it.
So we can just pass it an XMLHaarCascade ;)

So we build an XMLCascade then pass it to the `create` method of the iface
mapped Cascade

```java
   XMLHaarCascadeModel haarCascade = XMLHaarCascadeModel.load(
        ViolaJonesRaw.class.getResourceAsStream("haarcascade_frontalface_default.xml"));

   assert haarCascade instanceof Cascade; // Here it is just an interface

   Cascade cascade = Cascade.create(accelerator, haarCascade);

   // Now it can be used on the GPU
```

The implementation is currently hand crafted, but this could easily be automated.

```java
 static Cascade create(BufferAllocator bufferAllocator, XMLHaarCascadeModel haarCascade) {

        Cascade cascade = bufferAllocator.allocate(SegmentMapper.of(MethodHandles.lookup(), Cascade.class,
                JAVA_INT.withName("width"),
                JAVA_INT.withName("height"),
                JAVA_INT.withName("featureCount"),
                sequenceLayout(haarCascade.features.size(), Feature.layout.withName(Feature.class.getSimpleName())).withName("feature"),
                JAVA_INT.withName("stageCount"),
                sequenceLayout(haarCascade.stages.size(), Stage.layout.withName(Stage.class.getSimpleName())).withName("stage"),
                JAVA_INT.withName("treeCount"),
                sequenceLayout(haarCascade.trees.size(), Tree.layout.withName(Tree.class.getSimpleName())).withName("tree")
        ));
        cascade.width(haarCascade.width());
        cascade.height(haarCascade.height());
        cascade.featureCount(haarCascade.features.size());
        cascade.stageCount(haarCascade.stages.size());
        cascade.treeCount(haarCascade.trees.size());
        for (int idx = 0; idx < haarCascade.features.size(); idx++) {
            Cascade.Feature cascadeFeature = cascade.feature(idx);
            var haarfeature = haarCascade.features.get(idx);
            cascadeFeature.id(haarfeature.id());
            cascadeFeature.threshold(haarfeature.threshold());
            Cascade.Feature.LinkOrValue cascadeLeft = cascadeFeature.left();
            cascadeLeft.hasValue(haarfeature.left.hasValue());
            if (haarfeature.left.hasValue()) {
                cascadeLeft.anon().value(haarfeature.left.value());
            } else {
                cascadeLeft.anon().value(haarfeature.left.featureId());
            }
            Cascade.Feature.LinkOrValue cascadeRight = cascadeFeature.right();
            cascadeRight.hasValue(haarfeature.right.hasValue());
            if (haarfeature.right.hasValue()) {
                cascadeRight.anon().value(haarfeature.right.value());
            } else {
                cascadeRight.anon().featureId(haarfeature.right.featureId());
            }
            for (int r = 0; r < 3; r++) {
                var haarrect = haarfeature.rects[r];
                if (haarrect != null) {
                    Cascade.Feature.Rect cascadeRect = cascadeFeature.rect(r);
                    cascadeRect.x(haarrect.x());
                    cascadeRect.y(haarrect.y());
                    cascadeRect.width(haarrect.width());
                    cascadeRect.height(haarrect.height());
                    cascadeRect.weight(haarrect.weight());
                }
            }
        }
        for (XMLHaarCascadeModel.Stage haarstage : haarCascade.stages) {
            Cascade.Stage cascadeStage = cascade.stage(haarstage.id);
            cascadeStage.id(haarstage.id());
            cascadeStage.threshold(haarstage.threshold());
            cascadeStage.firstTreeId(haarstage.firstTreeId());
            cascadeStage.treeCount(haarstage.treeCount());
        }

        for (XMLHaarCascadeModel.Tree haarTree : haarCascade.trees) {
            Cascade.Tree cascadeTree = cascade.tree(haarTree.id());
            cascadeTree.id(haarTree.id());
            cascadeTree.firstFeatureId(haarTree.firstFeatureId());
            cascadeTree.featureCount(haarTree.featureCount());
        }
        return cascade;
    }
```

