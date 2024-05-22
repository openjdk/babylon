package hat.ifacemapper;


import java.lang.foreign.Arena;
import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.function.Function;
import java.util.stream.Stream;

/**
 * A segment mapper can project memory segment onto and from class instances.
 * <p>
 * More specifically, a segment mapper can project a backing
 * {@linkplain MemorySegment MemorySegment} into new {@link Record} instances or new
 * instances that implements an interface by means of matching the names of the record
 * components or interface methods with the names of member layouts in a group layout.
 * A segment mapper can also be used in the other direction, where records and interface
 * implementing instances can be used to update a target memory segment. By using any of
 * the {@linkplain #map(Class,  Function) map} operations, segment mappers can be
 * used to map between memory segments and additional Java types other than record and
 * interfaces (such as JavaBeans).
 *
 * <p>
 * In short, a segment mapper finds, for each record component or interface method,
 * a corresponding member layout with the same name in the group layout. There are some
 * restrictions on the record component type and the corresponding member layout type
 * (e.g. a record component of type {@code int} can only be matched with a member layout
 * having a carrier type of {@code int.class} (such as {@link ValueLayout#JAVA_INT})).
 * <p>
 * Using the member layouts (e.g. observing offsets and
 * {@link java.nio.ByteOrder byte ordering}), a number of extraction methods are then
 * identified for all the record components or interface methods and these are stored
 * internally in the segment mapper.
 *
 * <h2 id="mapping-kinds">Mapping kinds</h2>
 *
 * Segment mappers can be of two fundamental kinds;
 * <ul>
 *     <li>Record</li>
 *     <li>Interface</li>
 * </ul>
 * <p>
 * The characteristics of the mapper kinds are summarized in the following table:
 *
 * <blockquote><table class="plain">
 * <caption style="display:none">Mapper characteristics</caption>
 * <thead>
 * <tr>
 *     <th scope="col">Mapper kind</th>
 *     <th scope="col">Temporal mode</th>
 *     <th scope="col">Get operations</th>
 *     <th scope="col">Set operations</th>
 *     <th scope="col">Segment access</th>
 * </tr>
 * </thead>
 * <tbody>
 * <tr><th scope="row" style="font-weight:normal">Record</th>
 *     <td style="text-align:center;">Eager</td>
 *     <td style="text-align:center;">Extract all component values from the source segment, build the record</td>
 *     <td style="text-align:center;">Write all component values to the target segment</td>
 *     <td style="text-align:center;">N/A</td></tr>
 * <tr><th scope="row" style="font-weight:normal">Interface</th>
 *     <td style="text-align:center;">Lazy</td>
 *     <td style="text-align:center;">Wrap the source segment into a new interface instance</td>
 *     <td style="text-align:center;">Copy the relevant values from the initial source segment into the target segment</td>
 *     <td style="text-align:center;">via <code>SegmentMapper::segment</code></td></tr>
 * </tbody>
 * </table></blockquote>

 * <h2 id="mapping-records">Mapping Records</h2>
 *
 * The example below shows how to extract an instance of a public
 * <em>{@code Point} record class</em> from a {@link MemorySegment} and vice versa:
 * {@snippet lang = java:
 *
 *  static final GroupLayout POINT = MemoryLayout.structLayout(JAVA_INT.withName("x"), JAVA_INT.withName("y"));
 *  public record Point(int x, int y){}
 *  //...
 *  MemorySegment segment = MemorySegment.ofArray(new int[]{3, 4, 0, 0});
 *
 *  // Obtain a SegmentMapper for the Point record type
 *  SegmentMapper<Point> recordMapper = SegmentMapper.ofRecord(Point.class, POINT);
 *
 *  // Extracts a new Point record from the provided MemorySegment
 *  Point point = recordMapper.get(segment); // Point[x=3, y=4]
 *
 *  // Writes the Point record to another MemorySegment
 *  MemorySegment otherSegment = Arena.ofAuto().allocate(MemoryLayout.sequenceLayout(2, POINT));
 *  recordMapper.setAtIndex(otherSegment, 1, point); // segment: 0, 0, 3, 4
 *}
 * <p>
 * Boxing, widening, narrowing and general type conversion must be explicitly handled by
 * user code. In the following example, the above {@code Point} (using primitive
 * {@code int x} and {@code int y} coordinates) are explicitly mapped to a narrowed
 * point type (instead using primitive {@code byte x} and {@code byte y} coordinates):
 * <p>
 * {@snippet lang = java:
 * public record NarrowedPoint(byte x, byte y) {
 *
 *     static NarrowedPoint fromPoint(Point p) {
 *         return new NarrowedPoint((byte) p.x, (byte) p.y);
 *     }
 *
 *     static Point toPoint(NarrowedPoint p) {
 *         return new Point(p.x, p.y);
 *     }
 *
 * }
 *
 * SegmentMapper<NarrowedPoint> narrowedPointMapper =
 *         SegmentMapper.ofRecord(Point.class, POINT)              // SegmentMapper<Point>
 *         .map(NarrowedPoint.class, NarrowedPoint::fromPoint, NarrowedPoint::toPoint); // SegmentMapper<NarrowedPoint>
 *
 * // Extracts a new NarrowedPoint from the provided MemorySegment
 * NarrowedPoint narrowedPoint = narrowedPointMapper.get(segment); // NarrowedPoint[x=3, y=4]
 * }
 *
 * <h2 id="mapping-interfaces">Mapping Interfaces</h2>
 *
 * Here is another example showing how to extract an instance of a public
 * <em>interface with an external segment</em>:
 * {@snippet lang = java:
 *
 *  static final GroupLayout POINT = MemoryLayout.structLayout(JAVA_INT.withName("x"), JAVA_INT.withName("y"));
 *
 *  public interface Point {
 *       int x();
 *       void x(int x);
 *       int y();
 *       void y(int x);
 *  }
 *
 *  //...
 *
 *  MemorySegment segment = MemorySegment.ofArray(new int[]{3, 4, 0, 0});
 *
 *  SegmentMapper<Point> mapper = SegmentMapper.of(MethodHandles.lookup(), Point.class, POINT);
 *
 *  // Creates a new Point interface instance with an external segment
 *  Point point = mapper.get(segment); // Point[x=3, y=4]
 *  point.x(6); // Point[x=6, y=4]
 *  point.y(8); // Point[x=6, y=8]
 *
 *  MemorySegment otherSegment = Arena.ofAuto().allocate(MemoryLayout.sequenceLayout(2, POINT)); // otherSegment: 0, 0, 0, 0
 *  mapper.setAtIndex(otherSegment, 1, point); // segment: 0, 0, 6, 8
 *}
 *}
 * <p>
 * Boxing, widening, narrowing and general type conversion must be explicitly handled
 * by user code. In the following example, the above {@code PointAccessor} interface
 * (using primitive {@code int x} and {@code int y} coordinates) are explicitly mapped to
 * a narrowed point type (instead using primitive {@code byte x} and
 * {@code byte y} coordinates):
 * <p>
 * {@snippet lang = java:
 * interface NarrowedPointAccessor {
 *    byte x();
 *    void x(byte x);
 *    byte y();
 *    void y(byte y);
 *
 *    static NarrowedPointAccessor fromPointAccessor(PointAccessor pa) {
 *        return new NarrowedPointAccessor() {
 *            @Override public byte x()       { return (byte)pa.x(); }
 *            @Override public void x(byte x) { pa.x(x); }
 *            @Override public byte y()       { return (byte) pa.y();}
 *            @Override public void y(byte y) { pa.y(y); }
 *       };
 *    }
 *
 * }
 *
 * SegmentMapper<NarrowedPointAccessor> narrowedPointMapper =
 *          // SegmentMapper<PointAccessor>
 *           SegmentMapper.ofInterface(MethodHandles.lookup(), PointAccessor.class, POINT)
 *                   // SegmentMapper<NarrowedPointAccessor>
 *                  .map(NarrowedPointAccessor.class, NarrowedPointAccessor::fromPointAccessor);
 *
 * MemorySegment segment = MemorySegment.ofArray(new int[]{3, 4});
 *
 * // Creates a new NarrowedPointAccessor from the provided MemorySegment
 * NarrowedPointAccessor narrowedPointAccessor = narrowedPointMapper.get(segment); // NarrowedPointAccessor[x=3, y=4]
 *
 * MemorySegment otherSegment = Arena.ofAuto().allocate(MemoryLayout.sequenceLayout(2, POINT));
 * narrowedPointMapper.setAtIndex(otherSegment, 1, narrowedPointAccessor); // otherSegment = 0, 0, 3, 4
 *}
 *
 * <h2 id="segment-exposure">Backing segment exposure</h2>
 *
 * Implementations of interfaces that are obtained via segment mappers can be made to
 * reveal the underlying memory segment and memory segment offset. This is useful when
 * modelling structs that are passed and/or received by native calls:
 * <p>
 * {@snippet lang = java:
 * static final GroupLayout POINT = MemoryLayout.structLayout(JAVA_INT.withName("x"), JAVA_INT.withName("y"));
 *
 * public interface PointAccessor {
 *     int x();
 *     void x(int x);
 *     int y();
 *     void y(int x);
 * }
 *
 * static double nativeDistance(MemorySegment pointStruct) {
 *     // Calls a native method
 *     // ...
 * }
 *
 * public static void main(String[] args) {
 *
 *     SegmentMapper<PointAccessor> mapper =
 *             SegmentMapper.of(MethodHandles.lookup(), PointAccessor.class, POINT);
 *
 *     try (Arena arena = Arena.ofConfined()){
 *         // Creates an interface mapper backed by an internal segment
 *         PointAccessor point = mapper.get(arena);
 *         point.x(3);
 *         point.y(4);
 *
 *         // Pass the backing internal segment to a native method
 *         double distance = nativeDistance(mapper.segment(point).orElseThrow()); // 5
 *     }
 *
 * }
 *}
 *
 * <h2 id="formal-mapping">Formal mapping description</h2>
 *
 * Components and layouts are matched with respect to their name and the exact return type and/or
 * the exact parameter types. No widening or narrowing is employed.
 *
 * <h2 id="restrictions">Restrictions</h2>
 *
 * Generic interfaces need to have their generic type parameters (if any)
 * know at compile time. This applies to all extended interfaces recursively.
 * <p>
 * Interfaces and records must not implement (directly and/or via inheritance) more than
 * one abstract method with the same name and erased parameter types. Hence, covariant
 * overriding is not supported.
 *
 * @param <T> the type this mapper converts MemorySegments from and to.
 *
 * @implSpec Implementations of this interface are immutable, thread-safe and
 *           <a href="{@docRoot}/java.base/java/lang/doc-files/ValueBased.html">value-based</a>.
 *
 * @since 23
 */

// Todo: Map components to MemorySegment (escape hatch)
// Todo: How do we handle "extra" setters for interfaces? They should not append

// Cerializer
// Todo: Check all exceptions in JavaDocs: See TestScopedOperations
// Todo: Consider generating a graphics rendering.
// Todo: Add in doc that getting via an AddressValue will return a MS managed by Arena.global()
// Todo: Provide safe sharing across threads (e.g. implement a special Interface with piggybacking/volatile access)
// Todo: Prevent several variants in a record from being mapped to a union (otherwise, which will "win" when writing?)
// Todo: There seams to be a problem with the ByteOrder in the mapper. See TestJepExamplesUnions
// Todo: Let SegmentMapper::getHandle and ::setHandle return the sharp types (e.g. Point) see MethodHandles::exactInvoker

// Done: The generated interface classes should be @ValueBased
// Done: Python "Pandas" (tables), Tabular access from array, Joins etc. <- TEST
//       -> See TestDataProcessingRecord and TestDataProcessingInterface
// No: ~map() can be dropped in favour of "manual mapping"~
// Done: Interfaces with internal segments should be directly available via separate factory methods
//       -> Fixed via SegmentMapper::create
// Done: Discuss if an exception is thrown in one of the sub-setters... This means partial update of the MS
//       This can be fixed using double-buffering. Maybe provide a scratch segment somehow that tracks where writes
//       has been made (via a separate class BufferedMapper?)
//       -> Fixed via TestInterfaceMapper::doubleBuffered
public interface SegmentMapper<T> {

    /**
     * {@return the type that this mapper is mapping to and from}
     */
    Class<T> type();

    /**
     * {@return the original {@link GroupLayout } that this mapper is using to map
     *          record components or interface methods}
     * <p>
     * Composed segment mappers (obtained via either the {@link SegmentMapper#map(Class, Function)}
     * or the {@link SegmentMapper#map(Class, Function)} will still return the
     * group layout from the <em>original</em> SegmentMapper.
     */
    GroupLayout layout();

    // Convenience methods

    /**
     * {@return a new instance of type T projected at an internal {@code segment} at
     *          offset zero created by means of invoking the provided {@code arena}}
     * <p>
     * Calling this method is equivalent to the following code:
     * {@snippet lang = java:
     *    get(arena.allocate(layout()));
     * }
     *
     * @param  arena from which to {@linkplain Arena#allocate(MemoryLayout) allocate} an
     *         internal memory segment.
     * @throws IllegalStateException if the {@linkplain MemorySegment#scope() scope}
     *         associated with the provided segment is not
     *         {@linkplain MemorySegment.Scope#isAlive() alive}
     * @throws WrongThreadException if this method is called from a thread {@code T},
     *         such that {@code isAccessibleBy(T) == false}
     * @throws IllegalArgumentException if the access operation is
     *         <a href="MemorySegment.html#segment-alignment">incompatible with the alignment constraint</a>
     *         of the {@link #layout()}
     * @throws IndexOutOfBoundsException if
     *         {@code layout().byteSize() > segment.byteSize()}
     */
    default T allocate(Arena arena) {

        return get(arena.allocate(layout()),layout());
    }
    /**
     * {@return a new instance of type T projected at the provided
     *          external {@code segment} at offset zero}
     * <p>
     * Calling this method is equivalent to the following code:
     * {@snippet lang = java:
     *    get(segment, 0L);
     * }
     *
     * @param segment the external segment to be projected to the new instance
     * @throws IllegalStateException if the {@linkplain MemorySegment#scope() scope}
     *         associated with the provided segment is not
     *         {@linkplain MemorySegment.Scope#isAlive() alive}
     * @throws WrongThreadException if this method is called from a thread {@code T},
     *         such that {@code isAccessibleBy(T) == false}
     * @throws IllegalArgumentException if the access operation is
     *         <a href="MemorySegment.html#segment-alignment">incompatible with the alignment constraint</a>
     *         of the {@link #layout()}
     * @throws IndexOutOfBoundsException if
     *         {@code layout().byteSize() > segment.byteSize()}
     */
    default T get(MemorySegment segment) {
        return get(segment, 0L);
    }
    default T get(MemorySegment segment, GroupLayout groupLayout) {
        return get(segment, groupLayout, 0L);
    }
    /**
     * {@return a new instance of type T projected at the provided external
     *          {@code segment} at the given {@code index} scaled by the
     *          {@code layout().byteSize()}}
     * <p>
     * Calling this method is equivalent to the following code:
     * {@snippet lang = java:
     *    get(segment, layout().byteSize() * index);
     * }
     *
     * @param segment the external segment to be projected to the new instance
     * @param index a logical index, the offset in bytes (relative to the provided
     *              segment address) at which the access operation will occur can
     *              be expressed as {@code (index * layout().byteSize())}
     * @throws IllegalStateException if the {@linkplain MemorySegment#scope() scope}
     *         associated with the provided segment is not
     *         {@linkplain MemorySegment.Scope#isAlive() alive}
     * @throws WrongThreadException if this method is called from a thread {@code T},
     *         such that {@code isAccessibleBy(T) == false}
     * @throws IllegalArgumentException if the access operation is
     *         <a href="MemorySegment.html#segment-alignment">incompatible with the alignment constraint</a>
     *         of the {@link #layout()}
     * @throws IndexOutOfBoundsException if {@code index * layout().byteSize()} overflows
     * @throws IndexOutOfBoundsException if
     *         {@code index * layout().byteSize() > segment.byteSize() - layout.byteSize()}
     */
    default T getAtIndex(MemorySegment segment, long index) {
        return get(segment, layout().byteSize() * index);
    }

    /**
     * {@return a new sequential {@code Stream} of elements of type T}
     * <p>
     * Calling this method is equivalent to the following code:
     * {@snippet lang=java :
     * segment.elements(layout())
     *     .map(this::get);
     * }
     * @param segment to carve out instances from
     * @throws IllegalArgumentException if {@code layout().byteSize() == 0}.
     * @throws IllegalArgumentException if {@code segment.byteSize() % layout().byteSize() != 0}.
     * @throws IllegalArgumentException if {@code layout().byteSize() % layout().byteAlignment() != 0}.
     * @throws IllegalArgumentException if this segment is
     *         <a href="MemorySegment.html#segment-alignment">incompatible with the
     *         alignment constraint</a> in the layout of this segment mapper.
     */
    default Stream<T> stream(MemorySegment segment) {
        return segment.elements(layout())
                .map(this::get);
    }

    /**
     * {@return a new sequential {@code Stream} of {@code pageSize} elements of
     *          type T starting at the element {@code pageNumber * pageSize}}
     * <p>
     * Calling this method is equivalent to the following code:
     * {@snippet lang=java :
     * stream(segment)
     *     .skip(pageNumber * pageSize)
     *     .limit(pageSize);
     * }
     * but may be much more efficient for large page numbers.
     *
     * @param segment    to carve out instances from
     * @param pageSize   the size of each page
     * @param pageNumber the page number to which to skip
     * @throws IllegalArgumentException if {@code layout().byteSize() == 0}.
     * @throws IllegalArgumentException if {@code segment.byteSize() % layout().byteSize() != 0}.
     * @throws IllegalArgumentException if {@code layout().byteSize() % layout().byteAlignment() != 0}.
     * @throws IllegalArgumentException if this segment is
     *         <a href="MemorySegment.html#segment-alignment">incompatible with the
     *         alignment constraint</a> in the layout of this segment mapper.
     */
    default Stream<T> page(MemorySegment segment,
                           long pageSize,
                           long pageNumber) {
        long skipBytes = Math.min(segment.byteSize(), layout().scale(0, pageNumber * pageSize));
        MemorySegment skippedSegment = segment.asSlice(skipBytes);
        return stream(skippedSegment)
                .limit(pageSize);
    }

    /**
     * {@return a new instance of type T projected from at provided
     *          external {@code segment} at the provided {@code offset}}
     *
     * @param segment the external segment to be projected at the new instance
     * @param offset  from where in the segment to project the new instance
     * @throws IllegalStateException if the {@linkplain MemorySegment#scope() scope}
     *         associated with the provided segment is not
     *         {@linkplain MemorySegment.Scope#isAlive() alive}
     * @throws WrongThreadException if this method is called from a thread {@code T},
     *         such that {@code isAccessibleBy(T) == false}
     * @throws IllegalArgumentException if the access operation is
     *         <a href="MemorySegment.html#segment-alignment">incompatible with the alignment constraint</a>
     *         of the {@link #layout()}
     * @throws IndexOutOfBoundsException if
     *         {@code offset > segment.byteSize() - layout().byteSize()}
     */
    @SuppressWarnings("unchecked")
    default T get(MemorySegment segment, long offset) {
        try {
            return (T) getHandle()
                    .invokeExact(segment,offset);
        } catch (NullPointerException |
                 IndexOutOfBoundsException |
                 WrongThreadException |
                 IllegalStateException |
                 IllegalArgumentException rethrow) {
            throw rethrow;
        } catch (Throwable e) {
            throw new RuntimeException("Unable to invoke getHandle() with " +
                    "segment="  + segment +
                    ", offset=" + offset, e);
        }
    }

    @SuppressWarnings("unchecked")
    default T get(MemorySegment segment, GroupLayout layout,long offset) {
        try {
            return (T) getHandle()
                    .invokeExact(segment, layout, offset);
        } catch (NullPointerException |
                 IndexOutOfBoundsException |
                 WrongThreadException |
                 IllegalStateException |
                 IllegalArgumentException rethrow) {
            throw rethrow;
        } catch (Throwable e) {
            throw new RuntimeException("Unable to invoke getHandle() with " +
                    "segment="  + segment +
                    ", offset=" + offset, e);
        }
    }

    /**
     * Writes the provided instance {@code t} of type T into the provided {@code segment}
     * at offset zero.
     * <p>
     * Calling this method is equivalent to the following code:
     * {@snippet lang = java:
     *    set(segment, 0L, t);
     * }
     *
     * @param segment in which to write the provided {@code t}
     * @param t instance to write into the provided segment
     * @throws IllegalStateException if the {@linkplain MemorySegment#scope() scope}
     *         associated with this segment is not
     *         {@linkplain MemorySegment.Scope#isAlive() alive}
     * @throws WrongThreadException if this method is called from a thread {@code T},
     *         such that {@code isAccessibleBy(T) == false}
     * @throws IllegalArgumentException if the access operation is
     *         <a href="MemorySegment.html#segment-alignment">incompatible with the alignment constraint</a>
     *         of the {@link #layout()}
     * @throws IndexOutOfBoundsException if {@code layout().byteSize() > segment.byteSize()}
     * @throws UnsupportedOperationException if this segment is
     *         {@linkplain MemorySegment#isReadOnly() read-only}
     * @throws UnsupportedOperationException if {@code value} is not a
     *         {@linkplain MemorySegment#isNative() native} segment
     * @throws IllegalArgumentException if an array length does not correspond to the
     *         {@linkplain SequenceLayout#elementCount() element count} of a sequence layout
     * @throws NullPointerException if a required parameter is {@code null}
     */
    default void set(MemorySegment segment, T t) {
        set(segment, 0L, t);
    }

    /**
     * Writes the provided {@code t} instance of type T into the provided {@code segment}
     * at the provided {@code index} scaled by the {@code layout().byteSize()}}.
     * <p>
     * Calling this method is equivalent to the following code:
     * {@snippet lang = java:
     *    set(segment, layout().byteSize() * index, t);
     * }
     * @param segment in which to write the provided {@code t}
     * @param index a logical index, the offset in bytes (relative to the provided
     *              segment address) at which the access operation will occur can be
     *              expressed as {@code (index * layout().byteSize())}
     * @param t instance to write into the provided segment
     * @throws IllegalStateException if the {@linkplain MemorySegment#scope() scope}
     *         associated with this segment is not
     *         {@linkplain MemorySegment.Scope#isAlive() alive}
     * @throws WrongThreadException if this method is called from a thread {@code T},
     *         such that {@code isAccessibleBy(T) == false}
     * @throws IllegalArgumentException if the access operation is
     *         <a href="MemorySegment.html#segment-alignment">incompatible with the alignment constraint</a>
     *         of the {@link #layout()}
     * @throws IndexOutOfBoundsException if {@code offset > segment.byteSize() - layout.byteSize()}
     * @throws UnsupportedOperationException if this segment is
     *         {@linkplain MemorySegment#isReadOnly() read-only}
     * @throws UnsupportedOperationException if {@code value} is not a
     *         {@linkplain MemorySegment#isNative() native} segment
     * @throws IllegalArgumentException if an array length does not correspond to the
     *         {@linkplain SequenceLayout#elementCount() element count} of a sequence layout
     * @throws NullPointerException if a required parameter is {@code null}
     */
    default void setAtIndex(MemorySegment segment, long index, T t) {
        set(segment, layout().byteSize() * index, t);
    }

    /**
     * Writes the provided instance {@code t} of type T into the provided {@code segment}
     * at the provided {@code offset}.
     *
     * @param segment in which to write the provided {@code t}
     * @param offset offset in bytes (relative to the provided segment address) at which
     *               this access operation will occur
     * @param t instance to write into the provided segment
     * @throws IllegalStateException if the {@linkplain MemorySegment#scope() scope}
     *         associated with this segment is not
     *         {@linkplain MemorySegment.Scope#isAlive() alive}
     * @throws WrongThreadException if this method is called from a thread {@code T},
     *         such that {@code isAccessibleBy(T) == false}
     * @throws IllegalArgumentException if the access operation is
     *         <a href="MemorySegment.html#segment-alignment">incompatible with the alignment constraint</a>
     *         of the {@link #layout()}
     * @throws IndexOutOfBoundsException if {@code offset > segment.byteSize() - layout.byteSize()}
     * @throws UnsupportedOperationException if
     *         this segment is {@linkplain MemorySegment#isReadOnly() read-only}
     * @throws UnsupportedOperationException if
     *         {@code value} is not a {@linkplain MemorySegment#isNative() native} segment // Todo: only for pointers
     * @throws IllegalArgumentException if an array length does not correspond to the
     *         {@linkplain SequenceLayout#elementCount() element count} of a sequence layout
     * @throws NullPointerException if a required parameter is {@code null}
     */
    default void set(MemorySegment segment, long offset, T t) {
        try {
            setHandle()
                    .invokeExact(segment, offset, (Object) t);
        } catch (IndexOutOfBoundsException |
                 WrongThreadException |
                 IllegalStateException |
                 IllegalArgumentException |
                 UnsupportedOperationException |
                 NullPointerException rethrow) {
            throw rethrow;
        } catch (Throwable e) {
            throw new RuntimeException("Unable to invoke setHandle() with " +
                    "segment=" + segment +
                    ", offset=" + offset +
                    ", t=" + t, e);
        }
    }

    // Basic methods

    /**
     * {@return a method handle that returns new instances of type T projected at
     *          a provided external {@code MemorySegment} at a provided {@code long} offset}
     * <p>
     * The returned method handle has the following characteristics:
     * <ul>
     *     <li>its return type is {@code T};</li>
     *     <li>it has a leading parameter of type {@code MemorySegment}
     *         corresponding to the memory segment to be accessed</li>
     *     <li>it has a trailing {@code long} parameter, corresponding to
     *         the base offset</li>
     * </ul>
     *
     * @see #get(MemorySegment, long)
     */
    MethodHandle getHandle();

    /**
     * {@return a method handle that writes a provided instance of type T into
     *          a provided {@code MemorySegment} at a provided {@code long} offset}
     * <p>
     * The returned method handle has the following characteristics:
     * <ul>
     *     <li>its return type is void;</li>
     *     <li>it has a leading parameter of type {@code MemorySegment}
     *         corresponding to the memory segment to be accessed</li>
     *     <li>it has a following {@code long} parameter, corresponding to
     *         the base offset</li>
     *     <li>it has a trailing {@code T} parameter, corresponding to
     *         the value to set</li>
     * </ul>
     *
     * @see #set(MemorySegment, long, Object)
     */
    MethodHandle setHandle();

    /**
     * {@return a new segment mapper that would apply the provided {@code toMapper} after
     *          performing get operations on this segment mapper and that would throw an
     *          {@linkplain UnsupportedOperationException} for set operations if this
     *          segment mapper is a record mapper}
     * <p>
     * It should be noted that the type R can represent almost any class and is not
     * restricted to records and interfaces.
     * <p>
     * Interface segment mappers returned by this method does not support
     * {@linkplain #set(MemorySegment, Object) set} operations.
     *
     * @param  newType the new type the returned mapper shall use
     * @param toMapper to apply after get operations on this segment mapper
     * @param <R> the type of the new segment mapper
     */
    <R> SegmentMapper<R> map(Class<R> newType,
                             Function<? super T, ? extends R> toMapper);

    /**
     * {@return the backing segment of the provided {@code source},
     *          or, if no backing segment exists, {@linkplain Optional#empty()}}
     * <p>
     * Interfaces obtained from segment mappers have backing segments. Records obtained
     * from segment mappers do not.
     *
     * @param source from which to extract the backing segment
     */
    default Optional<MemorySegment> segment(T source) {
        Objects.requireNonNull(source);
        return Optional.empty();
    }

    /**
     * {@return the offset in the backing segment of the provided {@code source},
     *          or, if no backing segment exists, {@linkplain OptionalLong#empty()}}
     * <p>
     * Interfaces obtained from segment mappers have backing segments. Records obtained
     * from segment mappers do not.
     *
     * @param source from which to extract the backing segment
     */
    default OptionalLong offset(T source) {
        Objects.requireNonNull(source);
        return OptionalLong.empty();
    }

    /**
     * {@return a segment mapper that maps {@linkplain MemorySegment memory segments}
     *          to the provided interface {@code type} using the provided {@code layout}
     *          and using the provided {@code lookup}}
     *
     * @implNote The order in which methods appear (e.g. in the {@code toString} method)
     *           is derived from the provided {@code layout}.
     *
     * @implNote The returned class can be a
     *           <a href="{@docRoot}/java.base/java/lang/doc-files/ValueBased.html">value-based</a>
     *           class; programmers should treat instances that are
     *           {@linkplain Object#equals(Object) equal} as interchangeable and should
     *           not use instances for synchronization, or unpredictable behavior may
     *           occur. For example, in a future release, synchronization may fail.
     *
     * @implNote The returned class can be a {@linkplain Class#isHidden() hidden} class.
     *
     * @param lookup to use when performing reflective analysis on the
     *               provided {@code type}
     * @param type to map memory segment from and to
     * @param layout to be used when mapping the provided {@code type}
     * @param <T> the type the returned mapper converts MemorySegments from and to
     * @throws IllegalArgumentException if the provided {@code type} is not an interface
     * @throws IllegalArgumentException if the provided {@code type} is a hidden interface
     * @throws IllegalArgumentException if the provided {@code type} is a sealed interface
     * @throws IllegalArgumentException if the provided interface {@code type} directly
     *         declares any generic type parameter
     * @throws IllegalArgumentException if the provided interface {@code type} cannot be
     *         reflectively analysed using the provided {@code lookup}
     * @throws IllegalArgumentException if the provided interface {@code type} contains
     *         methods for which there are no exact mapping (of names and types) in
     *         the provided {@code layout} or if the provided {@code type} is not public or
     *         if the method is otherwise unable to create a segment mapper as specified above
     */
    static <T> SegmentMapper<T> of(MethodHandles.Lookup lookup,
                                   Class<T> type,
                                   GroupLayout layout) {
        Objects.requireNonNull(lookup);
        MapperUtil.requireImplementableInterfaceType(type);
        Objects.requireNonNull(layout);
        return SegmentInterfaceMapper.create(lookup, type, layout);
    }


    static <T> SegmentMapper<T> of(MethodHandles.Lookup lookup,
                                   Class<T> type,
                                   MemoryLayout ... elements) {

        StructLayout structlayout = MemoryLayout.structLayout(elements).withName(type.getSimpleName());
        return of(lookup,type, structlayout);
    }


    /**
     * Interfaces extending this interface will be provided
     * with additional methods for discovering the backing
     * memory segment and offset used as the backing storage.
     */
    interface Discoverable {

        /**
         * {@return the backing segment of this instance}
         */
        MemorySegment segment();

        /**
         * {@return the offset in the backing segment of this instance}
         */
        long offset();
    }

}
