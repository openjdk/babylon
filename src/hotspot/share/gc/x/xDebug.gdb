#
# GDB functions for debugging the Z Garbage Collector
#

printf "Loading zDebug.gdb\n"

# Print Klass*
define zpk
    printf "Klass: %s\n", (char*)((Klass*)($arg0))->_name->_body
end

# Print oop
define zpo
    set $obj = (oopDesc*)($arg0)

    printf "Oop:   0x%016llx\tState: ", (uintptr_t)$obj
    if ((uintptr_t)$obj & (uintptr_t)XAddressGoodMask)
        printf "Good "
        if ((uintptr_t)$obj & (uintptr_t)XAddressMetadataRemapped)
            printf "(Remapped)"
        else
            if ((uintptr_t)$obj & (uintptr_t)XAddressMetadataMarked)
                printf "(Marked)"
            else
                printf "(Unknown)"
            end
        end
    else
        printf "Bad "
        if ((uintptr_t)XAddressGoodMask & (uintptr_t)XAddressMetadataMarked)
            # Should be marked
            if ((uintptr_t)$obj & (uintptr_t)XAddressMetadataRemapped)
                printf "(Not Marked, Remapped)"
            else
                printf "(Not Marked, Not Remapped)"
            end
        else
            if ((uintptr_t)XAddressGoodMask & (uintptr_t)XAddressMetadataRemapped)
                # Should be remapped
                if ((uintptr_t)$obj & (uintptr_t)XAddressMetadataMarked)
                    printf "(Marked, Not Remapped)"
                else
                    printf "(Not Marked, Not Remapped)"
                end
            else
                # Unknown
                printf "(Unknown)"
            end
        end
    end
    printf "\t Page: %llu\n", ((uintptr_t)$obj & XAddressOffsetMask) >> XGranuleSizeShift
    x/16gx $obj
    if (UseCompressedClassPointers)
        set $klass = (Klass*)(void*)((uintptr_t)CompressedKlassPointers::_base +((uintptr_t)$obj->_metadata->_compressed_klass << CompressedKlassPointers::_shift))
    else
        set $klass = $obj->_metadata->_klass
    end
    printf "Mark:  0x%016llx\tKlass: %s\n", (uintptr_t)$obj->_mark, (char*)$klass->_name->_body
end

# Print heap page by page table index
define zpp
    set $page = (XPage*)((uintptr_t)XHeap::_heap._page_table._map._map[($arg0)] & ~1)
    printf "Page %p\n", $page
    print *$page
end

# Print page_table
define zpt
    printf "Pagetable (first 128 slots)\n"
    x/128gx XHeap::_heap._page_table._map._map
end

# Print live map
define __zmarked
    set $livemap   = $arg0
    set $bit        = $arg1
    set $size       = $livemap._bitmap._size
    set $segment    = $size / XLiveMap::nsegments
    set $segment_bit = 1 << $segment

    printf "Segment is "
    if !($livemap._segment_live_bits & $segment_bit)
        printf "NOT "
    end
    printf "live (segment %d)\n", $segment

    if $bit >= $size
        print "Error: Bit %z out of bounds (bitmap size %z)\n", $bit, $size
    else
        set $word_index = $bit / 64
        set $bit_index  = $bit % 64
        set $word       = $livemap._bitmap._map[$word_index]
        set $live_bit   = $word & (1 << $bit_index)

        printf "Object is "
        if $live_bit == 0
            printf "NOT "
        end
        printf "live (word index %d, bit index %d)\n", $word_index, $bit_index
    end
end

define zmarked
    set $addr          = $arg0
    set $obj           = ((uintptr_t)$addr & XAddressOffsetMask)
    set $page_index    = $obj >> XGranuleSizeShift
    set $page_entry    = (uintptr_t)XHeap::_heap._page_table._map._map[$page_index]
    set $page          = (XPage*)($page_entry & ~1)
    set $page_start    = (uintptr_t)$page._virtual._start
    set $page_end      = (uintptr_t)$page._virtual._end
    set $page_seqnum   = $page._livemap._seqnum
    set $global_seqnum = XGlobalSeqNum

    if $obj < $page_start || $obj >= $page_end
        printf "Error: %p not in page %p (start %p, end %p)\n", $obj, $page, $page_start, $page_end
    else
        printf "Page is "
        if $page_seqnum != $global_seqnum
            printf "NOT "
        end
        printf "live (page %p, page seqnum %d, global seqnum %d)\n", $page, $page_seqnum, $global_seqnum

        #if $page_seqnum == $global_seqnum
            set $offset = $obj - $page_start
            set $bit = $offset / 8
            __zmarked $page._livemap $bit
        #end
    end
end

# Print heap information
define zph
    printf "Heap\n"
    printf "     GlobalPhase:       %u\n", XGlobalPhase
    printf "     GlobalSeqNum:      %u\n", XGlobalSeqNum
    printf "     Offset Max:        %-15llu (0x%llx)\n", XAddressOffsetMax, XAddressOffsetMax
    printf "     Page Size Small:   %-15llu (0x%llx)\n", XPageSizeSmall, XPageSizeSmall
    printf "     Page Size Medium:  %-15llu (0x%llx)\n", XPageSizeMedium, XPageSizeMedium
    printf "Metadata Bits\n"
    printf "     Good:              0x%016llx\n", XAddressGoodMask
    printf "     Bad:               0x%016llx\n", XAddressBadMask
    printf "     WeakBad:           0x%016llx\n", XAddressWeakBadMask
    printf "     Marked:            0x%016llx\n", XAddressMetadataMarked
    printf "     Remapped:          0x%016llx\n", XAddressMetadataRemapped
end

# End of file
