#![no_std]

use allocator::{BaseAllocator, ByteAllocator, PageAllocator};

/// Early memory allocator
/// Use it before formal bytes-allocator and pages-allocator can work!
/// This is a double-end memory range:
/// - Alloc bytes forward
/// - Alloc pages backward
///
/// [ bytes-used | avail-area | pages-used ]
/// |            | -->    <-- |            |
/// start       b_pos        p_pos       end
///
/// For bytes area, 'count' records number of allocations.
/// When it goes down to ZERO, free bytes-used area.
/// For pages area, it will never be freed!
///
pub struct EarlyAllocator<const SIZE: usize> {
    start: usize,
    end: usize,
    b_pos: usize,
    p_pos: usize,
    count: usize,
    page_count: usize,
}

impl<const SIZE: usize> EarlyAllocator<SIZE> {
    pub const fn new() -> Self {
        Self {
            start: 0,
            end: 0,
            b_pos: 0,
            p_pos: 0,
            count: 0,
            page_count: 0,
        }
    }
}

impl<const SIZE: usize> BaseAllocator for EarlyAllocator<SIZE> {
    fn init(&mut self, start: usize, size: usize) {
        self.start = start;
        self.end = start + size;
        self.b_pos = start;
        self.p_pos = start + size;
        self.count = 0;
        self.page_count = 0;
    }

    fn add_memory(&mut self, start: usize, size: usize) -> allocator::AllocResult {
        // For bump allocator, extended memory only works if previous allocations
        // are absent (count == 0 and page_count == 0).
        if self.count == 0 && self.page_count == 0 {
            self.start = start;
            self.end = start + size;
            self.b_pos = start;
            self.p_pos = start + size;
            Ok(())
        } else {
            Err(allocator::AllocError::NoMemory)
        }
    }
}

impl<const SIZE: usize> ByteAllocator for EarlyAllocator<SIZE> {
    fn alloc(
        &mut self,
        layout: core::alloc::Layout,
    ) -> allocator::AllocResult<core::ptr::NonNull<u8>> {
        let align = layout.align();
        // Align b_pos up to alignment
        let alloc_start = (self.b_pos + align - 1) & !(align - 1);
        let alloc_end = alloc_start + layout.size();

        if alloc_end > self.p_pos {
            return Err(allocator::AllocError::NoMemory);
        }

        self.b_pos = alloc_end;
        self.count += 1;

        core::ptr::NonNull::new(alloc_start as *mut u8)
            .ok_or(allocator::AllocError::NoMemory)
    }

    fn dealloc(&mut self, _pos: core::ptr::NonNull<u8>, _layout: core::alloc::Layout) {
        self.count -= 1;
        if self.count == 0 {
            // Reset byte area entirely
            self.b_pos = self.start;
        }
    }

    fn total_bytes(&self) -> usize {
        self.end - self.start
    }

    fn used_bytes(&self) -> usize {
        self.b_pos - self.start
    }

    fn available_bytes(&self) -> usize {
        self.p_pos - self.b_pos
    }
}

impl<const SIZE: usize> PageAllocator for EarlyAllocator<SIZE> {
    const PAGE_SIZE: usize = SIZE;

    fn alloc_pages(
        &mut self,
        num_pages: usize,
        align_pow2: usize,
    ) -> allocator::AllocResult<usize> {
        let size = num_pages * Self::PAGE_SIZE;

        // Align p_pos down to align_pow2 boundary (pages grow backward)
        let alloc_end = self.p_pos & !(align_pow2 - 1);

        if alloc_end < size || alloc_end - size < self.b_pos {
            return Err(allocator::AllocError::NoMemory);
        }

        self.p_pos = alloc_end - size;
        self.page_count += num_pages;

        Ok(self.p_pos)
    }

    fn dealloc_pages(&mut self, _pos: usize, _num_pages: usize) {
        // Pages never freed
    }

    fn total_pages(&self) -> usize {
        (self.end - self.start) / Self::PAGE_SIZE
    }

    fn used_pages(&self) -> usize {
        self.page_count
    }

    fn available_pages(&self) -> usize {
        (self.p_pos - self.b_pos) / Self::PAGE_SIZE
    }
}
