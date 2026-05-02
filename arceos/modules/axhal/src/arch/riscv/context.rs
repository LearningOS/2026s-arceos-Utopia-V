use core::arch::asm;
use memory_addr::VirtAddr;
#[cfg(feature = "uspace")]
use memory_addr::PhysAddr;

include_asm_marcos!();

/// General registers of RISC-V.
#[allow(missing_docs)]
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct GeneralRegisters {
    pub ra: usize,
    pub sp: usize,
    pub gp: usize, // only valid for user traps
    pub tp: usize, // only valid for user traps
    pub t0: usize,
    pub t1: usize,
    pub t2: usize,
    pub s0: usize,
    pub s1: usize,
    pub a0: usize,
    pub a1: usize,
    pub a2: usize,
    pub a3: usize,
    pub a4: usize,
    pub a5: usize,
    pub a6: usize,
    pub a7: usize,
    pub s2: usize,
    pub s3: usize,
    pub s4: usize,
    pub s5: usize,
    pub s6: usize,
    pub s7: usize,
    pub s8: usize,
    pub s9: usize,
    pub s10: usize,
    pub s11: usize,
    pub t3: usize,
    pub t4: usize,
    pub t5: usize,
    pub t6: usize,
}

/// Saved registers when a trap (interrupt or exception) occurs.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct TrapFrame {
    /// All general registers.
    pub regs: GeneralRegisters,
    /// Supervisor Exception Program Counter.
    pub sepc: usize,
    /// Supervisor Status Register.
    pub sstatus: usize,
}

impl TrapFrame {
    /// Gets the 0th syscall argument.
    pub const fn arg0(&self) -> usize {
        self.regs.a0
    }

    /// Gets the 1st syscall argument.
    pub const fn arg1(&self) -> usize {
        self.regs.a1
    }

    /// Gets the 2nd syscall argument.
    pub const fn arg2(&self) -> usize {
        self.regs.a2
    }

    /// Gets the 3rd syscall argument.
    pub const fn arg3(&self) -> usize {
        self.regs.a3
    }

    /// Gets the 4th syscall argument.
    pub const fn arg4(&self) -> usize {
        self.regs.a4
    }

    /// Gets the 5th syscall argument.
    pub const fn arg5(&self) -> usize {
        self.regs.a5
    }
}

/// Saved hardware states of a task.
///
/// The context usually includes:
///
/// - Callee-saved registers
/// - Stack pointer register
/// - Thread pointer register (for thread-local storage, currently unsupported)
/// - FP/SIMD registers
///
/// On context switch, current task saves its context from CPU to memory,
/// and the next task restores its context from memory to CPU.
#[allow(missing_docs)]
#[repr(C)]
#[derive(Debug, Default)]
pub struct TaskContext {
    pub ra: usize, // return address (x1)
    pub sp: usize, // stack pointer (x2)

    pub s0: usize, // x8-x9
    pub s1: usize,

    pub s2: usize, // x18-x27
    pub s3: usize,
    pub s4: usize,
    pub s5: usize,
    pub s6: usize,
    pub s7: usize,
    pub s8: usize,
    pub s9: usize,
    pub s10: usize,
    pub s11: usize,

    pub tp: usize,
    /// The `satp` register value, i.e., the page table root.
    #[cfg(feature = "uspace")]
    pub satp: PhysAddr,
    // TODO: FP states
}

impl TaskContext {
    /// Creates a dummy context for a new task.
    ///
    /// Note the context is not initialized, it will be filled by [`switch_to`]
    /// (for initial tasks) and [`init`] (for regular tasks) methods.
    ///
    /// [`init`]: TaskContext::init
    /// [`switch_to`]: TaskContext::switch_to
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "uspace")]
            satp: crate::paging::kernel_page_table_root(),
            ..Default::default()
        }
    }

    /// Initializes the context for a new task, with the given entry point and
    /// kernel stack.
    pub fn init(&mut self, entry: usize, kstack_top: VirtAddr, tls_area: VirtAddr) {
        self.sp = kstack_top.as_usize();
        self.ra = entry;
        self.tp = tls_area.as_usize();
    }

    /// Changes the page table root (`satp` register for riscv64).
    ///
    /// If not set, the kernel page table root is used (obtained by
    /// [`axhal::paging::kernel_page_table_root`][1]).
    ///
    /// [1]: crate::paging::kernel_page_table_root
    #[cfg(feature = "uspace")]
    pub fn set_page_table_root(&mut self, satp: PhysAddr) {
        self.satp = satp;
    }

    /// Switches to another task.
    ///
    /// It first saves the current task's context from CPU to this place, and then
    /// restores the next task's context from `next_ctx` to CPU.
    pub fn switch_to(&mut self, next_ctx: &Self) {
        #[cfg(feature = "tls")]
        {
            self.tp = super::read_thread_pointer();
            unsafe { super::write_thread_pointer(next_ctx.tp) };
        }
        #[cfg(feature = "uspace")]
        unsafe {
            if self.satp != next_ctx.satp {
                super::write_page_table_root(next_ctx.satp);
            }
        }
        unsafe {
            // TODO: switch FP states
            context_switch(self, next_ctx)
        }
    }
}

/// Context to enter user space.
#[cfg(feature = "uspace")]
pub struct UspaceContext(TrapFrame);

#[cfg(feature = "uspace")]
impl UspaceContext {
    /// Creates an empty context with all registers set to zero.
    pub const fn empty() -> Self {
        unsafe { core::mem::MaybeUninit::zeroed().assume_init() }
    }

    /// Creates a new context with the given entry point, user stack pointer,
    /// and the argument.
    pub fn new(entry: usize, ustack_top: VirtAddr) -> Self {
        const SPIE: usize = 1 << 5;
        const SUM: usize = 1 << 18;
        Self(TrapFrame {
            regs: GeneralRegisters {
                sp: ustack_top.as_usize(),
                ..Default::default()
            },
            sepc: entry,
            sstatus: SPIE | SUM,
        })
    }

    /// Creates a new context from the given [`TrapFrame`].
    pub const fn from(trap_frame: &TrapFrame) -> Self {
        Self(*trap_frame)
    }

    /// Gets the instruction pointer.
    pub const fn get_ip(&self) -> usize {
        self.0.sepc
    }

    /// Gets the stack pointer.
    pub const fn get_sp(&self) -> usize {
        self.0.regs.sp
    }

    /// Sets the instruction pointer.
    pub const fn set_ip(&mut self, pc: usize) {
        self.0.sepc = pc;
    }

    /// Sets the stack pointer.
    pub const fn set_sp(&mut self, sp: usize) {
        self.0.regs.sp = sp;
    }

    /// Sets the return value register.
    pub const fn set_retval(&mut self, a0: usize) {
        self.0.regs.a0 = a0;
    }

    /// Enters user space.
    ///
    /// It restores the user registers and jumps to the user entry point
    /// (saved in `sepc`).
    /// When an exception or syscall occurs, the kernel stack pointer is
    /// switched to `kstack_top`.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it changes processor mode and the stack.
    #[inline(never)]
    #[no_mangle]
    pub unsafe fn enter_uspace(&self, kstack_top: VirtAddr) -> ! {
        use riscv::register::{sepc, sscratch};

        super::disable_irqs();
        sscratch::write(kstack_top.as_usize());
        sepc::write(self.0.sepc);
        // Address of the top of the kernel stack after saving the trap frame.
        let kernel_trap_addr = kstack_top.as_usize() - core::mem::size_of::<TrapFrame>();
        asm!("
            mv      sp, {tf}

            sd      gp, 16({kernel_trap_addr})
            ld      gp, 16(sp)

            sd      tp, 24({kernel_trap_addr})
            ld      tp, 24(sp)

            ld      t0, 256(sp)
            csrw    sstatus, t0
            ld      ra, 0(sp)
            ld      t0, 32(sp)
            ld      t1, 40(sp)
            ld      t2, 48(sp)
            ld      s0, 56(sp)
            ld      s1, 64(sp)
            ld      a0, 72(sp)
            ld      a1, 80(sp)
            ld      a2, 88(sp)
            ld      a3, 96(sp)
            ld      a4, 104(sp)
            ld      a5, 112(sp)
            ld      a6, 120(sp)
            ld      a7, 128(sp)
            ld      s2, 136(sp)
            ld      s3, 144(sp)
            ld      s4, 152(sp)
            ld      s5, 160(sp)
            ld      s6, 168(sp)
            ld      s7, 176(sp)
            ld      s8, 184(sp)
            ld      s9, 192(sp)
            ld      s10, 200(sp)
            ld      s11, 208(sp)
            ld      t3, 216(sp)
            ld      t4, 224(sp)
            ld      t5, 232(sp)
            ld      t6, 240(sp)
            ld      sp, 8(sp)
            sret",
            tf = in(reg) &(self.0),
            kernel_trap_addr = in(reg) kernel_trap_addr,
            options(noreturn),
        )
    }
}

#[unsafe(naked)]
unsafe extern "C" fn context_switch(_current_task: &mut TaskContext, _next_task: &TaskContext) {
    core::arch::naked_asm!(
        "
        // save old context (callee-saved registers)
        sd      ra, 0(a0)
        sd      sp, 8(a0)
        sd      s0, 16(a0)
        sd      s1, 24(a0)
        sd      s2, 32(a0)
        sd      s3, 40(a0)
        sd      s4, 48(a0)
        sd      s5, 56(a0)
        sd      s6, 64(a0)
        sd      s7, 72(a0)
        sd      s8, 80(a0)
        sd      s9, 88(a0)
        sd      s10, 96(a0)
        sd      s11, 104(a0)

        // restore new context
        ld      s11, 104(a1)
        ld      s10, 96(a1)
        ld      s9, 88(a1)
        ld      s8, 80(a1)
        ld      s7, 72(a1)
        ld      s6, 64(a1)
        ld      s5, 56(a1)
        ld      s4, 48(a1)
        ld      s3, 40(a1)
        ld      s2, 32(a1)
        ld      s1, 24(a1)
        ld      s0, 16(a1)
        ld      sp, 8(a1)
        ld      ra, 0(a1)

        ret",
    )
}
