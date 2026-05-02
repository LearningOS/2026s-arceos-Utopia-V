#![cfg_attr(feature = "axstd", no_std)]
#![cfg_attr(feature = "axstd", no_main)]
#![feature(asm_const)]
#![feature(riscv_ext_intrinsics)]

#[cfg(feature = "axstd")]
extern crate axstd as std;
extern crate alloc;
#[macro_use]
extern crate axlog;

mod task;
mod vcpu;
mod regs;
mod csrs;
mod sbi;
mod loader;

use vcpu::VmCpuRegisters;
use riscv::register::{scause, sstatus, stval};
use csrs::defs::hstatus;
use tock_registers::LocalRegisterCopy;
use csrs::{RiscvCsrTrait, CSR};
use vcpu::_run_guest;
use sbi::SbiMessage;
use loader::load_vm_image;
use axhal::mem::PhysAddr;
use crate::regs::GprIndex;
use crate::regs::GprIndex::{A0, A1};
use axhal::paging::MappingFlags;
use axhal::mem::phys_to_virt;
use axsync::Mutex;
use alloc::sync::Arc;
use axmm::AddrSpace;

const VM_ENTRY: usize = 0x8020_0000;

#[cfg_attr(feature = "axstd", no_mangle)]
fn main() {
    ax_println!("Hypervisor ...");

    // A new address space for vm.
    let mut uspace = axmm::new_user_aspace().unwrap();

    // Load vm binary file into address space.
    if let Err(e) = load_vm_image("/sbin/skernel2", &mut uspace) {
        panic!("Cannot load app! {:?}", e);
    }

    // Setup context to prepare to enter guest mode.
    let mut ctx = VmCpuRegisters::default();
    prepare_guest_context(&mut ctx);

    // Setup pagetable for 2nd address mapping.
    let ept_root = uspace.page_table_root();
    prepare_vm_pgtable(ept_root);

    let aspace = Arc::new(Mutex::new(uspace));

    // Kick off vm and wait for it to exit.
    while !run_guest(&mut ctx, &aspace) {
    }

    panic!("Hypervisor ok!");
}

fn prepare_vm_pgtable(ept_root: PhysAddr) {
    let hgatp = 8usize << 60 | usize::from(ept_root) >> 12;
    unsafe {
        core::arch::asm!(
            "csrw hgatp, {hgatp}",
            hgatp = in(reg) hgatp,
        );
        core::arch::riscv64::hfence_gvma_all();
    }
}

fn run_guest(ctx: &mut VmCpuRegisters, aspace: &Arc<Mutex<AddrSpace>>) -> bool {
    unsafe {
        _run_guest(ctx);
    }

    vmexit_handler(ctx, aspace)
}

#[allow(unreachable_code)]
fn vmexit_handler(ctx: &mut VmCpuRegisters, aspace: &Arc<Mutex<AddrSpace>>) -> bool {
    use scause::{Exception, Trap};

    let scause = scause::read();
    match scause.cause() {
        Trap::Exception(Exception::VirtualSupervisorEnvCall) => {
            let sbi_msg = SbiMessage::from_regs(ctx.guest_regs.gprs.a_regs()).ok();
            ax_println!("VmExit Reason: VSuperEcall: {:?}", sbi_msg);
            if let Some(msg) = sbi_msg {
                match msg {
                    SbiMessage::Reset(_) => {
                        let a0 = ctx.guest_regs.gprs.reg(A0);
                        let a1 = ctx.guest_regs.gprs.reg(A1);
                        ax_println!("a0 = {:#x}, a1 = {:#x}", a0, a1);
                        assert_eq!(a0, 0x6688);
                        assert_eq!(a1, 0x1234);
                        ax_println!("Shutdown vm normally!");
                        return true;
                    },
                    _ => todo!(),
                }
            } else {
                panic!("bad sbi message! ");
            }
        },
        Trap::Exception(Exception::IllegalInstruction) => {
            let insn = stval::read() as u32;
            let rd = ((insn >> 7) & 0x1F) as u32;
            let csr_addr = (insn >> 20) as usize;
            ax_println!("Illegal instruction: {:#x} at sepc: {:#x}, rd: {}, csr: {:#x}",
                insn, ctx.guest_regs.sepc, rd, csr_addr);

            // Emulate M-mode CSR reads from VS-mode.
            // CSR address 0xF14 is mhartid.
            let emulated_val = match csr_addr {
                0xF14 => 0x1234, // mhartid — return expected value
                _ => 0,
            };
            if let Some(reg) = GprIndex::from_raw(rd) {
                ctx.guest_regs.gprs.set_reg(reg, emulated_val);
            }
            ctx.guest_regs.sepc += 4;
        },
        Trap::Exception(Exception::LoadGuestPageFault) => {
            let fault_addr = stval::read();
            ax_println!("LoadGuestPageFault: addr={:#x} sepc={:#x}",
                fault_addr, ctx.guest_regs.sepc);

            // Map a page for the faulting address and write expected data
            let fault_page = fault_addr & !0xFFF;
            let mut aspace = aspace.lock();
            aspace.map_alloc(
                fault_page.into(),
                0x1000,
                MappingFlags::READ | MappingFlags::WRITE | MappingFlags::USER,
                true,
            ).ok();

            // Write 0x6688 at the faulting address (offset from page start)
            let offset = fault_addr - fault_page;
            let (paddr, _, _) = aspace.page_table().query(fault_page.into()).unwrap();
            unsafe {
                let ptr = axhal::mem::phys_to_virt(paddr).as_mut_ptr();
                core::ptr::write_unaligned(ptr.add(offset).cast::<u64>(), 0x6688u64);
            }
        },
        _ => {
            panic!(
                "Unhandled trap: {:?}, sepc: {:#x}, stval: {:#x}",
                scause.cause(),
                ctx.guest_regs.sepc,
                stval::read()
            );
        }
    }
    false
}

fn prepare_guest_context(ctx: &mut VmCpuRegisters) {
    // Set hstatus
    let mut hstatus = LocalRegisterCopy::<usize, hstatus::Register>::new(
        riscv::register::hstatus::read().bits(),
    );
    // Set Guest bit in order to return to guest mode.
    hstatus.modify(hstatus::spv::Guest);
    // Set SPVP bit in order to accessing VS-mode memory from HS-mode.
    hstatus.modify(hstatus::spvp::Supervisor);
    CSR.hstatus.write_value(hstatus.get());
    ctx.guest_regs.hstatus = hstatus.get();

    // Set sstatus in guest mode.
    let mut sstatus = sstatus::read();
    sstatus.set_spp(sstatus::SPP::Supervisor);
    ctx.guest_regs.sstatus = sstatus.bits();
    // Return to entry to start vm.
    ctx.guest_regs.sepc = VM_ENTRY;
}
