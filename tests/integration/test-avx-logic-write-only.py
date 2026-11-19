#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Check AVX logical destination/source aliasing with deferred YMM upper zeros.

Emit small static x86-64 ELF guests directly so this test needs no cross
assembler. Each guest stores all three registers for a byte-exact comparison;
the oracle includes source preservation, not just the destination result.
Optional --check-codegen requires a LATX debug build and checks the final host
dump for the unnecessary destination clear in a disjoint-source case.
Default runs use JIT only, even if the inherited environment enables AOT.
For final acceptance only, --aot with LATX_AOT=1 explicitly requests the
existing XOR cold/hot checks and requires a cache-load marker.
"""
import argparse
import hashlib
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import time


INITIAL = [bytes((n * 71 + i * 37 + 1) & 255 for i in range(32)) for n in range(3)]
# VEX.0F opcode, mandatory prefix, logical operation.
OPS = {
    'vxorps': (0x57, 0, 'xor'), 'vxorpd': (0x57, 1, 'xor'),
    'vandps': (0x54, 0, 'and'), 'vandpd': (0x54, 1, 'and'),
    'vandnps': (0x55, 0, 'andn'), 'vandnpd': (0x55, 1, 'andn'),
    'vorps': (0x56, 0, 'or'), 'vorpd': (0x56, 1, 'or'),
    'vpxor': (0xef, 1, 'xor'), 'vpand': (0xdb, 1, 'and'),
    'vpandn': (0xdf, 1, 'andn'), 'vpor': (0xeb, 1, 'or'),
}


def logical_result(kind, lhs, rhs):
    operations = {'xor': lambda a, b: a ^ b, 'and': lambda a, b: a & b,
                  'andn': lambda a, b: (~a) & b, 'or': lambda a, b: a | b}
    return bytes(operations[kind](a, b) for a, b in zip(lhs, rhs))


def make_elf(code, contents):
    base, data = 0x400000, 0x401000
    ident = b'\x7fELF' + bytes([2, 1, 1, 0]) + bytes(8)
    header = struct.pack('<16sHHIQQQIHHHHHH', ident, 2, 62, 1,
                         base + 256, 64, 0, 0, 64, 56, 2, 0, 0, 0)
    headers = struct.pack('<IIQQQQQQ', 1, 5, 0, base, base, 4096, 4096, 4096)
    headers += struct.pack('<IIQQQQQQ', 1, 6, 4096, data, data,
                           len(contents), len(contents), 4096)
    elf = (header + headers).ljust(256, b'\0') + code
    assert len(elf) <= 4096
    return elf.ljust(4096, b'\0') + contents


def make_guest(src1, src2, pending, mnemonic='vxorps', narrow=False, mem_offset=64):
    """None for src2 selects a memory operand, optionally unaligned."""
    data = 0x401000
    opcode, prefix, kind = OPS[mnemonic]
    code = bytearray(b'\x48\xbb' + struct.pack('<Q', data))  # mov rbx, data
    for reg in range(3):
        # vmovups ymmN, [rbx + N*32]
        code += bytes([0xc5, 0xfc, 0x10, 0x43 | (reg << 3), reg * 32])
    values = list(INITIAL)
    for reg in range(3):
        if pending & (1 << reg):
            # vmovaps xmmN, xmmN: low bits unchanged, upper bits logically zero.
            code += bytes([0xc5, 0xf8, 0x28, 0xc0 | (reg << 3) | reg])
            values[reg] = values[reg][:16] + bytes(16)
    # VEX.vvvv selects src1, L selects width, pp selects the mandatory prefix.
    vex = 0x80 | ((~src1 & 15) << 3) | (0 if narrow else 4) | prefix
    if src2 is None:
        code += bytes([0xc5, vex, opcode, 0x43, mem_offset])
        rhs = b''.join(INITIAL)[mem_offset:mem_offset + 32]
    else:
        code += bytes([0xc5, vex, opcode, 0xc0 | src2])
        rhs = values[src2]
    result = logical_result(kind, values[src1], rhs)
    values[0] = result[:16] + bytes(16) if narrow else result
    for reg in range(3):
        # vmovups [rbx + 128 + N*32], ymmN (disp32)
        code += bytes([0xc5, 0xfc, 0x11, 0x83 | (reg << 3)])
        code += struct.pack('<I', 128 + reg * 32)
    code += bytes.fromhex('b8 01 00 00 00 bf 01 00 00 00')  # write(1, ...)
    code += bytes.fromhex('48 8d b3 80 00 00 00 ba 60 00 00 00 0f 05')
    code += bytes.fromhex('b8 3c 00 00 00 31 ff 0f 05')  # exit(0)
    elf = make_elf(code, b''.join(INITIAL) + bytes(128))
    return elf, b''.join(values)


def make_fault_guest(src1, pending, mnemonic='vxorps', address=0, *,
                     instruction=None):
    """Fault on a null/cross-page source; inspect the signal frame and sigreturn.

    Linux x86-64 ucontext has mcontext at +40, RIP at +168 and fpregs at +224.
    The standard XSAVE layout has XMM at +160 and YMMH at +576. Check its
    advertised AVX feature before interpreting these offsets.
    """
    data, handler, restorer = 0x401000, 0x400500, 0x400600
    opcode, prefix, _ = OPS[mnemonic]
    code = bytearray.fromhex('b8 0d 00 00 00 bf 0b 00 00 00')  # rt_sigaction
    code += b'\x48\xbe' + struct.pack('<Q', data + 224)
    code += bytes.fromhex('31 d2 41 ba 08 00 00 00 0f 05 85 c0 79 0c')
    code += bytes.fromhex('b8 3c 00 00 00 bf 4d 00 00 00 0f 05')  # setup failed
    code += b'\x48\xbb' + struct.pack('<Q', data)
    values = list(INITIAL)
    for reg in range(3):
        code += bytes([0xc5, 0xfc, 0x10, 0x43 | (reg << 3), reg * 32])
    for reg in range(3):
        if pending & (1 << reg):
            code += bytes([0xc5, 0xf8, 0x28, 0xc0 | (reg << 3) | reg])
            values[reg] = values[reg][:16] + bytes(16)
    # Nonzero address leaves 16 readable bytes before an unmapped host page.
    code += b'\xb9' + struct.pack('<I', address)  # mov ecx, address
    fault_pc = 0x400100 + len(code)
    vex = 0x84 | ((~src1 & 15) << 3) | prefix
    instruction = instruction or bytes([0xc5, vex, opcode, 0x01])
    code += instruction
    for reg in range(3):
        code += bytes([0xc5, 0xfc, 0x11, 0x83 | (reg << 3)])
        code += struct.pack('<I', 128 + reg * 32)
    code += bytes.fromhex('b8 01 00 00 00 bf 01 00 00 00')
    code += bytes.fromhex('48 8d b3 80 00 00 00 ba 60 00 00 00 0f 05')
    code += bytes.fromhex('b8 3c 00 00 00 31 ff 0f 05')
    # Handler writes 832 bytes of XSAVE state, then the fault RIP. It advances
    # saved RIP past the failed operation; sigreturn restores pre-fault YMMs.
    code = code.ljust(handler - 0x400100, b'\0')
    code += bytes.fromhex('49 89 d4 48 8b 9a e0 00 00 00 48 89 de')
    code += bytes.fromhex('b8 01 00 00 00 bf 01 00 00 00 ba 40 03 00 00 0f 05')
    code += bytes.fromhex('49 8d b4 24 a8 00 00 00')
    code += bytes.fromhex('b8 01 00 00 00 bf 01 00 00 00 ba 08 00 00 00 0f 05')
    code += bytes.fromhex('49 83 84 24 a8 00 00 00') + bytes([len(instruction), 0xc3])
    code = code.ljust(restorer - 0x400100, b'\0')
    code += bytes.fromhex('b8 0f 00 00 00 0f 05 0f 0b')
    action = struct.pack('<QQQQ', handler, 0x04000004, restorer, 0)
    contents = b''.join(INITIAL) + bytes(128) + action
    if address:
        # Map the whole readable prefix, including on hosts with 16/64 KiB pages.
        contents = contents.ljust(address + 16 - data, b'\0')
    elf = make_elf(code, contents)
    return elf, b''.join(values), fault_pc


def check_fault_output(output, expected, fault_pc):
    assert len(output) == 936, f'missing fault frame or signal return: {len(output)} bytes'
    assert struct.unpack_from('<I', output, 464)[0] == 0x46505853, 'no XSAVE frame'
    assert struct.unpack_from('<Q', output, 472)[0] & 4, 'frame lacks AVX state'
    assert struct.unpack_from('<Q', output, 832)[0] == fault_pc, 'wrong fault RIP'
    saved = b''.join(output[160 + i * 16:176 + i * 16] +
                     output[576 + i * 16:592 + i * 16] for i in range(3))
    assert saved == expected, 'incorrect pre-fault registers in signal frame'
    assert output[840:] == expected, 'incorrect registers after sigreturn'


def check_codegen(log, mnemonic='vxorps'):
    log = re.sub(r'\x1b\[[0-9;]*m', '', log)
    host = [line for line in log.splitlines()
            if re.match(r'IR2\[\d+\] at 0x[0-9a-f]+ 0x[0-9a-f]+ ', line)]
    operation = 'xv' + OPS[mnemonic][2] + '.v'
    matches = [i for i, line in enumerate(host) if f' {operation} ' in line]
    if len(matches) != 1:
        raise AssertionError(f'expected one final {operation}; use a LATX debug build')
    index = matches[0]
    dest = host[index].split(operation, 1)[1].strip().split(',')[0]
    if any(re.search(r' xvpermi\.q\s+' + re.escape(dest) + r',[^,]+,2$', line)
           for line in host[:index]):
        raise AssertionError(f'pure-write destination still cleared before {mnemonic}')


def check_aot(emulator, work, name, elf, expected, fault_pc=None):
    # Each immutable guest has a unique path, hence its own AOT cache key.
    guest = work / name
    guest.write_bytes(elf)
    guest.chmod(0o755)
    marker = work / (name + '.loaded')
    cache = Path.home() / '.cache' / 'latx'
    key = hashlib.sha256(str(guest).encode()).hexdigest()
    pattern = 'v2-*-' + key + '.aot2'
    env = os.environ.copy()
    env.update(LATX_AOT='1', LATX_DUMP='00000')
    env.pop('LATX_TEST_AOT_LOAD_MARKER', None)
    try:
        for mode in ('cold', 'hot'):
            env['LATX_AOT_GENERATE'] = '1' if mode == 'cold' else '0'
            if mode == 'hot':
                env['LATX_TEST_AOT_LOAD_MARKER'] = str(marker)
            result = subprocess.run([emulator, str(guest)], env=env,
                                    capture_output=True, timeout=30)
            assert result.returncode == 0, (
                name, mode, result.returncode,
                result.stderr.decode(errors='replace')[-2000:])
            if fault_pc is None:
                assert result.stdout == expected, (name, mode, 'wrong registers')
            else:
                check_fault_output(result.stdout, expected, fault_pc)
            if mode == 'cold':
                deadline = time.monotonic() + 10
                while not any(p.stat().st_size for p in cache.glob(pattern)):
                    if time.monotonic() >= deadline:
                        raise AssertionError(f'{name}: no non-empty AOT artifact')
                    time.sleep(0.1)
            else:
                assert marker.exists() and str(guest) in marker.read_text().splitlines(), (
                    name, 'AOT artifact was not loaded')
    finally:
        # Only this temporary guest's cache, including its lock/temp files.
        for artifact in cache.glob(pattern + '*'):
            artifact.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('emulator')
    parser.add_argument('--check-codegen', action='store_true')
    parser.add_argument('--aot', action='store_true',
                        help='final acceptance only: XOR cold/hot AOT; requires '
                             'LATX_AOT=1 and an AOT-enabled build')
    args = parser.parse_args()
    if args.aot and os.environ.get('LATX_AOT') != '1':
        parser.error('--aot is for final acceptance; explicitly set LATX_AOT=1')
    emulator = str(Path(args.emulator).resolve())
    env = os.environ.copy()
    env.update(LATX_AOT='0', LATX_DUMP='00000')
    # Distinct sources, either destination alias, equal sources, all aliases,
    # and memory with or without a destination/src1 alias.
    forms = [(1, 2), (0, 2), (1, 0), (1, 1), (0, 0), (1, None), (0, None)]
    cases = [(s1, s2, 64) for s1, s2 in forms] + [(1, None, 63), (0, None, 63)]
    count = fault_count = 0
    page_size = max(4096, os.sysconf('SC_PAGE_SIZE'))
    cross_page = ((0x401000 // page_size) + 1) * page_size - 16
    with tempfile.TemporaryDirectory(prefix='latx-logic-write-only-') as work:
        guest = Path(work) / 'logic-probe'
        for mnemonic in OPS:
            for narrow in (False, True):
                for src1, src2, offset in cases:
                    for pending in range(8):
                        elf, expected = make_guest(src1, src2, pending, mnemonic,
                                                   narrow, offset)
                        guest.write_bytes(elf)
                        guest.chmod(0o755)
                        result = subprocess.run([emulator, str(guest)], env=env,
                                                capture_output=True, timeout=30)
                        if result.returncode or result.stdout != expected:
                            raise AssertionError(
                                f'{mnemonic} narrow={narrow} src={src1, src2} '
                                f'offset={offset} pending={pending}: exit={result.returncode}, '
                                f'expected={expected.hex()}, got={result.stdout.hex()}, '
                                f'stderr={result.stderr.decode(errors="replace")[-2000:]}')
                        count += 1
        print(f'PASS: {count} logical correctness cases (LATX_AOT=0)', flush=True)
        for mnemonic in OPS:
            for src1 in (0, 1):
                for pending in range(8):
                    for address in (0, cross_page):
                        elf, expected, pc = make_fault_guest(src1, pending, mnemonic, address)
                        guest.write_bytes(elf)
                        result = subprocess.run([emulator, str(guest)], env=env,
                                                capture_output=True, timeout=30)
                        context = (mnemonic, src1, pending, hex(address))
                        assert result.returncode == 0, (context, result.returncode)
                        try:
                            check_fault_output(result.stdout, expected, pc)
                        except AssertionError as error:
                            raise AssertionError(f'{context}: {error}') from error
                        fault_count += 1
        print(f'PASS: {fault_count} fault-frame and signal-return cases (LATX_AOT=0)',
              flush=True)
        if args.aot:
            work = Path(work)
            for pd in (False, True):
                mnemonic = 'vxorpd' if pd else 'vxorps'
                # All alias forms with pending zeros, plus a disjoint-source
                # destination-only case and XMM controls for both source forms.
                cases = [(s1, s2, 7, False) for s1, s2 in forms]
                cases += [(1, 2, 1, False), (1, 2, 7, True), (0, None, 7, True)]
                for index, (src1, src2, pending, narrow) in enumerate(cases):
                    elf, expected = make_guest(src1, src2, pending, mnemonic, narrow)
                    check_aot(emulator, work, f'aot-{int(pd)}-{index}', elf, expected)
                for src1 in (0, 1):
                    for pending in (1, 7):
                        elf, expected, pc = make_fault_guest(src1, pending, mnemonic)
                        check_aot(emulator, work, f'fault-{int(pd)}-{src1}-{pending}',
                                  elf, expected, pc)
            print('PASS: 28 cold/hot AOT pairs (20 normal, 8 fault); '
                  'all hot runs confirmed cache load', flush=True)
        if args.check_codegen:
            env['LATX_DUMP'] = '01110'
            for mnemonic in OPS:
                elf, expected = make_guest(1, 2, 1, mnemonic)
                guest.write_bytes(elf)
                result = subprocess.run([emulator, str(guest)], env=env,
                                        capture_output=True, timeout=30)
                if result.returncode or result.stdout != expected:
                    raise AssertionError(f'{mnemonic}: code-generation probe gave wrong result')
                check_codegen(result.stderr.decode(errors='replace'), mnemonic)
            print(f'PASS: {len(OPS)} pure-write final-host checks')


if __name__ == '__main__':
    main()
