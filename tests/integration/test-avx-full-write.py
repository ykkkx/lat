#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""JIT regression for write-only AVX destinations, with eager-zero controls.

Each pair starts with identical architectural state. The control materializes
pending upper zeros through stores before the operation; the other leaves them
pending. Compare every live vector register, MXCSR and RFLAGS. --baseline also
compares against an unchanged LAT binary; it is a regression oracle, not an
independent claim that every baseline instruction implements the ISA correctly.
No assembler is required. VEX encodings are listed explicitly below.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location(
    'logic_probe', Path(__file__).with_name('test-avx-logic-write-only.py'))
logic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(logic)

CASES = []


def add(name, opcode, *, map=1, pp=1, w=0, kind='binary', imm=None,
        widths=(0, 1), data='int'):
    CASES.append(dict(name=name, opcode=opcode, map=map, pp=pp, w=w, kind=kind,
                      imm=imm, widths=widths, data=data))


for suffix, pp, data in [('ps', 0, 'f32'), ('pd', 1, 'f64')]:
    for name, opcode in [('add', 0x58), ('sub', 0x5c), ('mul', 0x59),
                         ('div', 0x5e), ('min', 0x5d), ('max', 0x5f)]:
        add('v' + name + suffix, opcode, pp=pp, data=data)
    add('vsqrt' + suffix, 0x51, pp=pp, data=data, kind='unary')
    for name, opcode in [('addsub', 0xd0), ('hadd', 0x7c), ('hsub', 0x7d)]:
        add('v' + name + suffix, opcode, pp=3 if suffix == 'ps' else 1, data=data)
    for pred in range(32):
        add(f'vcmp{suffix}-{pred:02d}', 0xc2, pp=pp, data=data, imm=pred)
    add('vshuf' + suffix, 0xc6, pp=pp, imm=0x96, data=data)
    add('vblend' + suffix, 0x0c + pp, map=3, imm=0xa5, data=data)
    add('vblendv' + suffix, 0x4a + pp, map=3, imm=0x30, data=data)
    for high in (False, True):
        add('vunpck' + ('h' if high else 'l') + suffix, 0x15 if high else 0x14,
            pp=pp, data=data)
    for imm in (0, 4, 11):
        add(f'vround{suffix}-{imm}', 0x08 + pp, map=3, kind='unary', imm=imm, data=data)
    add('vpermil' + suffix + '-reg', 0x0c + pp, map=2, data=data)
    add('vpermil' + suffix + '-imm', 0x04 + pp, map=3, kind='unary', imm=0x96, data=data)
for name, opcode in [('vrsqrtps', 0x52), ('vrcpps', 0x53)]:
    add(name, opcode, pp=0, kind='unary', data='f32')
for name, opcode, pp in [('vmovsldup', 0x12, 2), ('vmovshdup', 0x16, 2),
                         ('vmovddup', 0x12, 3)]:
    add(name, opcode, pp=pp, kind='unary')
for name, opcode, pp, data in [('vcvtps2pd', 0x5a, 0, 'f32'),
                              ('vcvtdq2pd', 0xe6, 2, 'int'),
                              ('vcvtdq2ps', 0x5b, 0, 'int'),
                              ('vcvtps2dq', 0x5b, 1, 'f32'),
                              ('vcvttps2dq', 0x5b, 2, 'f32')]:
    add(name, opcode, pp=pp, kind='unary', data=data)
for name, opcode in [('vpaddb', 0xfc), ('vpaddw', 0xfd), ('vpaddd', 0xfe), ('vpaddq', 0xd4),
                     ('vpaddsb', 0xec), ('vpaddsw', 0xed), ('vpaddusb', 0xdc), ('vpaddusw', 0xdd),
                     ('vpsubb', 0xf8), ('vpsubw', 0xf9), ('vpsubd', 0xfa), ('vpsubq', 0xfb),
                     ('vpsubsb', 0xe8), ('vpsubsw', 0xe9), ('vpsubusb', 0xd8), ('vpsubusw', 0xd9),
                     ('vpcmpeqb', 0x74), ('vpcmpeqw', 0x75), ('vpcmpeqd', 0x76),
                     ('vpcmpgtb', 0x64), ('vpcmpgtw', 0x65), ('vpcmpgtd', 0x66),
                     ('vpacksswb', 0x63), ('vpackssdw', 0x6b), ('vpackuswb', 0x67),
                     ('vpunpcklbw', 0x60), ('vpunpcklwd', 0x61), ('vpunpckldq', 0x62),
                     ('vpunpcklqdq', 0x6c), ('vpunpckhbw', 0x68), ('vpunpckhwd', 0x69),
                     ('vpunpckhdq', 0x6a), ('vpunpckhqdq', 0x6d),
                     ('vpmullw', 0xd5), ('vpmulhw', 0xe5), ('vpmulhuw', 0xe4),
                     ('vpmuludq', 0xf4), ('vpmaddwd', 0xf5),
                     ('vpmaxub', 0xde), ('vpmaxsw', 0xee), ('vpminub', 0xda), ('vpminsw', 0xea),
                     ('vpavgb', 0xe0), ('vpavgw', 0xe3), ('vpsadbw', 0xf6)]:
    add(name, opcode)
for name, opcode in [('vpcmpeqq', 0x29), ('vpcmpgtq', 0x37), ('vpackusdw', 0x2b),
                     ('vpmulld', 0x40), ('vpmuldq', 0x28), ('vpmaddubsw', 0x04),
                     ('vpmulhrsw', 0x0b), ('vpshufb', 0x00),
                     ('vpsignb', 0x08), ('vpsignw', 0x09), ('vpsignd', 0x0a),
                     ('vphaddw', 0x01), ('vphaddd', 0x02), ('vphaddsw', 0x03),
                     ('vphsubw', 0x05), ('vphsubd', 0x06), ('vphsubsw', 0x07),
                     ('vpmaxsb', 0x3c), ('vpmaxsd', 0x3d), ('vpmaxuw', 0x3e), ('vpmaxud', 0x3f),
                     ('vpminsb', 0x38), ('vpminsd', 0x39), ('vpminuw', 0x3a), ('vpminud', 0x3b)]:
    add(name, opcode, map=2)
for name, opcode in [('vpabsb', 0x1c), ('vpabsw', 0x1d), ('vpabsd', 0x1e)]:
    add(name, opcode, map=2, kind='unary')
for signed, base in [('s', 0x20), ('z', 0x30)]:
    for i, suffix in enumerate(('bw', 'bd', 'bq', 'wd', 'wq', 'dq')):
        add('vpmov' + signed + 'x' + suffix, base + i, map=2, kind='unary')
for name, pp in [('vpshufd', 1), ('vpshufhw', 2), ('vpshuflw', 3)]:
    add(name, 0x70, pp=pp, kind='unary', imm=0x96)
for suffix, i in [('w', 0), ('d', 1), ('q', 2)]:
    for op, base, group in [('srl', 0xd1, 2), ('sll', 0xf1, 6), ('sra', 0xe1, 4)]:
        if op == 'sra' and suffix == 'q':
            continue
        add('vp' + op + suffix + '-reg', base + i)
        for count in (0, 1, 15, 16, 31, 32, 63, 64, 255):
            add(f'vp{op}{suffix}-imm{count}', 0x71 + i,
                kind='shift' + str(group), imm=count)
for name, opcode, w in [('vpsllvd', 0x47, 0), ('vpsllvq', 0x47, 1),
                         ('vpsrlvd', 0x45, 0), ('vpsrlvq', 0x45, 1), ('vpsravd', 0x46, 0)]:
    add(name, opcode, map=2, w=w)
for name, opcode in [('vpermd', 0x36), ('vpermps', 0x16)]:
    add(name, opcode, map=2, widths=(1,))
for name, opcode in [('vpermq', 0x00), ('vpermpd', 0x01)]:
    add(name, opcode, map=3, w=1, kind='unary', imm=0x96, widths=(1,))
for name, opcode in [('vperm2f128', 0x06), ('vperm2i128', 0x46),
                     ('vinsertf128', 0x18), ('vinserti128', 0x38)]:
    for imm in ((0, 1) if 'insert' in name else (0x20, 0x31, 0x88)):
        add(name + '-' + str(imm), opcode, map=3, imm=imm, widths=(1,))
for name, opcode, widths in [('vbroadcastss', 0x18, (0, 1)), ('vbroadcastsd', 0x19, (1,)),
                             ('vbroadcastf128', 0x1a, (1,)), ('vbroadcasti128', 0x5a, (1,))]:
    add(name, opcode, map=2, kind='memonly', widths=widths)
for name, opcode, imm in [('vpblendw', 0x0e, 0xa5), ('vpblendd', 0x02, 0xa5),
                         ('vpblendvb', 0x4c, 0x30), ('vdpps', 0x40, 0xf1),
                         ('vmpsadbw', 0x42, 5), ('vpclmulqdq', 0x44, 0x11)]:
    add(name, opcode, map=3, imm=imm, data='f32' if name == 'vdpps' else 'int')
for imm in (0, 1, 15, 16, 17, 31, 32, 255):
    add('vpalignr-' + str(imm), 0x0f, map=3, imm=imm)
# Masked loads overwrite all result lanes (masked-off lanes become zero).
for name, opcode, w in [('vmaskmovps', 0x2c, 0), ('vmaskmovpd', 0x2d, 0),
                         ('vpmaskmovd', 0x8c, 0), ('vpmaskmovq', 0x8c, 1)]:
    add(name, opcode, map=2, w=w, kind='maskload')
# Read-modify-write / read-only controls must continue materializing operand 0.
for suffix, w, data in [('ps', 0, 'f32'), ('pd', 1, 'f64')]:
    for stem, opcode in [('vfmadd', 0x98), ('vfmsub', 0x9a), ('vfnmadd', 0x9c),
                          ('vfnmsub', 0x9e), ('vfmaddsub', 0x96), ('vfmsubadd', 0x97)]:
        for order, delta in [('132', 0), ('213', 0x10), ('231', 0x20)]:
            add(stem + order + suffix, opcode + delta, map=2, w=w, data=data)
for name, opcode in [('vptest', 0x17), ('vtestps', 0x0e), ('vtestpd', 0x0f)]:
    add(name, opcode, map=2, kind='test')


def encode(case, width, src1, src2, *, fault=False):
    kind = case['kind']
    vvvv = src1 if kind in ('binary', 'maskload') else 0
    reg = 0
    if kind.startswith('shift'):
        reg = int(kind[-1])
        vvvv = 0  # immediate shifts encode destination in VEX.vvvv
    vex = bytes([0xc4, 0xe0 | case['map'],
                 (case['w'] << 7) | ((~vvvv & 15) << 3) | (width << 2) | case['pp']])
    if src2 is None:
        # Normal memory is [rbx+64]; faults use [rcx].
        operand = bytes([reg << 3 | 1]) if fault else bytes([0x43 | reg << 3, 64])
    else:
        operand = bytes([0xc0 | reg << 3 | src2])
    return vex + bytes([case['opcode']]) + operand + (
        bytes([case['imm']]) if case['imm'] is not None else b'')


def initial_data(kind, special):
    if kind == 'int':
        return bytes(((i * 37 + 0x81) if special else (i * 13 + 1)) & 255 for i in range(128))
    if kind == 'f32':
        words = ([0x80000000, 0, 0x7f800000, 0xff800000, 0x7fc12345, 0x7f812345,
                  1, 0x7f7fffff] if special else
                 [0x3f800000, 0xc0000000, 0x40800000, 0x41100000,
                  0x3fc00000, 0xc0200000, 0x3f000000, 0x41800000])
        return struct.pack('<32I', *(words * 4))
    words = ([0x8000000000000000, 0x7ff0000000000000, 0x7ff8000012345678,
              0x7ff0000012345678] if special else
             [0x3ff0000000000000, 0xc000000000000000, 0x4008000000000000,
              0x3fe0000000000000])
    return struct.pack('<16Q', *(words * 4))


def elf(code, contents):
    # Large batches still have disjoint code and data segments.
    code_size = (256 + len(code) + 4095) & ~4095
    assert code_size < 0x400000
    ident = b'\x7fELF' + bytes([2, 1, 1, 0]) + bytes(8)
    header = struct.pack('<16sHHIQQQIHHHHHH', ident, 2, 62, 1,
                         0x400100, 64, 0, 0, 64, 56, 2, 0, 0, 0)
    headers = struct.pack('<IIQQQQQQ', 1, 5, 0, 0x400000, 0x400000,
                          code_size, code_size, 4096)
    headers += struct.pack('<IIQQQQQQ', 1, 6, code_size, 0x800000, 0x800000,
                           len(contents), len(contents), 4096)
    return ((header + headers).ljust(256, b'\0') + code).ljust(code_size, b'\0') + contents


def forms(case):
    if case['kind'] in ('memonly', 'maskload'):
        return [(0, None), (1, None)]
    if case['kind'].startswith('shift'):
        return [(0, 0), (0, 1)]
    if case['kind'] in ('unary', 'test'):
        return [(0, 0), (0, 1), (0, None)]
    return [(1, 2), (0, 2), (1, 0), (1, 1), (0, 0), (1, None), (0, None)]


def make_batch(case):
    contents = bytearray(4096)
    contents[:128] = initial_data(case['data'], False)
    contents[1024:1152] = initial_data(case['data'], True)
    if case['name'].startswith(('vpsll', 'vpsrl', 'vpsra')) and case['name'].endswith('-reg'):
        # Exercise a non-saturating register count as well as the large count
        # in the special dataset; the count operand uses its low 64 bits only.
        struct.pack_into('<Q', contents, 64, 1)
    struct.pack_into('<II', contents, 128, 0x1f80, 0x7f80)
    code = bytearray()
    labels = []
    for width in case['widths']:
        for special in (False, True):
            for src1, src2 in forms(case):
                for pending in range(8):
                    labels.append((width, special, src1, src2, pending))
                    for eager in (False, True):
                        base = 0x800000 + (1024 if special else 0)
                        code += b'\x48\xbb' + struct.pack('<Q', base)
                        # Reset MXCSR for every run, then initialize all live registers.
                        code += b'\x48\xb8' + struct.pack('<Q', 0x800080 + 4 * special)
                        code += bytes.fromhex('0f ae 10')  # ldmxcsr [rax]
                        for reg in range(4):
                            code += bytes([0xc5, 0xfc, 0x10, 0x43 | reg << 3, reg * 32])
                        for reg in range(3):
                            if pending & (1 << reg):
                                code += bytes([0xc5, 0xf8, 0x28, 0xc0 | reg << 3 | reg])
                        if eager:
                            for reg in range(3):
                                code += bytes([0xc5, 0xfc, 0x11, 0x83 | reg << 3])
                                code += struct.pack('<I', 256 + reg * 32)
                        code += bytes.fromhex('31 c0')  # deterministic input flags
                        code += encode(case, width, src1, src2)
                        for reg in range(4):
                            code += bytes([0xc5, 0xfc, 0x11, 0x83 | reg << 3])
                            code += struct.pack('<I', 512 + reg * 32)
                        code += bytes.fromhex('0f ae 9b 80 02 00 00')  # stmxcsr +640
                        code += bytes.fromhex('9c 58 48 89 83 84 02 00 00')  # flags +644
                        code += bytes.fromhex('b8 01 00 00 00 bf 01 00 00 00')
                        code += bytes.fromhex('48 8d b3 00 02 00 00 ba 8c 00 00 00 0f 05')
    code += bytes.fromhex('b8 3c 00 00 00 31 ff 0f 05')
    return elf(code, contents), labels


def run(emulator, guest, env):
    result = subprocess.run([emulator, str(guest)], env=env, capture_output=True, timeout=60)
    assert result.returncode == 0, (guest.name, result.returncode,
                                   result.stderr.decode(errors='replace')[-1600:])
    return result.stdout


def check_pairs(output, labels, name):
    assert len(output) == len(labels) * 280, (name, 'missing output', len(output), len(labels))
    for i, label in enumerate(labels):
        pending, eager = output[i * 280:i * 280 + 140], output[i * 280 + 140:(i + 1) * 280]
        assert pending == eager, (name, label, 'pending/eager mismatch', pending.hex(), eager.hex())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('emulator')
    parser.add_argument('--baseline', help='unchanged binary for additional differential checks')
    parser.add_argument('--filter', default='', help='only case names containing this string')
    parser.add_argument('--evidence', type=Path, help='write per-case results to JSON')
    args = parser.parse_args()
    env = os.environ.copy()
    env.update(LATX_AOT='0', LATX_DUMP='00000')
    emulator = str(Path(args.emulator).resolve())
    baseline = str(Path(args.baseline).resolve()) if args.baseline else None
    records, total, faults = [], 0, 0
    cases = [c for c in CASES if args.filter in c['name']]
    assert cases, 'filter matched no cases'
    with tempfile.TemporaryDirectory(prefix='latx-full-write-') as work:
        work = Path(work)
        for i, case in enumerate(cases):
            guest = work / case['name']
            binary, labels = make_batch(case)
            guest.write_bytes(binary)
            guest.chmod(0o755)
            output = run(emulator, guest, env)
            check_pairs(output, labels, case['name'])
            if baseline:
                reference = run(baseline, guest, env)
                check_pairs(reference, labels, 'baseline:' + case['name'])
                assert output == reference, (case['name'], 'candidate/baseline mismatch')
            total += len(labels)
            records.append(dict(name=case['name'], pairs=len(labels), result='PASS'))
            if args.evidence:
                args.evidence.write_text(json.dumps(records, indent=2) + '\n')
            if (i + 1) % 20 == 0:
                print(f'PASS: {i + 1}/{len(cases)} encoding cases, {total} state pairs', flush=True)
        # Representative null and cross-host-page memory faults, covering both
        # destination/source aliasing and every pending mask. Generic fault
        # scaffold verifies fault RIP, XSAVE state and state after sigreturn.
        fault_names = {'vaddps', 'vdivpd', 'vsqrtps', 'vcvttps2dq', 'vcmpps-01',
                       'vpaddd', 'vpacksswb', 'vpshufb', 'vpsllvd', 'vpermps',
                       'vpalignr-15', 'vroundps-11', 'vfmadd132ps'}
        page = max(4096, os.sysconf('SC_PAGE_SIZE'))
        boundary = ((0x401000 // page) + 1) * page
        for case in cases:
            if case['name'] not in fault_names:
                continue
            for src1 in (0, 1):
                for mask in range(8):
                    for address in (0, boundary - 16):
                        instruction = encode(case, 1, src1, None, fault=True)
                        binary, expected, pc = logic.make_fault_guest(
                            src1, mask, address=address, instruction=instruction)
                        guest = work / 'fault-probe'
                        guest.write_bytes(binary)
                        guest.chmod(0o755)
                        logic.check_fault_output(run(emulator, guest, env), expected, pc)
                        if baseline:
                            logic.check_fault_output(run(baseline, guest, env), expected, pc)
                        faults += 1
    print(f'PASS: {len(cases)} encoding cases, {total} pending/eager state pairs, '
          f'{faults} fault cases; LATX_AOT=0', flush=True)
    if baseline:
        print('PASS: all results and architectural state match unchanged baseline', flush=True)


if __name__ == '__main__':
    main()
