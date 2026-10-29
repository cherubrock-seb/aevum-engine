#!/usr/bin/env python3
"""Gpu::saveProof() must survive a corrupt proof residue.

ProofSet::computeProof() reads the saved residues with File::readChecked(), which throws CRCError when a file is corrupt.
Reading the same file again fails the same way, so saveProof() has to leave the retry loop and try a lower proof power
(whose checkpoint table no longer needs that file) instead of letting the exception escape and kill the task.
CRCError is a plain struct (not a std::exception), so nothing above saveProof() would catch it either.

Checks the source of saveProof(): the computeProof() call sits inside a try block whose handler catches CRCError by
reference and leaves the retry loop with break, ahead of the reducePower() call.
"""
import pathlib
import re
import sys

src = (pathlib.Path(__file__).resolve().parent.parent / "src" / "Gpu.cpp").read_text()
m = re.search(r"fs::path Gpu::saveProof\(const Args& args, ProofSet& proofSet\) \{\n(.*?)\n\}\n", src, re.S)
if not m:
    print("FAIL: Gpu::saveProof not found")
    sys.exit(1)
body = m.group(1)
fail = []

compute = body.find("proofSet.computeProof(this)")
tryPos = body.rfind("try {", 0, compute)
handler = re.search(r"catch \(const CRCError&\s*\w*\)\s*\{(.*?)\}", body, re.S)
reduce = body.find("proofSet.reducePower()")

if compute < 0:
    fail.append("saveProof no longer calls computeProof")
if tryPos < 0:
    fail.append("computeProof is not inside a try block")
if not handler:
    fail.append("no catch (const CRCError&) handler")
else:
    if handler.start() < compute:
        fail.append("the CRCError handler precedes computeProof")
    if "break;" not in handler.group(1):
        fail.append("the CRCError handler does not leave the retry loop")
    if reduce >= 0 and handler.start() > reduce:
        fail.append("the CRCError handler comes after reducePower")
if reduce < 0:
    fail.append("saveProof no longer reduces the proof power")

if fail:
    for f in fail:
        print("FAIL:", f)
    sys.exit(1)
print("proof_crc_retry_source_test: ok")
