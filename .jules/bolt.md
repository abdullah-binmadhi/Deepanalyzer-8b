# Bolt Performance Journal

## 2025-05-18 - Single-Pass Luhn Checksum Algorithm
**Learning:** Replacing regex `re.sub(r"\D", "", ...)` and list reversal `[::-1]` with single-pass reverse iteration (`reversed(s)`) and ASCII integer math (`ord(c) - 48`) speeds up Luhn algorithm validation by ~1.9x while eliminating all temporary list and string allocations.
**Action:** Prefer single-pass character iteration with ASCII arithmetic over regex substitution and list reversals when validating high-frequency string identifiers.
