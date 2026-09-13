# Claude Code Instructions

## IMPORTANT: Investigate Before Acting

**ALWAYS explore existing code, tests, and tools before writing new scripts or making changes.**

- Read relevant files to understand the existing architecture
- Check for existing test files, diagnostic tools, or utilities
- Understand the code flow before proposing fixes
- Do NOT create new debug scripts when existing infrastructure might work
- Do NOT make changes to fix problems you haven't verified exist in actual usage

## Running Python

Use the venv for all Python commands:

```bash
source venv/bin/activate && python <script.py>
```

Or for one-off commands:
```bash
venv/bin/python <script.py>
```
