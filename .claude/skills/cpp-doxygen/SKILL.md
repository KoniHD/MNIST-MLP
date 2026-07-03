---
name: cpp-doxygen
description: Write or fix C++ comments in MNIST-CUDA the project's way — concise Doxygen that keeps IDE hover genuinely useful instead of dumping a screen of detail. Use whenever the user asks to add, fix, clean up, condense, or review comments / documentation / doc-comments on the C++ files (Utility.hpp, Utility.cpp, *_submission.cpp), mentions Doxygen or `@brief`/`@param`, or complains that IDE hover / tooltips show too much implementation detail.
---

Comment the MNIST-CUDA C++ so that **IDE hover stays short and useful**. This is the single most
important constraint, and the reason for every rule below.

## Why hover is the constraint

The repo-root `.clangd` sets `Documentation: CommentFormat: Doxygen`. clangd attaches the comment
block **immediately preceding a symbol** — whether that's a declaration in the header *or a
definition in a `.cpp`* — to the symbol, and renders it on hover at **every call site**. So a long
multi-paragraph banner above `void Module::backward(...)` in `student_submission.cpp` shows up,
in full, whenever anyone hovers `model.backward(...)` anywhere. Students use clangd (neovim, VS
Code, CLion all wire it the same way), so noisy hover hurts the people the code is written for.

The fix: the **header** carries concise canonical docs; the **`.cpp` files** carry no doc blocks
on their definitions, with the real "why" relocated *inside* function bodies where clangd ignores
it.

## Rules

### Doc-comment style
- Use `///` (triple-slash) Doxygen for documentation on declarations. Not `//`, not `/* */`.
- Tags: `/// @brief <one concise line>`, then `/// @param <name> <desc>` per parameter, and
  `/// @return <desc>` when meaningful. Use `@p name` for inline parameter references.
- A hover tooltip should be a few lines, never a screen. If you're writing a third paragraph,
  it belongs in a function body, not a doc comment.

### Header declarations (`Utility.hpp`) — the canonical hover doc
- One tight `@brief` per member/function plus `@param`/`@return`.
- **Remove** implementation walk-throughs, step-by-step lists, and cross-references to
  "scratchpad" / "Design §x.y" / README sections. Distil to one accurate line.

### Definition sites (`*.cpp`)
- Symbols are already documented in the header. **Do not place any comment block (`///`, `/** */`,
  or a multi-line `//` banner) on the line immediately above a function/method definition** — that
  is exactly what pollutes hover. Leave the line above the definition blank.
- Relocate genuinely useful rationale into plain `//` comments **inside the body** (first lines,
  or beside the relevant statement). In-body comments never attach to the symbol. Trim redundant
  narration; keep substantive "why".
- Trailing inline `// ...` comments on code lines, and section dividers that are not adjacent to a
  single symbol, are fine.

### File-top banner (every file)
Every file begins with **exactly** this 4-line banner as its very first lines:
```
//
// Created by Konstantin Zeck (konstantin.zeck@gmail.com) on 30/07/2026
// License: CC-BY-SA-4.0
//
```
If a file currently opens with a large strategy/overview banner, keep the license banner at the
very top and move the overview text **below** it (lightly condensed) before `#include` — that's
before any symbol, so it does not pollute hover.

## Don't change code
Edit comments only — no logic, signatures, includes, or member order. (You may move a rationale
sentence from above a definition into the body as a `//` comment, but add no code.)

## Finish up
After editing, run **clang-format** on the files (see the `format` skill). Verify `/* */` blocks
stay balanced (a broken delimiter silently swallows code), and that the line above every `.cpp`
definition is blank.
