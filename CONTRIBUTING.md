# Contributing to camxiom

## Commit conventions

Every commit MUST follow the rules below. They exist so the history is
readable by anyone — not only the author, and not only people who have
seen this project's internal planning notes.

### 1. Subject: `type(scope): summary`

- `type` is one of: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`.
- `scope` is the affected module: `types`, `model`, `projection`,
  `distortion`, `jacobian`, `lut`, `batch`, `compat`, `remap`,
  `optimizer`, `init`, `seed`, `calib`, … Omit the scope for
  repository-wide changes (`docs:`, `chore:`).
- `summary` is a short, lower-case, imperative phrase describing the
  code change. No trailing period. Written in English.

```
Good:  feat(init): Zhang pinhole intrinsics initialiser
Good:  fix(validate): widen DOUBLE_SPHERE xi range to (-1, 1)
Bad:   feat(remap): MS2-1 + MS2-1e + MS2-2 done
```

### 2. No internal planning vocabulary, anywhere in the message

The subject and body describe WHAT the code does and WHY, in domain
language a reader who has never seen the planning notes can follow. Do
NOT reference internal milestone or decision identifiers, phase names,
or scratch terminology (e.g. "MS2-3", "D46", "Phase B", "the staging
skeleton"). Translate the intent into plain technical language.

```
Bad body:   Implements MS2-3' per D46; staging skeleton removed.
Good body:  One Ceres pass refines the intrinsics and per-view poses;
            it carries no sampling or restart strategy — that is the
            caller's responsibility.
```

### 3. One logical change per commit

Each commit is a single coherent unit: one module / feature / fix,
bundling its implementation, its tests, and its build wiring
(`CMakeLists.txt`) together, so the commit builds and its tests pass on
its own. Never batch unrelated modules, or a feature plus an unrelated
document, into one commit. When in doubt, split.

### 4. Body and trailer

Wrapped prose, about 72 columns. Numbered steps are fine for
algorithms. End every commit with the trailer used across this repo:

```
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

### 5. When to commit

Commit only when explicitly asked to. Never auto-commit.

## Reference commits

`c9df471`, `5471a0d`, `eb3aa28`, and `a11cad4` satisfy all five rules
and are the canonical examples. Run `git log` before starting new work
and match that style.

## Rewriting a non-compliant commit

If a commit violates these rules and has not been pushed, rewrite it
rather than leaving it:

1. Set aside any uncommitted work with `git stash -u` **and** a
   physical backup outside the repo (a `git bundle --all` plus a copy
   of every modified and untracked file).
2. `git reset --soft` to the parent of the bad commit.
3. Rebuild the change as properly-scoped commits per the rules above.
4. Verify the resulting tree SHA is **byte-identical** to the original
   (`git rev-parse HEAD^{tree}`), so only history — never content —
   changed.
5. `git stash pop` and commit the set-aside work, also per the rules.

Never force-push shared history. A local, unpushed branch can be
rewritten freely.
