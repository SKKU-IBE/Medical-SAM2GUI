# Repository History Rewrite

The Git history was rewritten on 2026-07-14 before the v1.1.0 release. The
rewrite removed checkpoint binaries and medical demonstration media for which
public redistribution rights were unavailable or insufficiently documented.
The current repository contains only attributed public derived media and does
not contain source medical volumes.

## Existing clones

Do not pull, merge, or rebase an existing clone across this rewrite. Old object
IDs can reintroduce removed content when pushed. Preserve any uncommitted work
as a patch without binary data, then make a fresh clone and apply the reviewed
patch there:

```bash
git diff --binary > local-work.patch
cd ..
git clone https://github.com/SKKU-IBE/Medical-SAM2GUI.git Medical-SAM2GUI-clean
cd Medical-SAM2GUI-clean
git apply ../Medical-SAM2GUI/local-work.patch
```

Review the patch before applying it and remove any checkpoint, medical image,
mask, or private path content.

## Maintainer verification

- Force-push every normal branch and tag from the sanitized mirror.
- Confirm that no removed path is reachable from branch or tag refs.
- Ask GitHub Support to expire cached views and server-managed pull-request refs
  that still expose pre-rewrite objects; pull-request refs cannot be force-pushed.
- Keep the private pre-rewrite backup access-restricted and outside the repository.
- Require contributors with old clones to reclone before accepting pushes.
