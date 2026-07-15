# Repository History Rewrite

The Git history was rewritten on 2026-07-14 before the v1.1.0 release. The
rewrite removed checkpoint binaries and medical demonstration media for which
public redistribution rights were unavailable or insufficiently documented.
The sanitized repository initially contained only attributed public derived
media. It now also contains one explicitly licensed CC BY 4.0 FLAIR volume and
label map in `test_data/`; no checkpoint, private medical data, or media with
undocumented redistribution rights is included.

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

Review the patch before applying it and remove any checkpoint, private medical
image, undocumented mask, or private path content. The public fixture under
`test_data/` is the only approved source-volume exception.

## Maintainer verification

- Force-push every normal branch and tag from the sanitized mirror.
- Confirm that no removed path is reachable from branch or tag refs.
- Ask GitHub Support to expire cached views and server-managed pull-request refs
  that still expose pre-rewrite objects; pull-request refs cannot be force-pushed.
- Keep the private pre-rewrite backup access-restricted and outside the repository.
- Require contributors with old clones to reclone before accepting pushes.
