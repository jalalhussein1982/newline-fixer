# 0003. Publish weights to the Hugging Face Hub, pin the revision, keep them out of git

Date: 2026-10-01. Status: accepted.

## Context

The submission is a git bundle. The fine-tuned encoder is on the order of a hundred
megabytes or more. The service must be runnable by a reviewer from the bundle.

## Options

1. **Commit weights to git.** Self-contained, but bloats the bundle and every clone,
   and git handles large binaries badly.
2. **Git LFS.** Cleaner, but a bundle does not carry LFS objects, so the reviewer would
   still need a remote.
3. **Hugging Face Hub model repository, pinned revision, downloaded at image build.**
   Small bundle, reproducible, and the Hub page doubles as the model card.
4. **Train in the Dockerfile.** Reproducible in principle, far too slow and fragile.

## Decision

Option 3, with a documented local-weights path for an offline build.

## Consequences

- The build needs network access once. The revision is pinned in the configuration.
- The model card and evaluation table live in two places and must be kept in sync by
  the publish script.
