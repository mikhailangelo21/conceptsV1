# Handoff validation

The package was assembled from regular files, with no symlinks. [FILE_MANIFEST.json](FILE_MANIFEST.json) lists the copied evidence files with SHA-256 checksums, byte counts, original locations and roles. The package builder refuses to replace a previously copied file if its local content has been edited outside the builder.

Checks performed on 1 October 2026:

- Every listed evidence file matched its recorded SHA-256 value and size.
- Links in the handoff guides resolved within this folder.
- The copied fixture bank passed the versioned loader's hash, ID, split and schema checks: 6,322 selected cases, 1,888 selected comparison questions, 6,494 selected contrasts and 240 selected hue triplets.
- From this folder, the offline test suite produced `30 passed, 1 skipped, 1 deselected in 21.91s`. The integration test was deselected. The skip was an environment-dependent cached-tokenizer test; it does not indicate a failed scientific result.
- The original findings validation record independently recomputed 564 saved scalar prediction partitions and reconciled all 1,888 behavioral records. It is copied at [reports/quality_suite_v3_findings/validation.json](reports/quality_suite_v3_findings/validation.json).
- The illustrated HTML was previously verified at 1440px and 390px for rendering, overflow and source interaction. It is copied byte-for-byte from that checked artifact.

These checks establish that the handoff copied and organized the saved evidence. They do not add an independent scientific replication or restore the omitted activation/model caches.
