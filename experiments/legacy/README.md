# Historical experiment — unsupported

These files preserve the initial commit `d345650` for inspection. They are not
installed, imported or exercised by the supported package. The original README
is saved as `README.original.md`; its claims and commands are historical, not
current instructions. The dependency lists include unrelated packages and must
not be used to install the supported implementation.

The OLMoE route has random attention/router components and is not a faithful
pretrained OLMoE implementation. The original GLM4 path repeats the full prefix
on every token and contains export/dtype/API drift defects. The routing-log and
oracle-training code does not establish native routing prediction quality.

Use the root README and `src/disk_moe/` for the supported baseline. New safe
exports use a versioned safetensors artifact; legacy `.pt` exports must be
re-exported from the original trusted checkpoint, not unpickled automatically.
