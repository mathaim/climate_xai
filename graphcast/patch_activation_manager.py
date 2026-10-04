#!/usr/bin/env python3
"""
Patch the forked graphcast ActivationManager to support `save_components`,
filtering saved activations by GNN component name (e.g. only "mesh_gnn").

Bug: _should_save() gated on step_idx but never the GNN name. Encoder
(grid2mesh_gnn) and decoder (mesh2grid_gnn) both run message-passing step 0,
so save_steps=[0] (Layer 0) matched all three GNNs -> grid2mesh/mesh2grid
strays written alongside mesh_gnn. This adds a component filter.

Idempotent + self-verifying. Run inside the extraction venv:
    PYTHONNOUSERSITE=1 python scripts/patches/patch_activation_manager.py
"""
import shutil, datetime, sys
import graphcast.deep_typed_graph_net as m
F = m.__file__
src = open(F).read()
if "save_components" in src:
    print("ALREADY PATCHED:", F); sys.exit(0)
edits = [
 ("signature",
  '                 mode: str = "post_res"):\n',
  '                 mode: str = "post_res",\n                 save_components: Optional[Sequence[str]] = None):\n'),
 ("init body",
  '        self.mode = mode\n',
  '        self.mode = mode\n        self.save_components = save_components\n'),
 ("should_save filter",
  '        if self.mode not in tag and self.mode != "both":\n            return False\n',
  '        if self.mode not in tag and self.mode != "both":\n            return False\n'
  '        if self.save_components is not None and tag.split("/")[0] not in self.save_components:\n            return False\n'),
]
for desc, old, _ in edits:
    n = src.count(old)
    if n != 1:
        print(f"ABORT: anchor '{desc}' matched {n}x (expected 1). Nothing written."); sys.exit(1)
bak = F + ".bak." + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
shutil.copy2(F, bak); print("Backup:", bak)
for desc, old, new in edits:
    src = src.replace(old, new, 1); print("applied:", desc)
open(F, "w").write(src); print("WROTE", F)
