from pathlib import Path
import petsc4py
from petsc4py import PETSc

print("PETSc version:", PETSc.Sys.getVersion())
print("hypre available:", PETSc.Sys.hasExternalPackage("hypre"))

config = petsc4py.get_config()
print("Build configuration:", config)
root = Path(config["PETSC_DIR"])
arch = config.get("PETSC_ARCH", "")

for folder in dict.fromkeys([root / arch / "include", root / "include"]):
    for filename in ("petscconf.h", "HYPRE_config.h"):
        path = folder / filename
        if path.exists():
            print(f"\n{path}")
            for line in path.read_text().splitlines():
                if any(flag in line for flag in (
                    "PETSC_HAVE_CUDA", "PETSC_HAVE_HYPRE",
                    "HYPRE_USING_CUDA", "HYPRE_USING_GPU",
                )):
                    print(line)