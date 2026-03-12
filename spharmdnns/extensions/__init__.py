# import spharmdnns.extensions.TriangleSearchCUDA as TriangleSearchCUDA

# __all__ = ["TriangleSearchCUDA"]


# spharmdnns/extensions/__init__.py
from importlib import import_module
from pathlib import Path
from torch.utils.cpp_extension import load

try:
    # 이미 빌드된 확장 모듈이 있으면 사용
    TriangleSearchCUDA = import_module("spharmdnns.extensions.TriangleSearchCUDA")
except ModuleNotFoundError:
    # 없으면 csrc에서 JIT 컴파일 후 로드
    src = Path(__file__).resolve().parent.parent / "csrc"
    TriangleSearchCUDA = load(
        name="spharmdnns_extensions_TriangleSearchCUDA",
        sources=[str(src / "triangle_search.cpp"),
                 str(src / "triangle_search_kernel.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )

__all__ = ["TriangleSearchCUDA"]