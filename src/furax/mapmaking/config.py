from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import yaml
from apischema import deserialize, serialize
from jax.typing import DTypeLike

ValidLandscapeType = Literal['WCS', 'Healpix']


@dataclass(frozen=True)
class SolverConfig:
    rtol: float = 1e-6
    atol: float = 0
    max_steps: int = 1_000


@dataclass(frozen=True)
class LandscapeConfig:
    type: ValidLandscapeType = 'WCS'
    resolution: float = 8.0
    nside: int = 512


@dataclass(frozen=True)
class _PolyTemplateConfig:
    max_poly_order: int = 3


@dataclass(frozen=True)
class _ScanSynchronousTemplateConfig:
    min_poly_order: int = 3
    max_poly_order: int = 7


@dataclass(frozen=True)
class _HWPSynchronousTemplateConfig:
    n_harmonics: int = 3


@dataclass(frozen=True)
class TemplatesConfig:
    polynomial: _PolyTemplateConfig = _PolyTemplateConfig()
    scan_synchronous: _ScanSynchronousTemplateConfig = _ScanSynchronousTemplateConfig()
    hwp_synchronous: _HWPSynchronousTemplateConfig = _HWPSynchronousTemplateConfig()


@dataclass(frozen=True)
class MapMakingConfig:
    binned: bool = False
    demodulated: bool = False
    scanning_mask: bool = False
    correlation_length: int = 1_000
    psd_fmin: float = 1e-2
    hits_cut: float = 1e-2
    cond_cut: float = 1e-2
    precision: Literal[32, 64] = 64
    debug: bool = True
    solver: SolverConfig = SolverConfig()
    landscape: LandscapeConfig = LandscapeConfig()
    templates: TemplatesConfig | None = None

    @classmethod
    def get_defaults(cls) -> 'MapMakingConfig':
        """Create a config with default values for all fields including optional ones."""
        return cls(templates=TemplatesConfig())

    @classmethod
    def load_yaml(cls, path: str | Path) -> 'MapMakingConfig':
        """Load and instantiate a ``MapMakingConfig`` from a YAML file."""
        data = yaml.safe_load(Path(path).read_text())
        return deserialize(MapMakingConfig, data)  # type: ignore[no-any-return]

    def dump_yaml(self, path: str | Path) -> None:
        """Dump the config to a YAML file.

        The '.yaml' suffix is automatically added if not already present.
        """
        filename = Path(path).with_suffix('.yaml')
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text(self._to_yaml())

    def _to_yaml(self) -> str:
        """Serialize the config to a YAML string."""
        data = serialize(MapMakingConfig, self)
        return yaml.dump(data, indent=2)

    @property
    def use_templates(self) -> bool:
        return self.templates is not None

    @property
    def dtype(self) -> DTypeLike:
        return jnp.float32 if self.precision == 32 else jnp.float64  # type: ignore[no-any-return]
