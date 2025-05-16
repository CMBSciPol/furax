from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path

import jax.numpy as jnp
import yaml
from apischema import deserialize, serialize
from jax.typing import DTypeLike


class Landscapes(Enum):
    WCS = 'WCS'
    HPIX = 'Healpix'


class Methods(Enum):
    BINNED = 'Binned'
    MAXL = 'ML'
    TWOSTEP = 'TwoStep'
    ATOP = 'ATOP'


@dataclass(frozen=True)
class SolverConfig:
    rtol: float = 1e-6
    atol: float = 0
    max_steps: int = 1_000


@dataclass(frozen=True)
class LandscapeConfig:
    type: Landscapes = Landscapes.WCS
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
    polynomial: _PolyTemplateConfig | None = None
    scan_synchronous: _ScanSynchronousTemplateConfig | None = None
    hwp_synchronous: _HWPSynchronousTemplateConfig | None = None

    @classmethod
    def full_defaults(cls) -> 'TemplatesConfig':
        """Create a template config with default values for all templates."""
        return cls(
            polynomial=_PolyTemplateConfig(),
            scan_synchronous=_ScanSynchronousTemplateConfig(),
            hwp_synchronous=_HWPSynchronousTemplateConfig(),
        )

    @property
    def empty(self) -> bool:
        return all(getattr(self, f.name) is None for f in fields(self))


@dataclass(frozen=True)
class MapMakingConfig:
    method: Methods = Methods.BINNED
    binned: bool = True
    demodulated: bool = False
    scanning_mask: bool = False
    correlation_length: int = 1_000
    nperseg: int = 1_024
    psd_fmin: float = 1e-2
    hits_cut: float = 1e-2
    cond_cut: float = 1e-2
    double_precision: bool = True
    pointing_on_the_fly: bool = False
    pointing_chunk_size: int = 4
    fit_noise_model: bool = True
    debug: bool = True
    solver: SolverConfig = SolverConfig()
    landscape: LandscapeConfig = LandscapeConfig()
    templates: TemplatesConfig | None = None
    atop_tau: int = 0

    @classmethod
    def full_defaults(cls) -> 'MapMakingConfig':
        """Create a config with default values for all fields including optional ones."""
        return cls(templates=TemplatesConfig.full_defaults())

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
        return (self.templates is not None) and (not self.templates.empty)

    @property
    def dtype(self) -> DTypeLike:
        return jnp.float64 if self.double_precision else jnp.float32  # type: ignore[no-any-return]
