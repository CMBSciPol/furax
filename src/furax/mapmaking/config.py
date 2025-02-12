from pathlib import Path
from typing import Literal

import jax.numpy as jnp
from attrs import frozen
from cattrs.preconf.pyyaml import make_converter
from jax.typing import DTypeLike

ValidLandscapeType = Literal['WCS', 'Healpix']


@frozen
class SolverConfig:
    rtol: float = 1e-6
    atol: float = 0
    max_steps: int = 1_000


@frozen
class LandscapeConfig:
    type: ValidLandscapeType = 'WCS'
    resolution: float = 8.0
    nside: int = 512


@frozen
class _PolyTemplateConfig:
    max_poly_order: int = 3


@frozen
class _ScanSynchronousTemplateConfig:
    min_poly_order: int = 3
    max_poly_order: int = 7


@frozen
class _HWPSynchronousTemplateConfig:
    n_harmonics: int = 3


@frozen
class TemplatesConfig:
    polynomial: _PolyTemplateConfig = _PolyTemplateConfig()
    scan_synchronous: _ScanSynchronousTemplateConfig = _ScanSynchronousTemplateConfig()
    hwp_synchronous: _HWPSynchronousTemplateConfig = _HWPSynchronousTemplateConfig()


# forbid extra keys in the yaml file to catch possible typos
_yaml_converter = make_converter(forbid_extra_keys=True)


@frozen
class MapMakingConfig:
    binned: bool = False
    demodulated: bool = False
    scanning_mask: bool = False
    correlation_length: int = 1_000
    psd_fmin: float = 1e-2
    hits_cut: float = 1e-2
    cond_cut: float = 1e-2
    dtype: DTypeLike = jnp.float64
    debug: bool = True
    solver: SolverConfig = SolverConfig()
    landscape: LandscapeConfig = LandscapeConfig()
    templates: TemplatesConfig | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'MapMakingConfig':
        """Create a Config from a yaml file."""
        return _yaml_converter.loads(Path(path).read_text(), cls)  # type: ignore[no-any-return]

    @classmethod
    def get_defaults(cls) -> 'MapMakingConfig':
        """Create a Config with default values for all fields including optional ones."""
        return cls(templates=TemplatesConfig())  # type: ignore[call-arg]

    def to_yaml(self, path: str | Path) -> None:
        """Serialize the Config to a yaml file."""
        filename = Path(path).with_suffix('.yaml')
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text(_yaml_converter.dumps(self))

    @property
    def use_templates(self) -> bool:
        return self.templates is not None


if __name__ == '__main__':
    config = MapMakingConfig.get_defaults()
    config.to_yaml('defaults.yaml')
