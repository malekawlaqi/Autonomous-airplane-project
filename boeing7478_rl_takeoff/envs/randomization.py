"""Domain randomization: sample a complete, self-consistent episode specification."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from envs.runway import Runway

NUM_LEVELS: int = 4


@dataclass(frozen=True)
class LoadingConditions:
    """Aircraft loading for one episode.

    Attributes:
        fuel_lbs: Fuel on board (distributed over the FDM tanks, which sit at the CG station).
        payload_lbs: Payload represented as a single point mass.
        payload_station_in: Fuselage station (x, inches) of the payload point mass.
        requested_cg_shift_in: CG shift (relative to the empty-aircraft CG) that was requested.
    """

    fuel_lbs: float
    payload_lbs: float
    payload_station_in: float
    requested_cg_shift_in: float


@dataclass(frozen=True)
class WeatherConditions:
    """Atmosphere and wind for one episode (wind is runway-relative)."""

    delta_isa_c: float
    qnh_hpa: float
    headwind_kt: float
    crosswind_from_right_kt: float
    gust_sigma_kt: float
    gust_time_constant_s: float
    turbulence_severity: float


@dataclass(frozen=True)
class EpisodeConditions:
    """Everything that is random about one episode."""

    seed: int
    level: int
    runway: Runway
    loading: LoadingConditions
    weather: WeatherConditions
    initial_lateral_offset_m: float
    initial_heading_offset_deg: float
    start_along_m: float

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a plain dictionary (for logging/CSV)."""
        return asdict(self)


class DomainRandomizer:
    """Samples :class:`EpisodeConditions` from curriculum-level ranges in YAML."""

    def __init__(self, randomization_cfg: dict[str, Any], aircraft_cfg: dict[str, Any]) -> None:
        """Create a sampler.

        Args:
            randomization_cfg: Parsed ``domain_randomization.yaml``.
            aircraft_cfg: Parsed ``aircraft.yaml`` (for OEW, MTOW, fuel capacity, payload limits).
        """
        self.cfg = randomization_cfg
        self.levels: dict[int, dict[str, list[float]]] = {int(k): v for k, v in randomization_cfg["levels"].items()}
        sampling = randomization_cfg["sampling"]
        self.runway_width_m = float(sampling["runway_width_m"])
        self.gust_tau_s = float(sampling["gust_time_constant_s"])
        self.lower_level_probability = float(sampling["lower_level_probability"])
        ov = aircraft_cfg["overrides"]
        lim = aircraft_cfg["limits"]
        self.oew_lbs = float(ov["empty_weight_lbs"]["value"])
        self.fuel_capacity_lbs = float(ov["fuel_capacity_total_lbs"]["value"])
        self.mtow_lbs = float(lim["mtow_lbs"])
        self.max_payload_lbs = float(lim["max_payload_lbs"])
        self.nominal_cg_x_in = float(lim["nominal_cg_x_in"])
        self.payload_station_range_in = tuple(float(v) for v in lim["payload_station_range_in"])

    def choose_level(self, current_level: int, rng: np.random.Generator) -> int:
        """Pick the level used for an episode (occasionally a lower one, to limit forgetting)."""
        if current_level > 1 and rng.random() < self.lower_level_probability:
            return int(rng.integers(1, current_level))
        return int(current_level)

    def _uniform(self, rng: np.random.Generator, bounds: list[float]) -> float:
        """Sample uniformly from ``[bounds[0], bounds[1]]`` (degenerate ranges allowed)."""
        low, high = float(bounds[0]), float(bounds[1])
        return float(low if high <= low else rng.uniform(low, high))

    def _sample_loading(self, rng: np.random.Generator, spec: dict[str, list[float]]) -> LoadingConditions:
        """Sample fuel/payload/CG so that the weight stays within published limits."""
        target_weight = self._uniform(rng, spec["total_weight_frac_mtow"]) * self.mtow_lbs
        extra = max(target_weight - self.oew_lbs, 0.0)
        payload_cap = min(self.max_payload_lbs, extra)
        payload = self._uniform(rng, spec["payload_frac_of_max"]) * payload_cap
        fuel = extra - payload
        if fuel > self.fuel_capacity_lbs:  # not enough tank capacity: move the excess to payload
            payload = min(payload + (fuel - self.fuel_capacity_lbs), self.max_payload_lbs)
            fuel = self.fuel_capacity_lbs
        total = self.oew_lbs + payload + fuel
        shift = self._uniform(rng, spec["cg_shift_in"])
        low, high = self.payload_station_range_in
        if payload > 1000.0:
            station = float(np.clip(self.nominal_cg_x_in + shift * total / payload, low, high))
        else:
            station = self.nominal_cg_x_in
        return LoadingConditions(fuel_lbs=float(fuel), payload_lbs=float(payload),
                                 payload_station_in=station, requested_cg_shift_in=float(shift))

    def _sample_runway(self, rng: np.random.Generator, spec: dict[str, list[float]]) -> Runway:
        """Sample runway geometry, elevation and surface condition."""
        static_f = self._uniform(rng, spec["static_friction_factor"])
        rolling_f = self._uniform(rng, spec["rolling_friction_factor"])
        if static_f >= 0.9:
            surface = "dry"
        elif static_f >= 0.55:
            surface = "wet"
        else:
            surface = "contaminated"
        return Runway(length_m=self._uniform(rng, spec["runway_length_m"]), width_m=self.runway_width_m,
                      heading_deg=self._uniform(rng, spec["runway_heading_deg"]),
                      elevation_ft=self._uniform(rng, spec["runway_elevation_ft"]),
                      rolling_friction_factor=rolling_f, static_friction_factor=static_f, surface=surface)

    def _sample_weather(self, rng: np.random.Generator, spec: dict[str, list[float]]) -> WeatherConditions:
        """Sample temperature, pressure, wind, gust and turbulence."""
        crosswind = self._uniform(rng, spec["crosswind_abs_kt"]) * float(rng.choice([-1.0, 1.0]))
        return WeatherConditions(
            delta_isa_c=self._uniform(rng, spec["delta_isa_c"]), qnh_hpa=self._uniform(rng, spec["qnh_hpa"]),
            headwind_kt=self._uniform(rng, spec["headwind_kt"]), crosswind_from_right_kt=crosswind,
            gust_sigma_kt=self._uniform(rng, spec["gust_sigma_kt"]), gust_time_constant_s=self.gust_tau_s,
            turbulence_severity=self._uniform(rng, spec["turbulence_severity"]))

    def sample(self, level: int, seed: int) -> EpisodeConditions:
        """Sample one episode's conditions.

        Args:
            level: Curriculum level in ``1..4``.
            seed: Episode seed. The same ``(level, seed)`` always yields the same conditions.

        Returns:
            Fully specified :class:`EpisodeConditions`.
        """
        if level not in self.levels:
            raise ValueError(f"Unknown curriculum level {level}; valid levels: {sorted(self.levels)}")
        rng = np.random.default_rng(seed)
        spec = self.levels[level]
        return EpisodeConditions(
            seed=int(seed), level=int(level), runway=self._sample_runway(rng, spec),
            loading=self._sample_loading(rng, spec), weather=self._sample_weather(rng, spec),
            initial_lateral_offset_m=self._uniform(rng, spec["initial_lateral_offset_m"]),
            initial_heading_offset_deg=self._uniform(rng, spec["initial_heading_offset_deg"]),
            start_along_m=self._uniform(rng, spec["start_along_m"]))
