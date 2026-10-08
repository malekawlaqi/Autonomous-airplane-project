"""Runway geometry: a local runway-relative frame used for every learning signal.

Frame definition (all lengths in metres):
    x  = distance along the runway heading, measured from the threshold
    y  = lateral offset from the centerline, positive to the RIGHT of the heading
Heading is measured clockwise from true north, in degrees (JSBSim convention).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

FT_PER_M: float = 3.280839895
M_PER_FT: float = 1.0 / FT_PER_M
KT_PER_FPS: float = 0.592484
FPS_PER_KT: float = 1.0 / KT_PER_FPS
EARTH_RADIUS_M: float = 6_371_000.0


def wrap_angle_rad(angle_rad: float) -> float:
    """Wrap an angle to the interval [-pi, pi].

    Args:
        angle_rad: Angle in radians.

    Returns:
        Equivalent angle in [-pi, pi].
    """
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def wrap_angle_deg(angle_deg: float) -> float:
    """Wrap an angle to the interval [-180, 180] degrees."""
    return (angle_deg + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class Runway:
    """A straight runway described in a local frame.

    Attributes:
        length_m: Usable takeoff run available from the threshold.
        width_m: Paved width.
        heading_deg: Runway true heading (clockwise from north).
        elevation_ft: Runway elevation above mean sea level.
        threshold_lat_deg: Geocentric latitude of the threshold.
        threshold_lon_deg: Longitude of the threshold.
        rolling_friction_factor: Multiplier on the FDM rolling friction (1.0 = dry baseline).
        static_friction_factor: Multiplier on the FDM static friction.
        surface: Human-readable surface label ("dry", "wet", "contaminated").
    """

    length_m: float
    width_m: float
    heading_deg: float
    elevation_ft: float
    threshold_lat_deg: float = 47.0
    threshold_lon_deg: float = -122.0
    rolling_friction_factor: float = 1.0
    static_friction_factor: float = 1.0
    surface: str = "dry"

    @property
    def half_width_m(self) -> float:
        """Half of the paved width."""
        return 0.5 * self.width_m

    @property
    def heading_rad(self) -> float:
        """Runway heading in radians."""
        return math.radians(self.heading_deg)

    def ned_to_runway(self, north_m: float, east_m: float) -> tuple[float, float]:
        """Convert a north/east displacement to runway (along, lateral-right) coordinates.

        Args:
            north_m: Displacement towards north.
            east_m: Displacement towards east.

        Returns:
            Tuple ``(along_m, lateral_right_m)``.
        """
        psi = self.heading_rad
        along = north_m * math.cos(psi) + east_m * math.sin(psi)
        lateral = -north_m * math.sin(psi) + east_m * math.cos(psi)
        return along, lateral

    def runway_to_ned(self, along_m: float, lateral_m: float) -> tuple[float, float]:
        """Inverse of :meth:`ned_to_runway`.

        Returns:
            Tuple ``(north_m, east_m)``.
        """
        psi = self.heading_rad
        north = along_m * math.cos(psi) - lateral_m * math.sin(psi)
        east = along_m * math.sin(psi) + lateral_m * math.cos(psi)
        return north, east

    def runway_to_latlon(self, along_m: float, lateral_m: float) -> tuple[float, float]:
        """Return the (lat_deg, lon_deg) of a point in runway coordinates (flat-earth, short range)."""
        north, east = self.runway_to_ned(along_m, lateral_m)
        dlat = math.degrees(north / EARTH_RADIUS_M)
        dlon = math.degrees(east / (EARTH_RADIUS_M * math.cos(math.radians(self.threshold_lat_deg))))
        return self.threshold_lat_deg + dlat, self.threshold_lon_deg + dlon

    def remaining_m(self, along_m: float) -> float:
        """Runway distance remaining ahead of ``along_m`` (negative once past the end)."""
        return self.length_m - along_m

    def wind_components(self, wind_north_mps: float, wind_east_mps: float) -> tuple[float, float]:
        """Express a wind vector (the air-mass velocity) as headwind/crosswind.

        Returns:
            Tuple ``(headwind_mps, crosswind_from_right_mps)``.  Headwind > 0 opposes the
            takeoff run.  Crosswind > 0 means the air moves from the right side of the runway
            towards the left (i.e. a wind blowing *from* the right).
        """
        along, lateral = self.ned_to_runway(wind_north_mps, wind_east_mps)
        return -along, -lateral

    def wind_ned_from_components(self, headwind_mps: float, crosswind_from_right_mps: float) -> tuple[float, float]:
        """Build a (north, east) air-mass velocity from headwind and crosswind components."""
        return self.runway_to_ned(-headwind_mps, -crosswind_from_right_mps)
