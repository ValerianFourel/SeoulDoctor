"""Authoritative hard-rule filtering over the complete facility catalog."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

import numpy as np
import pandas as pd

from query_facets import (
    DISEASE_ALIASES,
    GENDER_ALIASES,
    PLACE_ALIASES,
    SPECIALTY_ALIASES,
)

from .availability import has_tuesday_evening
from .contracts import AreaRule, DistanceRule, EligibleScope, SearchRules
from .rules import HARD_CONCEPT_ALIASES, SPECIALTY_IDS


EARTH_RADIUS_KM = 6371.0
SEARCHABLE_ATTRIBUTE_FIELDS = (
    "name",
    "category",
    "address",
    "business_hours",
    "Summaries",
    "Summaries_Korean",
    "Key_Highlights",
    "amenities",
    "medical_info_parsed",
)


class ScopeSchemaError(ValueError):
    """The facility catalog cannot satisfy the scope contract."""


def _normalized(value: Any) -> str:
    if value is None:
        value = ""
    elif not isinstance(value, str):
        try:
            if pd.isna(value):
                value = ""
        except (TypeError, ValueError):
            pass
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.sub(r"[^0-9a-z가-힣]+", " ", text).split())


def _slug(value: str) -> str:
    return _normalized(value).replace(" ", "_")


def _flatten_positive_values(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)):
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(nested, (bool, np.bool_)):
                if bool(nested):
                    yield str(key)
                continue
            yield str(key)
            yield from _flatten_positive_values(nested)
        return
    if isinstance(value, (list, tuple, set, frozenset, np.ndarray)):
        for item in value:
            yield from _flatten_positive_values(item)
        return
    try:
        if pd.isna(value):
            return
    except (TypeError, ValueError):
        pass
    yield str(value)


def _row_search_text(row: pd.Series) -> str:
    parts: list[str] = []
    for field in SEARCHABLE_ATTRIBUTE_FIELDS:
        if field not in row.index:
            continue
        parts.extend(_flatten_positive_values(row[field]))
    return _normalized(" ".join(parts))


def _matches_term(text: str, term: str) -> bool:
    normalized_term = _normalized(term)
    if not normalized_term:
        return False
    if re.fullmatch(r"[0-9a-z ]+", normalized_term):
        return bool(re.search(
            rf"(?<![0-9a-z]){re.escape(normalized_term)}(?![0-9a-z])",
            text,
        ))
    return normalized_term in text


def _matches_any(text: str, terms: Iterable[str]) -> bool:
    return any(_matches_term(text, term) for term in terms)


def _hard_concept_terms(concept_id: str) -> tuple[str, ...]:
    direct = HARD_CONCEPT_ALIASES.get(concept_id)
    if direct:
        return tuple(direct)
    for canonical, aliases in (*GENDER_ALIASES, *DISEASE_ALIASES):
        if _slug(canonical) == concept_id:
            return tuple((canonical, *aliases))
    if concept_id.startswith("literal:"):
        return (concept_id.removeprefix("literal:"),)
    return (concept_id.replace("_", " "),)


def _specialty_terms(specialty_id: str) -> tuple[str, ...]:
    for label, identifier in SPECIALTY_IDS.items():
        if identifier == specialty_id:
            aliases = next(
                aliases
                for canonical, aliases in SPECIALTY_ALIASES
                if canonical == label
            )
            return tuple((label, *aliases))
    if specialty_id.startswith("category:"):
        return (specialty_id.removeprefix("category:"),)
    raise ScopeSchemaError(f"unknown canonical specialty ID: {specialty_id}")


def _district_terms(area: AreaRule) -> tuple[str, ...]:
    canonical_id = area.area_id.removeprefix("district:")
    for canonical, aliases in PLACE_ALIASES:
        if _slug(canonical) == canonical_id:
            return tuple((canonical, *aliases))
    return (area.display_name, canonical_id)


def _haversine_series(
    origin_latitude: float,
    origin_longitude: float,
    latitudes: pd.Series,
    longitudes: pd.Series,
) -> pd.Series:
    latitude_values = pd.to_numeric(latitudes, errors="coerce")
    longitude_values = pd.to_numeric(longitudes, errors="coerce")
    phi_1 = np.radians(origin_latitude)
    phi_2 = np.radians(latitude_values)
    delta_phi = np.radians(latitude_values - origin_latitude)
    delta_lambda = np.radians(longitude_values - origin_longitude)
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi_1)
        * np.cos(phi_2)
        * np.sin(delta_lambda / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    return pd.Series(
        2.0 * EARTH_RADIUS_KM * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)),
        index=latitudes.index,
        dtype=float,
    )


@dataclass(frozen=True)
class ScopeSelection:
    """Complete eligible IDs plus optional exact distance annotations."""

    descriptor: EligibleScope
    facility_ids: tuple[str, ...]
    distance_km_by_facility: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.descriptor.facility_count != len(self.facility_ids):
            raise ValueError("scope count must equal the complete facility ID count")
        if len(set(self.facility_ids)) != len(self.facility_ids):
            raise ValueError("facility IDs in a scope must be unique")
        if not set(self.distance_km_by_facility).issubset(self.facility_ids):
            raise ValueError("distance annotations must belong to the scope")
        object.__setattr__(
            self,
            "distance_km_by_facility",
            MappingProxyType(dict(self.distance_km_by_facility)),
        )

    def restrict_dataframe(self, facilities: pd.DataFrame) -> pd.DataFrame:
        """Adapt this complete scope to the legacy dataframe RAG boundary."""
        if "place_id" not in facilities.columns:
            raise ScopeSchemaError("facility catalog requires a place_id column")
        allowed = frozenset(self.facility_ids)
        identifiers = facilities["place_id"].astype(str)
        scoped = facilities[identifiers.isin(allowed)].copy()
        if self.distance_km_by_facility:
            scoped["distance_km"] = scoped["place_id"].astype(str).map(
                self.distance_km_by_facility
            )
        scoped.attrs["eligible_scope"] = {
            "index_version": self.descriptor.index_version,
            "rules_hash": self.descriptor.rules_hash,
            "scope_digest": self.descriptor.scope_digest,
            "facility_count": self.descriptor.facility_count,
        }
        return scoped

    def assert_contains_only(self, facilities: pd.DataFrame) -> None:
        """Reject a downstream dataframe that escaped the hard scope."""
        if "place_id" not in facilities.columns:
            raise ScopeSchemaError("candidate dataframe requires a place_id column")
        unexpected = set(facilities["place_id"].astype(str)) - set(self.facility_ids)
        if unexpected:
            raise ValueError("candidate dataframe contains facilities outside scope")


class ScopeBuilder:
    """Apply every hard rule once over the full facility catalog."""

    def build(
        self,
        facilities: pd.DataFrame,
        rules: SearchRules,
        *,
        index_version: str,
    ) -> ScopeSelection:
        self._validate_catalog(facilities, rules)
        identifiers = facilities["place_id"].astype(str)
        eligible = pd.Series(True, index=facilities.index, dtype=bool)

        if rules.hard.prohibited_facility_ids:
            eligible &= ~identifiers.isin(rules.hard.prohibited_facility_ids)

        if rules.hard.specialty_ids:
            categories = facilities["category"].map(_normalized)
            accepted_categories = {
                _normalized(term)
                for specialty_id in rules.hard.specialty_ids
                for term in _specialty_terms(specialty_id)
            }
            eligible &= categories.isin(accepted_categories)

        distances = pd.Series(np.nan, index=facilities.index, dtype=float)
        geography = rules.hard.geography
        if isinstance(geography, DistanceRule):
            distances = _haversine_series(
                geography.anchor.latitude,
                geography.anchor.longitude,
                facilities["lat"],
                facilities["lon"],
            )
            eligible &= distances.notna() & (distances <= geography.max_km + 1e-9)
        elif geography.area_kind == "district":
            terms = {_normalized(term) for term in _district_terms(geography)}
            district_match = pd.Series(False, index=facilities.index, dtype=bool)
            if "file_district" in facilities.columns:
                district_match |= facilities["file_district"].map(_normalized).isin(terms)
            if "address" in facilities.columns:
                address_terms = (geography.display_name,)
                district_match |= facilities["address"].map(
                    lambda value: _matches_any(
                        _normalized(value),
                        address_terms,
                    )
                )
            eligible &= district_match

        needs_search_text = bool(
            rules.hard.required_attributes
            or rules.hard.prohibited_taxonomy_ids
        )
        search_text = (
            facilities.apply(_row_search_text, axis=1)
            if needs_search_text
            else None
        )
        if search_text is not None:
            for requirement in rules.hard.required_attributes:
                if requirement.concept_id == "hours:tuesday_evening":
                    if "business_hours" not in facilities.columns:
                        eligible &= False
                    else:
                        eligible &= facilities["business_hours"].map(
                            has_tuesday_evening
                        )
                    continue
                terms = (*requirement.terms_en, *requirement.terms_ko)
                eligible &= search_text.map(lambda text: _matches_any(text, terms))
            for concept_id in rules.hard.prohibited_taxonomy_ids:
                terms = _hard_concept_terms(concept_id)
                eligible &= ~search_text.map(lambda text: _matches_any(text, terms))

        facility_ids = tuple(identifiers[eligible].tolist())
        distance_by_facility = {
            identifiers.at[index]: float(distances.at[index])
            for index in facilities.index[eligible]
            if pd.notna(distances.at[index])
        }
        scope_digest = self._scope_digest(
            index_version,
            rules.rules_hash,
            facility_ids,
        )
        descriptor = EligibleScope(
            index_version=index_version,
            rules_hash=rules.rules_hash,
            facility_bitmap_ref=f"memory://{scope_digest}",
            facility_count=len(facility_ids),
            scope_digest=scope_digest,
        )
        return ScopeSelection(
            descriptor=descriptor,
            facility_ids=facility_ids,
            distance_km_by_facility=distance_by_facility,
        )

    @staticmethod
    def _validate_catalog(
        facilities: pd.DataFrame,
        rules: SearchRules,
    ) -> None:
        if not isinstance(facilities, pd.DataFrame):
            raise ScopeSchemaError("facilities must be a pandas DataFrame")
        if "place_id" not in facilities.columns:
            raise ScopeSchemaError("facility catalog requires a place_id column")
        identifiers = facilities["place_id"].fillna("").astype(str).str.strip()
        if identifiers.eq("").any():
            raise ScopeSchemaError("facility IDs cannot be empty")
        if identifiers.duplicated().any():
            raise ScopeSchemaError("facility IDs must be unique")
        if rules.hard.specialty_ids and "category" not in facilities.columns:
            raise ScopeSchemaError("specialty rules require a category column")
        if isinstance(rules.hard.geography, DistanceRule):
            missing = {"lat", "lon"} - set(facilities.columns)
            if missing:
                raise ScopeSchemaError("distance rules require lat and lon columns")
        if (
            isinstance(rules.hard.geography, AreaRule)
            and rules.hard.geography.area_kind == "district"
            and not {"file_district", "address"}.intersection(facilities.columns)
        ):
            raise ScopeSchemaError(
                "district rules require file_district or address"
            )

    @staticmethod
    def _scope_digest(
        index_version: str,
        rules_hash: str,
        facility_ids: Sequence[str],
    ) -> str:
        payload = json.dumps(
            {
                "index_version": index_version,
                "rules_hash": rules_hash,
                "facility_ids": sorted(facility_ids),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(payload.encode("utf-8")).hexdigest()
