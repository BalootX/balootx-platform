-- Research Scenario V4 query projections are rebuildable views of immutable results.
-- Version 3 adds scenario, proof, project, exact-state, and decision facets.
ALTER TABLE research.results
  ALTER COLUMN projection_version SET DEFAULT 3;

UPDATE research.results
SET projection_version = 0
WHERE projection_version < 3;
