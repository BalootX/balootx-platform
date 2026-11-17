import type {JsonObject, JsonValue} from '../../shared/json';
import type {ConfigProfile} from '../configuration/model';

export type ProfileComparisonChange = {
  path: string;
  change: 'added' | 'removed' | 'changed';
  left?: JsonValue;
  right?: JsonValue;
};

export type ProfileComparison = {
  left: ConfigProfile;
  right: ConfigProfile;
  changes: ProfileComparisonChange[];
};

export type RulesAuthorityIdentity = {
  profileId: string;
  version: string;
  profileStatus: string;
  expertRatified: boolean;
  payloadSha256: string;
};

export type RulesCertificationBoundary = {
  status: string;
  certificationEligible: boolean;
  sourceCatalog: {
    id: string;
    version: string;
    sha256: string;
  };
};

function object(value: JsonValue | undefined): JsonObject {
  return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
}

function requiredString(value: JsonValue | undefined, code: string): string {
  if (typeof value !== 'string' || !value) throw new Error(code);
  return value;
}

export function rulesAuthorityIdentity(
  profile: ConfigProfile,
  expectedRevisionId: string,
  expectedRulesHash: string,
): RulesAuthorityIdentity {
  if (
    profile.family !== 'rules'
    || profile.revisionId !== expectedRevisionId
    || profile.payloadSha256 !== expectedRulesHash
  ) throw new Error('research_rule_authority_identity_mismatch');
  const authority = object(profile.payload?.profile);
  const identity = {
    profileId: requiredString(authority.id, 'research_rule_authority_profile_id_missing'),
    version: requiredString(authority.version, 'research_rule_authority_version_missing'),
    profileStatus: requiredString(authority.status, 'research_rule_authority_status_missing'),
    expertRatified: authority.expert_ratified === true,
    payloadSha256: profile.payloadSha256,
  };
  if (identity.profileId !== profile.profileId || identity.version !== profile.version) {
    throw new Error('research_rule_authority_registry_identity_mismatch');
  }
  return identity;
}

export function rulesCertificationBoundary(
  profile: ConfigProfile,
  rules: RulesAuthorityIdentity,
): RulesCertificationBoundary {
  if (profile.family !== 'certification') {
    throw new Error('research_rule_authority_certification_family_mismatch');
  }
  const payload = object(profile.payload);
  const authority = object(payload.profile);
  const candidate = object(payload.candidate_profile_identity);
  const sourceCatalog = object(payload.rule_source_catalog);
  if (
    candidate.id !== rules.profileId
    || candidate.version !== rules.version
    || candidate.sha256 !== rules.payloadSha256
  ) throw new Error('research_rule_authority_certification_identity_mismatch');
  return {
    status: requiredString(authority.status, 'research_rule_authority_boundary_missing'),
    certificationEligible: payload.certification_eligible === true,
    sourceCatalog: {
      id: requiredString(sourceCatalog.id, 'research_rule_authority_catalog_id_missing'),
      version: requiredString(sourceCatalog.version, 'research_rule_authority_catalog_version_missing'),
      sha256: requiredString(sourceCatalog.sha256, 'research_rule_authority_catalog_hash_missing'),
    },
  };
}

export function semanticRulesDelta(comparison: ProfileComparison): ProfileComparisonChange[] {
  if (comparison.left.family !== 'rules' || comparison.right.family !== 'rules') {
    throw new Error('research_rule_authority_comparison_family_mismatch');
  }
  return comparison.changes.filter((item)=>!item.path.startsWith('profile.'));
}
