#ifndef BALOOT_NATIVE_H
#define BALOOT_NATIVE_H
#include <stdint.h>
#ifdef _WIN32
# ifdef BALOOT_NATIVE_BUILD
#  define BALOOT_API __declspec(dllexport)
# else
#  define BALOOT_API __declspec(dllimport)
# endif
#else
# define BALOOT_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
#define BALOOT_NATIVE_ABI_V1 1u
#define BALOOT_NATIVE_ABI_V2 2u
/* Compatibility alias: V1 callers keep seeing the V1 contract. */
#define BALOOT_NATIVE_ABI_VERSION BALOOT_NATIVE_ABI_V1
#define BALOOT_NATIVE_CAP_LEGAL_MOVES 1u
#define BALOOT_NATIVE_CAP_BATCH_LEGAL_MOVES 2u
#define BALOOT_NATIVE_CAP_CARD_POINTS 4u
#define BALOOT_NATIVE_CAP_TRICK_EVALUATION 8u
#define BALOOT_NATIVE_CAP_STATE_HASH 16u
#define BALOOT_NATIVE_CAP_APPLY_PLAY 32u
#define BALOOT_NATIVE_CAP_APPLY_UNDO 64u
#define BALOOT_NATIVE_CAP_EXACT_SEARCH 128u
#define BALOOT_NATIVE_ABI_V3 3u

#define BALOOT_NATIVE_SEARCH_EXACT 0u
#define BALOOT_NATIVE_SEARCH_BUDGET_EXHAUSTED 1u
#define BALOOT_NATIVE_SEARCH_CANCELLED 2u
#define BALOOT_NATIVE_SEARCH_FAILED 3u
#define BALOOT_NATIVE_SEARCH_HEURISTIC_COMPLETE 4u

/* Roll a contracted play state from its current position without applying a card. */
#define BALOOT_NATIVE_ROLLOUT_NO_INITIAL_CARD 255u

typedef struct baloot_rules_v1 {
 uint32_t abi_version; int32_t sun_strength[8]; int32_t hokum_non_trump_strength[8];
 int32_t hokum_trump_strength[8]; uint8_t allow_discard_when_cannot_overtrump;
 uint8_t partner_winning_exception; uint8_t locked_hokum_blocks_trump_lead; uint8_t reserved;
} baloot_rules_v1;
typedef struct baloot_play_v1 { uint8_t seat,card,ekka,reserved; } baloot_play_v1;
typedef struct baloot_state_v1 {
 uint32_t abi_version; uint32_t hands[4]; uint8_t contract; int8_t trump_suit;
 uint8_t current_player,locked,trick_size,reserved[3]; baloot_play_v1 trick[3];
} baloot_state_v1;

typedef struct baloot_rules_v2 {
 uint32_t abi_version;
 int32_t strengths[3][8]; /* sun, hokum non-trump, hokum trump */
 int16_t points[3][8];
 uint16_t sequence_values_sun[3]; /* three, four, five+ */
 uint16_t sequence_values_hokum[3];
 uint16_t four_kind_values_sun[8];
 uint16_t four_kind_values_hokum[8];
 uint16_t last_trick_bonus_raw,baloot_game_points,kabout_sun,kabout_hokum,match_target;
 uint8_t must_follow_suit,must_trump_when_void,must_overtrump_when_possible;
 uint8_t allow_discard_when_cannot_overtrump,partner_winning_exception;
 uint8_t locked_hokum_blocks_trump_lead,ekka_enabled,projects_enabled;
 uint8_t reserved[8];
} baloot_rules_v2;
typedef struct baloot_play_v2 { uint8_t seat,card,ekka,baloot; } baloot_play_v2;
typedef struct baloot_state_v2 {
 uint32_t abi_version; uint32_t hands[4]; uint32_t initial_hands[4];
 uint8_t contract; int8_t trump_suit; uint8_t current_player,locked,trump_broken;
 uint8_t buyer,dealer,first_leader,trick_size,completed_tricks,escalation_multiplier;
 uint8_t reserved0[2]; int32_t raw_points[2]; int32_t project_points[2];
 int32_t baloot_points[2]; int32_t match_score[2]; baloot_play_v2 trick[3]; uint8_t reserved1[4];
} baloot_state_v2;
typedef struct baloot_trick_result_v2 { int32_t winner; int32_t raw_points; } baloot_trick_result_v2;
typedef struct baloot_undo_v2 { baloot_state_v2 previous; } baloot_undo_v2;

/*
 * V3 keeps the proven V2 transition state intact and adds only the scoring
 * inputs required for a complete, coarse-grained endgame solve.  The entire
 * tree stays native; callers do not cross the ABI once per node.
 */
typedef struct baloot_rules_v3 {
 baloot_rules_v2 base;
 uint16_t card_game_point_totals[2]; /* sun, hokum */
 uint8_t project_multiplier_cap,baloot_multiplier_cap,kabout_enabled,reserved[5];
} baloot_rules_v3;
typedef struct baloot_state_v3 {
 baloot_state_v2 base;
 uint8_t declarer,counting_team,escalation_mode;
 uint8_t tricks_won[2];
 uint8_t baloot_eligible_seats_mask,baloot_declared_teams_mask,reserved[1];
} baloot_state_v3;
typedef struct baloot_search_limits_v3 {
 uint64_t maximum_nodes;
 uint32_t maximum_milliseconds,maximum_cache_entries;
 uint8_t maximum_depth,team,reserved[6];
} baloot_search_limits_v3;
typedef struct baloot_search_result_v3 {
 uint64_t nodes;
 int32_t value,own_points,opponent_points;
 int32_t score_after_round[2];
 int8_t match_winner;
 uint8_t exact,status,principal_variation_length;
 uint8_t contract_succeeded,round_lead,reserved[3];
 uint8_t principal_variation[32];
} baloot_search_result_v3;
typedef struct baloot_rollout_result_v3 {
 uint64_t nodes;
 int32_t value,own_points,opponent_points;
 int32_t score_after_round[2];
 int8_t match_winner;
 uint8_t exact,status,contract_succeeded,round_lead,reserved[3];
} baloot_rollout_result_v3;
typedef struct baloot_rollout_task_v3 {
 baloot_state_v3 state;
 uint8_t initial_card,reserved[3]; /* card id 0..31, or BALOOT_NATIVE_ROLLOUT_NO_INITIAL_CARD */
} baloot_rollout_task_v3;

BALOOT_API uint32_t baloot_native_abi_version(void);
BALOOT_API uint32_t baloot_native_abi_version_v2(void);
BALOOT_API uint32_t baloot_native_capabilities_v2(void);
BALOOT_API uint32_t baloot_sizeof_rules_v2(void);
BALOOT_API uint32_t baloot_sizeof_state_v2(void);
BALOOT_API uint32_t baloot_sizeof_play_v2(void);
BALOOT_API uint32_t baloot_sizeof_undo_v2(void);
BALOOT_API uint32_t baloot_legal_moves_v1(const baloot_rules_v1*,const baloot_state_v1*);
BALOOT_API int32_t baloot_trick_winner_v1(const baloot_rules_v1*,uint8_t,int8_t,const baloot_play_v1[4]);
BALOOT_API uint32_t baloot_legal_moves_v2(const baloot_rules_v2*,const baloot_state_v2*);
BALOOT_API int32_t baloot_legal_moves_batch_v2(const baloot_rules_v2*,const baloot_state_v2*,uint32_t,uint32_t*);
BALOOT_API int32_t baloot_card_points_v2(const baloot_rules_v2*,uint8_t,uint8_t,int8_t);
BALOOT_API int32_t baloot_evaluate_trick_v2(const baloot_rules_v2*,uint8_t,int8_t,const baloot_play_v2[4],baloot_trick_result_v2*);
BALOOT_API int32_t baloot_partial_trick_winner_v2(const baloot_rules_v2*,uint8_t,int8_t,const baloot_play_v2*,uint8_t);
BALOOT_API uint64_t baloot_state_hash_v2(const baloot_state_v2*);
BALOOT_API int32_t baloot_apply_play_v2(const baloot_rules_v2*,baloot_state_v2*,uint8_t,uint8_t,uint8_t);
BALOOT_API int32_t baloot_apply_play_with_undo_v2(const baloot_rules_v2*,baloot_state_v2*,uint8_t,uint8_t,uint8_t,baloot_undo_v2*);
BALOOT_API int32_t baloot_undo_play_v2(baloot_state_v2*,const baloot_undo_v2*);
BALOOT_API uint32_t baloot_native_abi_version_v3(void);
BALOOT_API uint32_t baloot_sizeof_rules_v3(void);
BALOOT_API uint32_t baloot_sizeof_state_v3(void);
BALOOT_API uint32_t baloot_sizeof_search_limits_v3(void);
BALOOT_API uint32_t baloot_sizeof_search_result_v3(void);
BALOOT_API uint32_t baloot_sizeof_rollout_result_v3(void);
BALOOT_API uint32_t baloot_sizeof_rollout_task_v3(void);
BALOOT_API int32_t baloot_exact_search_v3(
 const baloot_rules_v3*,const baloot_state_v3*,const baloot_search_limits_v3*,
 const volatile uint32_t*,baloot_search_result_v3*);
BALOOT_API int32_t baloot_heuristic_rollout_batch_v3(
 const baloot_rules_v3*,const baloot_rollout_task_v3*,uint32_t,uint8_t,
 const volatile uint32_t*,baloot_search_result_v3*);
BALOOT_API int32_t baloot_heuristic_rollout_batch_compact_v3(
 const baloot_rules_v3*,const baloot_rollout_task_v3*,uint32_t,uint8_t,
 const volatile uint32_t*,baloot_rollout_result_v3*);

/* DD V1 consumes terminal payoffs compiled by the shared Python scoring authority.
 * Layout: [first-trick winner 0..3][sweep: none, NS, WE][terminal NS raw].
 * No legacy scoring interpretation or approximate result crosses this ABI. */
typedef struct baloot_dd_result_v1 {
 uint64_t nodes;
 int32_t values[64];
 uint8_t status,best_action,action_count,reserved;
 uint8_t actions[64];
} baloot_dd_result_v1;
BALOOT_API uint32_t baloot_sizeof_dd_result_v1(void);
BALOOT_API uint32_t baloot_double_dummy_abi_version(void);
/* Single-world DD, including the cached call, accepts zero milliseconds for a node-only budget. */
BALOOT_API int32_t baloot_double_dummy_v1(
 const baloot_rules_v2*,const baloot_state_v2*,const int32_t*,uint32_t,
 uint16_t,uint8_t,uint8_t,uint8_t,const baloot_search_limits_v3*,
 const volatile uint32_t*,baloot_dd_result_v1*);
/* One thread at a time per cache; the capacity is shared across all worlds. */
BALOOT_API void* baloot_dd_cache_create_v1(uint32_t);
BALOOT_API void baloot_dd_cache_free_v1(void*);
BALOOT_API int32_t baloot_dd_cache_stats_v1(void*,uint64_t[4]);
BALOOT_API int32_t baloot_double_dummy_cached_v1(
 void*,const baloot_rules_v2*,const baloot_state_v2*,const int32_t*,uint32_t,
 uint16_t,uint8_t,uint8_t,uint8_t,const baloot_search_limits_v3*,
 const volatile uint32_t*,baloot_dd_result_v1*);
/* One cohort call owns its cache. Worker threads share bounds safely; results
 * retain input order and each world proves every legal root alternative. */
typedef struct baloot_dd_task_v1 {
 baloot_state_v2 state;
 const int32_t* payoffs;
 uint32_t payoff_count;
 uint16_t raw_total;
 uint8_t first_winner,won_mask;
} baloot_dd_task_v1;
BALOOT_API uint32_t baloot_sizeof_dd_task_v1(void);
BALOOT_API int32_t baloot_double_dummy_batch_v1(
 void*,const baloot_rules_v2*,const baloot_dd_task_v1*,uint32_t,uint8_t,
 const baloot_search_limits_v3*,const volatile uint32_t*,baloot_dd_result_v1*);
#ifdef __cplusplus
}
#endif
#endif
