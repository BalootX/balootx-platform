#define BALOOT_NATIVE_BUILD
#include "baloot/native.h"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>
namespace {
constexpr uint8_t suit_of(uint8_t c) noexcept{return c/8u;} constexpr uint8_t rank_of(uint8_t c) noexcept{return c%8u;}
constexpr uint8_t team_of(uint8_t s) noexcept{return (s==0u||s==2u)?0u:1u;}
int strength1(const baloot_rules_v1*r,uint8_t c,uint8_t contract,int8_t trump) noexcept{auto k=rank_of(c);if(contract==1u&&trump>=0&&suit_of(c)==static_cast<uint8_t>(trump))return r->hokum_trump_strength[k];return contract==1u?r->hokum_non_trump_strength[k]:r->sun_strength[k];}
int strength2(const baloot_rules_v2*r,uint8_t c,uint8_t contract,int8_t trump) noexcept{auto k=rank_of(c);if(contract==1u&&trump>=0&&suit_of(c)==static_cast<uint8_t>(trump))return r->strengths[2][k];return contract==1u?r->strengths[1][k]:r->strengths[0][k];}
uint32_t suit_mask(uint32_t hand,uint8_t suit) noexcept{return hand&(0xffu<<(static_cast<uint32_t>(suit)*8u));}
template<class P,class Strength> int winner_impl(const P*plays,uint8_t count,uint8_t contract,int8_t trump,Strength strength) noexcept{if(count==0)return -1;auto led=suit_of(plays[0].card);bool any=false;if(contract==1u&&trump>=0)for(uint8_t i=0;i<count;++i)if(suit_of(plays[i].card)==static_cast<uint8_t>(trump))any=true;int best=-1,bv=-1;for(uint8_t i=0;i<count;++i){auto suit=suit_of(plays[i].card);if(any?suit!=static_cast<uint8_t>(trump):suit!=led)continue;int v=strength(plays[i].card);if(v>bv){bv=v;best=plays[i].seat;}}return best;}
template<class P> uint8_t winning_card(const P*plays,uint8_t count,int winner) noexcept{for(uint8_t i=0;i<count;++i)if(plays[i].seat==winner)return plays[i].card;return plays[0].card;}
template<class Strength> uint32_t higher_mask(uint32_t candidates,uint8_t target_card,Strength strength) noexcept{int target=strength(target_card);uint32_t out=0;while(candidates){uint8_t c=static_cast<uint8_t>(std::countr_zero(candidates));candidates&=candidates-1u;if(strength(c)>target)out|=1u<<c;}return out;}
bool valid_ekka2(const baloot_rules_v2*r,const baloot_state_v2*s,uint8_t card) noexcept{
 if(!r->ekka_enabled||s->trick_size!=0u||s->contract!=1u||s->trump_suit<0||suit_of(card)==static_cast<uint8_t>(s->trump_suit))return false;
 uint32_t remaining=0u;for(auto hand:s->hands)remaining|=hand;
 auto suited=suit_mask(remaining,suit_of(card));
 return higher_mask(suited,card,[&](uint8_t value){return r->strengths[0][rank_of(value)];})==0u;
}
uint64_t fnv(uint64_t h,uint64_t v) noexcept{for(int i=0;i<8;++i){h^=static_cast<uint8_t>(v>>(i*8));h*=1099511628211ull;}return h;}
}
extern "C" uint32_t baloot_native_abi_version(void){return BALOOT_NATIVE_ABI_V1;}
extern "C" uint32_t baloot_native_abi_version_v2(void){return BALOOT_NATIVE_ABI_V2;}
extern "C" uint32_t baloot_native_capabilities_v2(void){return BALOOT_NATIVE_CAP_LEGAL_MOVES|BALOOT_NATIVE_CAP_BATCH_LEGAL_MOVES|BALOOT_NATIVE_CAP_CARD_POINTS|BALOOT_NATIVE_CAP_TRICK_EVALUATION|BALOOT_NATIVE_CAP_STATE_HASH|BALOOT_NATIVE_CAP_APPLY_PLAY|BALOOT_NATIVE_CAP_APPLY_UNDO|BALOOT_NATIVE_CAP_EXACT_SEARCH;}
extern "C" uint32_t baloot_sizeof_rules_v2(void){return static_cast<uint32_t>(sizeof(baloot_rules_v2));}
extern "C" uint32_t baloot_sizeof_state_v2(void){return static_cast<uint32_t>(sizeof(baloot_state_v2));}
extern "C" uint32_t baloot_sizeof_play_v2(void){return static_cast<uint32_t>(sizeof(baloot_play_v2));}
extern "C" uint32_t baloot_sizeof_undo_v2(void){return static_cast<uint32_t>(sizeof(baloot_undo_v2));}
extern "C" uint32_t baloot_legal_moves_v1(const baloot_rules_v1*r,const baloot_state_v1*s){if(!r||!s||r->abi_version!=1u||s->abi_version!=1u||s->current_player>3u||s->trick_size>3u)return 0;auto seat=s->current_player;auto hand=s->hands[seat];if(!hand)return 0;auto str=[&](uint8_t c){return strength1(r,c,s->contract,s->trump_suit);};if(!s->trick_size){if(s->contract==1u&&s->locked&&s->trump_suit>=0&&r->locked_hokum_blocks_trump_lead){auto n=hand&~suit_mask(hand,static_cast<uint8_t>(s->trump_suit));if(n)return n;}return hand;}const uint8_t led=suit_of(s->trick[0].card);const uint32_t followers=suit_mask(hand,led);int w=winner_impl(s->trick,s->trick_size,s->contract,s->trump_suit,str);if(followers){if(s->contract==1u&&s->trump_suit>=0&&led==static_cast<uint8_t>(s->trump_suit)&&w>=0&&team_of(static_cast<uint8_t>(w))!=team_of(seat)){auto hi=higher_mask(followers,winning_card(s->trick,s->trick_size,w),str);if(hi)return hi;}return followers;}if(s->contract!=1u||s->trump_suit<0)return hand;auto tr=suit_mask(hand,static_cast<uint8_t>(s->trump_suit));if(!tr)return hand;if(w<0)return tr;auto wc=winning_card(s->trick,s->trick_size,w);if(team_of(static_cast<uint8_t>(w))==team_of(seat)){bool third=s->trick_size==2u,partner=s->trick[0].seat==((seat+2u)%4u),ace=rank_of(s->trick[0].card)==7u;if(r->partner_winning_exception&&third&&partner&&!(s->trick[0].ekka||ace))return tr;return hand;}if(suit_of(wc)==static_cast<uint8_t>(s->trump_suit)){auto hi=higher_mask(tr,wc,str);if(hi)return hi;return r->allow_discard_when_cannot_overtrump?hand:tr;}return tr;}
extern "C" int32_t baloot_trick_winner_v1(const baloot_rules_v1*r,uint8_t c,int8_t t,const baloot_play_v1 p[4]){if(!r||!p||r->abi_version!=1u)return -1;return winner_impl(p,4,c,t,[&](uint8_t x){return strength1(r,x,c,t);});}
extern "C" uint32_t baloot_legal_moves_v2(const baloot_rules_v2*r,const baloot_state_v2*s){if(!r||!s||r->abi_version!=2u||s->abi_version!=2u||s->current_player>3u||s->trick_size>3u)return 0;const uint8_t seat=s->current_player;const uint32_t hand=s->hands[seat];if(!hand)return 0;auto str=[&](uint8_t c){return strength2(r,c,s->contract,s->trump_suit);};if(!s->trick_size){if(s->contract==1u&&s->locked&&s->trump_suit>=0&&r->locked_hokum_blocks_trump_lead){auto n=hand&~suit_mask(hand,static_cast<uint8_t>(s->trump_suit));if(n)return n;}return hand;}auto led=suit_of(s->trick[0].card);auto follow=suit_mask(hand,led);int w=winner_impl(s->trick,s->trick_size,s->contract,s->trump_suit,str);if(follow&&r->must_follow_suit){if(s->contract==1u&&s->trump_suit>=0&&led==static_cast<uint8_t>(s->trump_suit)&&r->must_overtrump_when_possible&&w>=0&&team_of(static_cast<uint8_t>(w))!=team_of(seat)){auto hi=higher_mask(follow,winning_card(s->trick,s->trick_size,w),str);if(hi)return hi;}return follow;}if(s->contract!=1u||s->trump_suit<0||!r->must_trump_when_void)return hand;auto tr=suit_mask(hand,static_cast<uint8_t>(s->trump_suit));if(!tr)return hand;if(w<0)return tr;auto wc=winning_card(s->trick,s->trick_size,w);if(team_of(static_cast<uint8_t>(w))==team_of(seat)){bool third=s->trick_size==2u,partner=s->trick[0].seat==((seat+2u)%4u),ace=rank_of(s->trick[0].card)==7u;if(r->partner_winning_exception&&third&&partner&&!(s->trick[0].ekka||ace))return tr;return hand;}if(suit_of(wc)==static_cast<uint8_t>(s->trump_suit)&&r->must_overtrump_when_possible){auto hi=higher_mask(tr,wc,str);if(hi)return hi;return r->allow_discard_when_cannot_overtrump?hand:tr;}return tr;}
extern "C" int32_t baloot_legal_moves_batch_v2(const baloot_rules_v2*r,const baloot_state_v2*s,uint32_t n,uint32_t*out){if(!r||!s||!out)return -1;for(uint32_t i=0;i<n;++i)out[i]=baloot_legal_moves_v2(r,&s[i]);return static_cast<int32_t>(n);}
extern "C" int32_t baloot_card_points_v2(const baloot_rules_v2*r,uint8_t card,uint8_t contract,int8_t trump){if(!r||r->abi_version!=2u||card>31u)return -1;auto rank=rank_of(card);if(contract==1u&&trump>=0&&suit_of(card)==static_cast<uint8_t>(trump))return r->points[2][rank];return contract==1u?r->points[1][rank]:r->points[0][rank];}
extern "C" int32_t baloot_partial_trick_winner_v2(const baloot_rules_v2*r,uint8_t contract,int8_t trump,const baloot_play_v2*p,uint8_t count){if(!r||!p||r->abi_version!=2u||count<1||count>4)return -1;for(uint8_t i=0;i<count;++i)if(p[i].card>31||p[i].seat>3)return -1;return winner_impl(p,count,contract,trump,[&](uint8_t x){return strength2(r,x,contract,trump);});}
extern "C" int32_t baloot_evaluate_trick_v2(const baloot_rules_v2*r,uint8_t contract,int8_t trump,const baloot_play_v2 p[4],baloot_trick_result_v2*out){if(!r||!p||!out||r->abi_version!=2u)return -1;int w=winner_impl(p,4,contract,trump,[&](uint8_t x){return strength2(r,x,contract,trump);});int pts=0;for(int i=0;i<4;++i){int v=baloot_card_points_v2(r,p[i].card,contract,trump);if(v<0)return -2;pts+=v;}out->winner=w;out->raw_points=pts;return 0;}
extern "C" uint64_t baloot_state_hash_v2(const baloot_state_v2*s){if(!s||s->abi_version!=2u)return 0;uint64_t h=1469598103934665603ull;for(auto v:s->hands)h=fnv(h,v);for(auto v:s->initial_hands)h=fnv(h,v);h=fnv(h,s->contract);h=fnv(h,static_cast<uint8_t>(s->trump_suit));h=fnv(h,s->current_player);h=fnv(h,s->locked);h=fnv(h,s->trump_broken);h=fnv(h,s->buyer);h=fnv(h,s->dealer);h=fnv(h,s->first_leader);h=fnv(h,s->trick_size);h=fnv(h,s->completed_tricks);h=fnv(h,s->escalation_multiplier);for(auto v:s->raw_points)h=fnv(h,static_cast<uint32_t>(v));for(auto v:s->project_points)h=fnv(h,static_cast<uint32_t>(v));for(auto v:s->baloot_points)h=fnv(h,static_cast<uint32_t>(v));for(auto v:s->match_score)h=fnv(h,static_cast<uint32_t>(v));for(uint8_t i=0;i<s->trick_size&&i<3;++i){h=fnv(h,s->trick[i].seat);h=fnv(h,s->trick[i].card);h=fnv(h,s->trick[i].ekka);h=fnv(h,s->trick[i].baloot);}return h;}
extern "C" int32_t baloot_apply_play_v2(const baloot_rules_v2*r,baloot_state_v2*s,uint8_t card,uint8_t ekka,uint8_t baloot){if(!r||!s||card>31u)return -1;auto legal=baloot_legal_moves_v2(r,s);if(!(legal&(1u<<card)))return -2;if(ekka&&!valid_ekka2(r,s,card))return -4;auto seat=s->current_player;s->hands[seat]&=~(1u<<card);if(s->trick_size<3u){s->trick[s->trick_size]={seat,card,ekka,baloot};++s->trick_size;s->current_player=(seat+1u)%4u;return 0;}baloot_play_v2 all[4];for(int i=0;i<3;++i)all[i]=s->trick[i];all[3]={seat,card,ekka,baloot};baloot_trick_result_v2 result{};if(baloot_evaluate_trick_v2(r,s->contract,s->trump_suit,all,&result)!=0)return -3;int team=team_of(static_cast<uint8_t>(result.winner));s->raw_points[team]+=result.raw_points;if(s->completed_tricks==7u)s->raw_points[team]+=r->last_trick_bonus_raw;++s->completed_tricks;s->trick_size=0;s->current_player=static_cast<uint8_t>(result.winner);for(auto &p:s->trick)p={0,0,0,0};return 1;}

extern "C" int32_t baloot_apply_play_with_undo_v2(const baloot_rules_v2*r,baloot_state_v2*s,uint8_t card,uint8_t ekka,uint8_t baloot,baloot_undo_v2*u){if(!s||!u)return -10;u->previous=*s;auto result=baloot_apply_play_v2(r,s,card,ekka,baloot);if(result<0)*s=u->previous;return result;}
extern "C" int32_t baloot_undo_play_v2(baloot_state_v2*s,const baloot_undo_v2*u){if(!s||!u)return -1;*s=u->previous;return 0;}

namespace {

constexpr uint8_t no_seat_v3 = 255u;

struct search_position_v3 {
    std::array<uint32_t, 4> hands{};
    std::array<int32_t, 2> raw_points{};
    std::array<int32_t, 2> baloot_points{};
    std::array<baloot_play_v2, 3> trick{};
    std::array<uint8_t, 2> tricks_won{};
    uint8_t current_player{};
    uint8_t trick_size{};
    uint8_t completed_tricks{};
    uint8_t trump_broken{};
    uint8_t baloot_declared_teams_mask{};

    bool operator==(const search_position_v3& other) const noexcept {
        if (hands != other.hands || raw_points != other.raw_points ||
            baloot_points != other.baloot_points || tricks_won != other.tricks_won ||
            current_player != other.current_player || trick_size != other.trick_size ||
            completed_tricks != other.completed_tricks || trump_broken != other.trump_broken ||
            baloot_declared_teams_mask != other.baloot_declared_teams_mask) {
            return false;
        }
        for (size_t index = 0; index < trick.size(); ++index) {
            if (trick[index].seat != other.trick[index].seat ||
                trick[index].card != other.trick[index].card ||
                trick[index].ekka != other.trick[index].ekka ||
                trick[index].baloot != other.trick[index].baloot) {
                return false;
            }
        }
        return true;
    }
};

struct search_cache_key_v3 {
    search_position_v3 position{};
    uint8_t depth{};
    bool operator==(const search_cache_key_v3& other) const noexcept {
        return depth == other.depth && position == other.position;
    }
};

struct search_position_hash_v3 {
    size_t operator()(const search_position_v3& value) const noexcept {
        uint64_t hash = 1469598103934665603ull;
        for (const auto item : value.hands) hash = fnv(hash, item);
        for (const auto item : value.raw_points) hash = fnv(hash, static_cast<uint32_t>(item));
        for (const auto item : value.baloot_points) hash = fnv(hash, static_cast<uint32_t>(item));
        for (const auto item : value.tricks_won) hash = fnv(hash, item);
        hash = fnv(hash, value.current_player);
        hash = fnv(hash, value.trick_size);
        hash = fnv(hash, value.completed_tricks);
        hash = fnv(hash, value.trump_broken);
        hash = fnv(hash, value.baloot_declared_teams_mask);
        for (size_t index = 0; index < value.trick_size && index < value.trick.size(); ++index) {
            hash = fnv(hash, value.trick[index].seat);
            hash = fnv(hash, value.trick[index].card);
            hash = fnv(hash, value.trick[index].ekka);
            hash = fnv(hash, value.trick[index].baloot);
        }
        return static_cast<size_t>(hash ^ (hash >> 32u));
    }
};

struct search_cache_hash_v3 {
    size_t operator()(const search_cache_key_v3& value) const noexcept {
        const auto base = search_position_hash_v3{}(value.position);
        return base ^ (static_cast<size_t>(value.depth) * static_cast<size_t>(0x9e3779b9u));
    }
};

struct node_result_v3 {
    int32_t value{};
    int32_t own_points{};
    int32_t opponent_points{};
    std::array<int32_t, 2> score_after{};
    int8_t match_winner{-1};
    bool contract_succeeded{};
    bool round_lead{};
    std::array<uint8_t, 32> line{};
    uint8_t line_length{};
};

struct node_budget_v3 final : std::exception {};
struct time_budget_v3 final : std::exception {};
struct cancelled_v3 final : std::exception {};

search_position_v3 position_of_v3(const baloot_state_v3& state) noexcept {
    search_position_v3 result{};
    for (size_t index = 0; index < 4; ++index) result.hands[index] = state.base.hands[index];
    for (size_t index = 0; index < 2; ++index) {
        result.raw_points[index] = state.base.raw_points[index];
        result.baloot_points[index] = state.base.baloot_points[index];
        result.tricks_won[index] = state.tricks_won[index];
    }
    for (size_t index = 0; index < 3; ++index) result.trick[index] = state.base.trick[index];
    result.current_player = state.base.current_player;
    result.trick_size = state.base.trick_size;
    result.completed_tricks = state.base.completed_tricks;
    result.trump_broken = state.base.trump_broken;
    result.baloot_declared_teams_mask = state.baloot_declared_teams_mask;
    return result;
}

int32_t sun_game_points_v3(int32_t raw) noexcept {
    if (raw % 10 == 5) return raw / 5;
    return ((raw + 5) / 10) * 2;
}

int32_t hokum_game_points_v3(int32_t raw) noexcept {
    const auto tens = raw / 10;
    const auto units = raw % 10;
    return tens + (units >= 6 ? 1 : 0);
}

uint8_t round_winner_v3(
    const baloot_state_v3& state,
    const std::array<int32_t, 2>& totals) noexcept {
    if (totals[0] != totals[1]) return totals[0] > totals[1] ? 0u : 1u;
    const auto counting = state.counting_team;
    const auto units = state.base.raw_points[counting] % 10;
    if (state.base.contract == 1u) {
        return (units >= 2 && units <= 5) ? counting : team_of(state.declarer);
    }
    return (units >= 1 && units <= 4) ? counting : team_of(state.declarer);
}

node_result_v3 terminal_v3(
    const baloot_rules_v3& rules,
    const baloot_state_v3& state,
    uint8_t observer_team) noexcept {
    const auto counting = state.counting_team;
    const auto total_cards = static_cast<int32_t>(
        rules.card_game_point_totals[state.base.contract == 1u ? 1u : 0u]);
    const auto counting_points = state.base.contract == 1u
        ? hokum_game_points_v3(state.base.raw_points[counting])
        : sun_game_points_v3(state.base.raw_points[counting]);
    std::array<int32_t, 2> card_points{};
    card_points[counting] = counting_points;
    card_points[1u - counting] = total_cards - counting_points;
    std::array<int32_t, 2> totals{};
    for (size_t index = 0; index < 2; ++index) {
        totals[index] = card_points[index] + state.base.project_points[index] +
            state.base.baloot_points[index];
    }
    const auto winner = round_winner_v3(state, totals);
    const auto declarer_team = team_of(state.declarer);
    const bool kabout = rules.kabout_enabled != 0u &&
        (state.tricks_won[0] == 0u || state.tricks_won[1] == 0u);
    const bool gahwa = state.escalation_mode == 4u;
    std::array<int32_t, 2> awarded{};
    if (!gahwa && kabout) {
        const auto base = static_cast<int32_t>(state.base.contract == 1u
            ? rules.base.kabout_hokum : rules.base.kabout_sun);
        awarded[winner] = base * static_cast<int32_t>(state.base.escalation_multiplier)
            + state.base.project_points[winner] * static_cast<int32_t>(
                std::min<uint8_t>(state.base.escalation_multiplier, rules.project_multiplier_cap))
            + state.base.baloot_points[winner] * static_cast<int32_t>(
                std::min<uint8_t>(state.base.escalation_multiplier, rules.baloot_multiplier_cap));
    } else if (!gahwa && state.escalation_mode == 0u) {
        if (winner == declarer_team) {
            awarded = totals;
        } else {
            awarded[winner] = total_cards + state.base.project_points[0] +
                state.base.project_points[1] + state.base.baloot_points[0] +
                state.base.baloot_points[1];
        }
    } else if (!gahwa) {
        awarded[winner] = total_cards * static_cast<int32_t>(state.base.escalation_multiplier)
            + (state.base.project_points[0] + state.base.project_points[1]) *
                static_cast<int32_t>(std::min<uint8_t>(
                    state.base.escalation_multiplier, rules.project_multiplier_cap))
            + (state.base.baloot_points[0] + state.base.baloot_points[1]) *
                static_cast<int32_t>(std::min<uint8_t>(
                    state.base.escalation_multiplier, rules.baloot_multiplier_cap));
    }
    std::array<int32_t, 2> score_after{
        state.base.match_score[0], state.base.match_score[1]};
    int8_t match_winner = -1;
    if (gahwa) {
        score_after[winner] = std::max<int32_t>(
            static_cast<int32_t>(rules.base.match_target), score_after[1u - winner] + 1);
        match_winner = static_cast<int8_t>(winner);
    } else {
        score_after[0] += awarded[0];
        score_after[1] += awarded[1];
        if (std::max(score_after[0], score_after[1]) >=
                static_cast<int32_t>(rules.base.match_target) &&
            score_after[0] != score_after[1]) {
            match_winner = static_cast<int8_t>(score_after[0] > score_after[1] ? 0 : 1);
        }
    }
    node_result_v3 result{};
    result.own_points = awarded[observer_team];
    result.opponent_points = awarded[1u - observer_team];
    result.value = result.own_points - result.opponent_points;
    result.score_after = score_after;
    result.match_winner = match_winner;
    const bool declarer_succeeded = winner == declarer_team;
    result.contract_succeeded = observer_team == declarer_team
        ? declarer_succeeded : !declarer_succeeded;
    result.round_lead = result.value > 0;
    return result;
}

int32_t apply_search_play_v3(
    const baloot_rules_v3& rules,
    baloot_state_v3& state,
    uint8_t card) noexcept {
    if (state.base.current_player > 3u || card > 31u) return -1;
    const auto legal = baloot_legal_moves_v2(&rules.base, &state.base);
    if ((legal & (1u << card)) == 0u) return -2;
    const auto seat = state.base.current_player;
    const auto led_suit = state.base.trick_size > 0u
        ? suit_of(state.base.trick[0].card) : suit_of(card);
    state.base.hands[seat] &= ~(1u << card);
    if (state.base.contract == 1u && state.base.trump_suit >= 0 &&
        suit_of(card) == static_cast<uint8_t>(state.base.trump_suit) &&
        (rank_of(card) == 5u || rank_of(card) == 6u) &&
        (state.baloot_eligible_seats_mask & (1u << seat)) != 0u) {
        const auto other_rank = rank_of(card) == 5u ? 6u : 5u;
        const auto other = static_cast<uint8_t>(
            static_cast<uint8_t>(state.base.trump_suit) * 8u + other_rank);
        const auto team = team_of(seat);
        if ((state.base.initial_hands[seat] & (1u << other)) != 0u &&
            (state.base.hands[seat] & (1u << other)) == 0u &&
            (state.baloot_declared_teams_mask & (1u << team)) == 0u) {
            state.base.baloot_points[team] += rules.base.baloot_game_points;
            state.baloot_declared_teams_mask |= static_cast<uint8_t>(1u << team);
        }
    }
    if (state.base.trump_suit >= 0 &&
        suit_of(card) == static_cast<uint8_t>(state.base.trump_suit) &&
        led_suit != static_cast<uint8_t>(state.base.trump_suit)) {
        state.base.trump_broken = 1u;
    }
    if (state.base.trick_size < 3u) {
        state.base.trick[state.base.trick_size] = {seat, card, 0u, 0u};
        ++state.base.trick_size;
        state.base.current_player = static_cast<uint8_t>((seat + 1u) % 4u);
        return 0;
    }
    baloot_play_v2 plays[4]{};
    for (size_t index = 0; index < 3; ++index) plays[index] = state.base.trick[index];
    plays[3] = {seat, card, 0u, 0u};
    baloot_trick_result_v2 trick_result{};
    if (baloot_evaluate_trick_v2(
            &rules.base, state.base.contract, state.base.trump_suit,
            plays, &trick_result) != 0 || trick_result.winner < 0) {
        return -3;
    }
    const auto winner = static_cast<uint8_t>(trick_result.winner);
    const auto team = team_of(winner);
    state.base.raw_points[team] += trick_result.raw_points;
    state.tricks_won[team] = static_cast<uint8_t>(state.tricks_won[team] + 1u);
    if (state.base.completed_tricks == 7u) {
        state.base.raw_points[team] += rules.base.last_trick_bonus_raw;
    }
    ++state.base.completed_tricks;
    state.base.trick_size = 0u;
    for (auto& play : state.base.trick) play = {0u, 0u, 0u, 0u};
    state.base.current_player = state.base.completed_tricks == 8u ? no_seat_v3 : winner;
    return state.base.completed_tricks == 8u ? 2 : 1;
}

uint8_t heuristic_card_v3(
    const baloot_rules_v3& rules,
    const baloot_state_v3& state) {
    auto legal = baloot_legal_moves_v2(&rules.base, &state.base);
    if (legal == 0u) throw std::runtime_error("native rollout found no legal move");
    std::vector<uint8_t> candidates;
    auto scan = legal;
    if (state.base.trick_size == 3u) {
        while (scan != 0u) {
            const auto card = static_cast<uint8_t>(std::countr_zero(scan));
            scan &= scan - 1u;
            baloot_play_v2 plays[4]{};
            for (size_t index = 0; index < 3; ++index) plays[index] = state.base.trick[index];
            plays[3] = {state.base.current_player, card, 0u, 0u};
            baloot_trick_result_v2 result{};
            if (baloot_evaluate_trick_v2(
                    &rules.base, state.base.contract, state.base.trump_suit,
                    plays, &result) == 0 &&
                result.winner == state.base.current_player) {
                candidates.push_back(card);
            }
        }
    }
    if (candidates.empty()) {
        while (legal != 0u) {
            const auto card = static_cast<uint8_t>(std::countr_zero(legal));
            legal &= legal - 1u;
            candidates.push_back(card);
        }
    }
    const auto cost = [&rules, &state](uint8_t card) {
        return std::array<int32_t, 3>{
            baloot_card_points_v2(
                &rules.base, card, state.base.contract, state.base.trump_suit),
            strength2(
                &rules.base, card, state.base.contract, state.base.trump_suit),
            static_cast<int32_t>(card),
        };
    };
    return *std::min_element(
        candidates.begin(), candidates.end(),
        [&cost](uint8_t left, uint8_t right) { return cost(left) < cost(right); });
}

node_result_v3 heuristic_rollout_v3(
    const baloot_rules_v3& rules,
    baloot_state_v3 state,
    uint8_t team,
    const volatile uint32_t* cancelled) {
    std::array<uint8_t, 32> line{};
    uint8_t length = 0u;
    while (state.base.current_player <= 3u) {
        if (cancelled != nullptr && *cancelled != 0u) throw cancelled_v3{};
        const auto card = heuristic_card_v3(rules, state);
        if (length < line.size()) line[length++] = card;
        if (apply_search_play_v3(rules, state, card) < 0) {
            throw std::runtime_error("native rollout transition failed");
        }
    }
    auto result = terminal_v3(rules, state, team);
    result.line = line;
    result.line_length = length;
    return result;
}

class exact_search_context_v3 {
public:
    exact_search_context_v3(
        const baloot_rules_v3& rules,
        const baloot_search_limits_v3& limits,
        const volatile uint32_t* cancelled)
        : rules_(rules), limits_(limits), cancelled_(cancelled),
          started_(std::chrono::steady_clock::now()) {
        cache_.reserve(std::min<uint32_t>(limits.maximum_cache_entries, 1000000u));
        preferred_.reserve(std::min<uint32_t>(limits.maximum_cache_entries, 1000000u));
    }

    uint64_t nodes() const noexcept { return nodes_; }

    node_result_v3 solve(baloot_state_v3 root, uint8_t depth) {
        return visit(root, depth, std::numeric_limits<int32_t>::min() / 2,
            std::numeric_limits<int32_t>::max() / 2);
    }

    node_result_v3 rollout(baloot_state_v3 state) const {
        return heuristic_rollout_v3(rules_, state, limits_.team, cancelled_);
    }

private:
    void check_limits() {
        if (cancelled_ != nullptr && *cancelled_ != 0u) throw cancelled_v3{};
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - started_).count() >=
            static_cast<int64_t>(limits_.maximum_milliseconds)) {
            throw time_budget_v3{};
        }
        ++nodes_;
        if (nodes_ > limits_.maximum_nodes) throw node_budget_v3{};
    }

    node_result_v3 visit(
        const baloot_state_v3& state,
        uint8_t depth,
        int32_t alpha,
        int32_t beta) {
        check_limits();
        if (state.base.current_player > 3u || state.base.completed_tricks == 8u) {
            return terminal_v3(rules_, state, limits_.team);
        }
        if (depth == 0u) return rollout(state);
        const auto position = position_of_v3(state);
        const search_cache_key_v3 cache_key{position, depth};
        if (const auto found = cache_.find(cache_key); found != cache_.end()) {
            return found->second;
        }
        auto legal_mask = baloot_legal_moves_v2(&rules_.base, &state.base);
        if (legal_mask == 0u) throw std::runtime_error("native search found no legal move");
        std::vector<uint8_t> cards;
        while (legal_mask != 0u) {
            const auto card = static_cast<uint8_t>(std::countr_zero(legal_mask));
            legal_mask &= legal_mask - 1u;
            cards.push_back(card);
        }
        if (const auto found = preferred_.find(position); found != preferred_.end()) {
            const auto preferred = found->second;
            const auto item = std::find(cards.begin(), cards.end(), preferred);
            if (item != cards.end()) std::rotate(cards.begin(), item, item + 1);
        }
        const bool maximizing = team_of(state.base.current_player) == limits_.team;
        node_result_v3 best{};
        bool have_best = false;
        bool cutoff = false;
        uint8_t best_card = cards.front();
        for (const auto card : cards) {
            auto child = state;
            if (apply_search_play_v3(rules_, child, card) < 0) {
                throw std::runtime_error("native search transition failed");
            }
            auto candidate = visit(child, static_cast<uint8_t>(depth - 1u), alpha, beta);
            if (!have_best || (maximizing
                    ? std::pair<int32_t, int32_t>{candidate.value, -static_cast<int32_t>(card)} >
                        std::pair<int32_t, int32_t>{best.value, -static_cast<int32_t>(best_card)}
                    : std::pair<int32_t, int32_t>{candidate.value, static_cast<int32_t>(card)} <
                        std::pair<int32_t, int32_t>{best.value, static_cast<int32_t>(best_card)})) {
                best = candidate;
                best_card = card;
                have_best = true;
                const auto child_length = std::min<size_t>(candidate.line_length, best.line.size() - 1u);
                std::move_backward(best.line.begin(), best.line.begin() + child_length,
                    best.line.begin() + child_length + 1u);
                best.line[0] = card;
                best.line_length = static_cast<uint8_t>(child_length + 1u);
            }
            if (maximizing) alpha = std::max(alpha, candidate.value);
            else beta = std::min(beta, candidate.value);
            if (beta <= alpha) {
                cutoff = true;
                break;
            }
        }
        if (!have_best) throw std::runtime_error("native search produced no result");
        preferred_[position] = best_card;
        if (!cutoff && limits_.maximum_cache_entries > 0u &&
            cache_.size() < limits_.maximum_cache_entries) {
            cache_.emplace(cache_key, best);
        }
        return best;
    }

    const baloot_rules_v3& rules_;
    const baloot_search_limits_v3& limits_;
    const volatile uint32_t* cancelled_;
    std::chrono::steady_clock::time_point started_;
    uint64_t nodes_{0u};
    std::unordered_map<search_cache_key_v3, node_result_v3, search_cache_hash_v3> cache_;
    std::unordered_map<search_position_v3, uint8_t, search_position_hash_v3> preferred_;
};

void export_result_v3(
    const node_result_v3& source,
    uint64_t nodes,
    uint8_t status,
    baloot_search_result_v3& target) noexcept {
    target = {};
    target.nodes = nodes;
    target.value = source.value;
    target.own_points = source.own_points;
    target.opponent_points = source.opponent_points;
    target.score_after_round[0] = source.score_after[0];
    target.score_after_round[1] = source.score_after[1];
    target.match_winner = source.match_winner;
    target.exact = status == BALOOT_NATIVE_SEARCH_EXACT ? 1u : 0u;
    target.status = status;
    target.principal_variation_length = source.line_length;
    target.contract_succeeded = source.contract_succeeded ? 1u : 0u;
    target.round_lead = source.round_lead ? 1u : 0u;
    for (size_t index = 0; index < source.line_length; ++index) {
        target.principal_variation[index] = source.line[index];
    }
}

void export_rollout_result_v3(
    const node_result_v3& source,
    uint64_t nodes,
    uint8_t status,
    baloot_rollout_result_v3& target) noexcept {
    target = {};
    target.nodes = nodes;
    target.value = source.value;
    target.own_points = source.own_points;
    target.opponent_points = source.opponent_points;
    target.score_after_round[0] = source.score_after[0];
    target.score_after_round[1] = source.score_after[1];
    target.match_winner = source.match_winner;
    target.exact = status == BALOOT_NATIVE_SEARCH_EXACT ? 1u : 0u;
    target.status = status;
    target.contract_succeeded = source.contract_succeeded ? 1u : 0u;
    target.round_lead = source.round_lead ? 1u : 0u;
}

template <class Result, class Export>
int32_t heuristic_rollout_batch_impl_v3(
    const baloot_rules_v3* rules,
    const baloot_rollout_task_v3* tasks,
    uint32_t count,
    uint8_t team,
    const volatile uint32_t* cancelled,
    Result* results,
    Export export_result) {
    if (rules == nullptr || tasks == nullptr || results == nullptr ||
        rules->base.abi_version != BALOOT_NATIVE_ABI_V2 || team > 1u) {
        return -1;
    }
    try {
        for (uint32_t index = 0; index < count; ++index) {
            if (cancelled != nullptr && *cancelled != 0u) return -2;
            auto state = tasks[index].state;
            const auto initial_card = tasks[index].initial_card;
            if (state.base.abi_version != BALOOT_NATIVE_ABI_V2 ||
                (initial_card != BALOOT_NATIVE_ROLLOUT_NO_INITIAL_CARD &&
                 (initial_card > 31u ||
                  apply_search_play_v3(*rules, state, initial_card) < 0))) {
                return -3;
            }
            const auto outcome = heuristic_rollout_v3(*rules, state, team, cancelled);
            export_result(
                outcome, outcome.line_length,
                BALOOT_NATIVE_SEARCH_HEURISTIC_COMPLETE, results[index]);
        }
    } catch (const cancelled_v3&) {
        return -2;
    } catch (...) {
        return -4;
    }
    return static_cast<int32_t>(count);
}

} // namespace

extern "C" uint32_t baloot_native_abi_version_v3(void) { return BALOOT_NATIVE_ABI_V3; }
extern "C" uint32_t baloot_sizeof_rules_v3(void) { return static_cast<uint32_t>(sizeof(baloot_rules_v3)); }
extern "C" uint32_t baloot_sizeof_state_v3(void) { return static_cast<uint32_t>(sizeof(baloot_state_v3)); }
extern "C" uint32_t baloot_sizeof_search_limits_v3(void) { return static_cast<uint32_t>(sizeof(baloot_search_limits_v3)); }
extern "C" uint32_t baloot_sizeof_search_result_v3(void) { return static_cast<uint32_t>(sizeof(baloot_search_result_v3)); }
extern "C" uint32_t baloot_sizeof_rollout_result_v3(void) { return static_cast<uint32_t>(sizeof(baloot_rollout_result_v3)); }
extern "C" uint32_t baloot_sizeof_rollout_task_v3(void) { return static_cast<uint32_t>(sizeof(baloot_rollout_task_v3)); }

extern "C" int32_t baloot_exact_search_v3(
    const baloot_rules_v3* rules,
    const baloot_state_v3* state,
    const baloot_search_limits_v3* limits,
    const volatile uint32_t* cancelled,
    baloot_search_result_v3* result) {
    if (rules == nullptr || state == nullptr || limits == nullptr || result == nullptr ||
        rules->base.abi_version != BALOOT_NATIVE_ABI_V2 ||
        state->base.abi_version != BALOOT_NATIVE_ABI_V2 ||
        limits->maximum_nodes == 0u || limits->maximum_milliseconds == 0u ||
        limits->team > 1u || state->declarer > 3u || state->counting_team > 1u ||
        state->escalation_mode > 4u) {
        return -1;
    }
    try {
        const auto remaining = static_cast<uint8_t>(
            std::popcount(state->base.hands[0]) + std::popcount(state->base.hands[1]) +
            std::popcount(state->base.hands[2]) + std::popcount(state->base.hands[3]));
        const auto depth_budget = limits->maximum_depth == 0u
            ? remaining : std::min<uint8_t>(remaining, limits->maximum_depth);
        exact_search_context_v3 context(*rules, *limits, cancelled);
        node_result_v3 best{};
        uint8_t status = BALOOT_NATIVE_SEARCH_BUDGET_EXHAUSTED;
        try {
            best = context.rollout(*state);
            if (remaining == 0u || state->base.current_player > 3u) {
                best = terminal_v3(*rules, *state, limits->team);
                status = BALOOT_NATIVE_SEARCH_EXACT;
            } else {
                for (uint8_t depth = 1u; depth <= depth_budget; ++depth) {
                    best = context.solve(*state, depth);
                    status = depth == remaining
                        ? BALOOT_NATIVE_SEARCH_EXACT
                        : BALOOT_NATIVE_SEARCH_BUDGET_EXHAUSTED;
                }
            }
        } catch (const cancelled_v3&) {
            status = BALOOT_NATIVE_SEARCH_CANCELLED;
        } catch (const node_budget_v3&) {
            status = BALOOT_NATIVE_SEARCH_BUDGET_EXHAUSTED;
        } catch (const time_budget_v3&) {
            status = BALOOT_NATIVE_SEARCH_BUDGET_EXHAUSTED;
        } catch (...) {
            status = BALOOT_NATIVE_SEARCH_FAILED;
        }
        export_result_v3(best, context.nodes(), status, *result);
    } catch (const cancelled_v3&) {
        export_result_v3(
            node_result_v3{}, 0u, BALOOT_NATIVE_SEARCH_CANCELLED, *result);
    } catch (...) {
        export_result_v3(node_result_v3{}, 0u, BALOOT_NATIVE_SEARCH_FAILED, *result);
    }
    return 0;
}

extern "C" int32_t baloot_heuristic_rollout_batch_v3(
    const baloot_rules_v3* rules,
    const baloot_rollout_task_v3* tasks,
    uint32_t count,
    uint8_t team,
    const volatile uint32_t* cancelled,
    baloot_search_result_v3* results) {
    return heuristic_rollout_batch_impl_v3(
        rules, tasks, count, team, cancelled, results, export_result_v3);
}

extern "C" int32_t baloot_heuristic_rollout_batch_compact_v3(
    const baloot_rules_v3* rules,
    const baloot_rollout_task_v3* tasks,
    uint32_t count,
    uint8_t team,
    const volatile uint32_t* cancelled,
    baloot_rollout_result_v3* results) {
    return heuristic_rollout_batch_impl_v3(
        rules, tasks, count, team, cancelled, results, export_rollout_result_v3);
}
