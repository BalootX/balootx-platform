#define BALOOT_NATIVE_BUILD
#include "baloot/native.h"
#include "dd_small_list.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr int inf = 1000000000;
int team(int seat) { return seat & 1; }
struct stopped {};
struct node {
    baloot_state_v2 play;
    uint8_t first_winner, won_mask;
};
struct key {
    std::array<uint32_t, 4> hands;
    uint32_t history, score;
    bool operator==(const key&) const = default;
};
struct hash {
    std::size_t operator()(const key& k) const noexcept {
        uint64_t h = 1469598103934665603ull;
        for (auto v : k.hands) h = (h ^ v) * 1099511628211ull;
        h = (h ^ k.history) * 1099511628211ull;
        h = (h ^ k.score) * 1099511628211ull;
        h = (h ^ (h >> 30u)) * 0xbf58476d1ce4e5b9ull;
        h = (h ^ (h >> 27u)) * 0x94d049bb133111ebull;
        return static_cast<std::size_t>(h ^ (h >> 31u));
    }
};
key position(const node& n, uint8_t context = 0) {
    key k{};
    std::copy(std::begin(n.play.hands), std::end(n.play.hands), k.hands.begin());
    k.history = n.play.current_player | (n.play.trick_size << 2u);
    for (int i = 0; i < n.play.trick_size; ++i) {
        k.history |= (static_cast<uint32_t>(n.play.trick[i].card) |
            (static_cast<uint32_t>(n.play.trick[i].ekka) << 5u)) << (4u+6u*i);
    }
    k.score = static_cast<uint32_t>(n.play.raw_points[0]) |
        (static_cast<uint32_t>(n.first_winner) << 16u) |
        (static_cast<uint32_t>(n.won_mask) << 19u) |
        (static_cast<uint32_t>(n.play.trump_broken) << 21u) |
        (static_cast<uint32_t>(context) << 22u);
    return k;
}
struct entry { int lower = -inf, upper = inf; uint8_t action = 255; };
struct bucket { key position{}; entry value{}; uint64_t generation = 0; int remaining = 0; };
struct workspace {
    uint32_t capacity;
    uint64_t hits = 0, lookups = 0, resets = 0;
    std::vector<bucket> entries;
    std::atomic<uint64_t> generation{0};
    uint64_t used = 0;
    std::vector<std::string> contexts;
    std::array<std::mutex,64> locks;
    std::mutex statistics;
    explicit workspace(uint32_t limit) : capacity(limit), entries(limit) {}
    void reset() {
        for (auto& slot : entries) slot.generation = 0;
        generation = 0; used = 0; ++resets;
    }
    uint64_t next_epoch() {
        const auto value = generation.fetch_add(1,std::memory_order_relaxed)+1;
        if (!value) throw std::runtime_error("DD cache age exhausted");
        return value;
    }
    std::pair<std::size_t,std::size_t> range(const key& k) const {
        const auto ways = std::min<std::size_t>(2,entries.size());
        return {(hash{}(k) % (entries.size()/ways))*ways,ways};
    }
    std::optional<entry> find(const key& k, bool concurrent, uint64_t epoch) {
        if (entries.empty()) return {};
        const auto [start,ways] = range(k);
        std::unique_lock guard(locks[start%locks.size()],std::defer_lock);
        if (concurrent) guard.lock();
        for (std::size_t i = start; i < start+ways; ++i)
            if (entries[i].generation && entries[i].position == k) {
                entries[i].generation = std::max(entries[i].generation,epoch); return entries[i].value;
            }
        return {};
    }
    bool store(const key& k, int remaining, const entry& value, bool concurrent, uint64_t epoch) {
        if (entries.empty()) return false;
        const auto [start,ways] = range(k);
        std::unique_lock guard(locks[start%locks.size()],std::defer_lock);
        if (concurrent) guard.lock();
        auto* target = &entries[start];
        for (std::size_t i = start; i < start+ways; ++i) {
            auto& slot = entries[i];
            if (slot.generation && slot.position == k) {
                slot.value.lower = std::max(slot.value.lower,value.lower);
                slot.value.upper = std::min(slot.value.upper,value.upper);
                slot.generation = std::max(slot.generation,epoch);
                if (value.action != 255) slot.value.action = value.action;
                if (slot.value.lower > slot.value.upper) throw std::runtime_error("DD contradictory cached bounds");
                return false;
            }
            if (!slot.generation) { target = &slot; break; }
            if (slot.generation < target->generation || (slot.generation == target->generation &&
                    slot.remaining < target->remaining)) target = &slot;
        }
        if (target->generation >= epoch && target->remaining > remaining) return false;
        const bool empty = !target->generation;
        *target = bucket{k,value,epoch,remaining};
        return empty;
    }
    void record(uint64_t new_entries, uint64_t new_hits, uint64_t new_lookups) {
        std::lock_guard guard(statistics);
        used += new_entries; hits += new_hits; lookups += new_lookups;
    }
    uint8_t bind(const baloot_rules_v2& rules, const baloot_state_v2& state,
            const int32_t* payoffs, uint32_t count, uint16_t total) {
        std::string signature(reinterpret_cast<const char*>(&rules),sizeof(rules));
        signature.append(reinterpret_cast<const char*>(payoffs),count*sizeof(int32_t));
        signature.append(reinterpret_cast<const char*>(&total),sizeof(total));
        signature.push_back(static_cast<char>(state.contract));
        signature.push_back(static_cast<char>(state.trump_suit));
        signature.push_back(static_cast<char>(state.locked));
        auto found = std::find(contexts.begin(),contexts.end(),signature);
        if (found != contexts.end()) return static_cast<uint8_t>(found-contexts.begin());
        if (contexts.size() == 256) { reset(); contexts.clear(); }
        contexts.push_back(std::move(signature));
        return static_cast<uint8_t>(contexts.size()-1);
    }
};
using action_list = baloot::dd::small_list<uint8_t,16>;
class solver {
    const baloot_rules_v2& rules;
    const baloot_search_limits_v3& limits;
    const int32_t* payoffs;
    const volatile uint32_t* cancelled;
    int total;
    std::chrono::steady_clock::time_point deadline;
    workspace& shared;
    uint8_t context;
    bool concurrent;
    uint64_t epoch;
    std::array<bool, 12> monotone{};
    std::array<std::array<uint8_t,2>,32> killers{};
    int payoff(int first, int sweep, int raw) const {
        if (first < 0 || first > 3 || sweep < 0 || sweep > 2 || raw < 0 || raw > total)
            throw std::runtime_error("DD terminal payoff index outside table");
        return payoffs[(first*3+sweep)*(total+1)+raw];
    }
    int point(uint8_t card, const node& n) const {
        return baloot_card_points_v2(&rules, card, n.play.contract, n.play.trump_suit);
    }
    int strength(uint8_t card, const node& n) const {
        const auto table = n.play.contract == 1 ?
            (card/8 == n.play.trump_suit ? 2 : 1) : 0;
        return rules.strengths[table][card%8];
    }
    bool ekka(const node& n, uint8_t card) const {
        if (!rules.ekka_enabled || n.play.contract != 1 || n.play.trick_size || card/8 == n.play.trump_suit) return false;
        uint32_t remaining = n.play.hands[0] | n.play.hands[1] | n.play.hands[2] | n.play.hands[3];
        while (remaining) {
            const auto other = static_cast<uint8_t>(std::countr_zero(remaining));
            remaining &= remaining-1;
            if (other/8 == card/8 && rules.strengths[0][other%8] > rules.strengths[0][card%8]) return false;
        }
        return true;
    }
    bool equivalent(const node& n, uint8_t a, uint8_t b) const {
        if (a/8 != b/8 || point(a,n) != point(b,n) || ekka(n,a) != ekka(n,b)) return false;
        const int lo = std::min(strength(a,n),strength(b,n)), hi = std::max(strength(a,n),strength(b,n));
        uint32_t remaining = n.play.hands[0] | n.play.hands[1] | n.play.hands[2] | n.play.hands[3];
        for (int i = 0; i < n.play.trick_size; ++i) remaining |= 1u << n.play.trick[i].card;
        while (remaining) {
            const auto c = static_cast<uint8_t>(std::countr_zero(remaining));
            remaining &= remaining-1;
            if (c/8 == a/8 && strength(c,n) > lo && strength(c,n) < hi &&
                !(n.play.hands[n.play.current_player] & (1u << c))) return false;
        }
        return true;
    }
    action_list actions(const node& n, uint8_t preferred, bool collapse) const {
        action_list result;
        auto legal = baloot_legal_moves_v2(&rules, &n.play);
        while (legal) {
            const auto card = static_cast<uint8_t>(std::countr_zero(legal)); legal &= legal-1;
            if (collapse && std::any_of(result.begin(),result.end(),[&](uint8_t prior) {
                    return prior < 32 && equivalent(n, prior, card); })) continue;
            result.push_back(card);
            if (ekka(n,card)) result.push_back(static_cast<uint8_t>(card+32));
        }
        std::array<int,64> order{};
        for (auto action : result) {
            const auto card = static_cast<uint8_t>(action%32);
            int score = strength(card,n)*32 + point(card,n);
            if (n.play.trick_size) {
                std::array<baloot_play_v2,4> trick{};
                std::copy_n(n.play.trick,n.play.trick_size,trick.begin());
                trick[n.play.trick_size] = {n.play.current_player,card,static_cast<uint8_t>(action>=32),0};
                const auto winner = baloot_partial_trick_winner_v2(&rules,n.play.contract,n.play.trump_suit,
                    trick.data(),static_cast<uint8_t>(n.play.trick_size+1));
                // Ordering only: all legal alternatives remain in the search.
                score = team(winner) == team(n.play.current_player) ?
                    10000 + (winner != n.play.current_player ? point(card,n)*64 : -strength(card,n)*32) :
                    -point(card,n)*64-strength(card,n);
            }
            const auto& killer = killers[n.play.completed_tricks*4+n.play.trick_size];
            score += action == killer[0] ? 512 : action == killer[1] ? 256 : 0;
            order[action] = action == preferred ? inf : score;
        }
        result.sort([&](uint8_t a,uint8_t b) {
            const auto left = order[a], right = order[b];
            return left != right ? left > right : a < b;
        });
        return result;
    }
    node advance(node n, uint8_t action) const {
        const auto card = static_cast<uint8_t>(action%32);
        const auto led = n.play.trick_size ? n.play.trick[0].card/8 : card/8;
        const auto old_count = n.play.completed_tricks;
        if (baloot_apply_play_v2(&rules,&n.play,card,action >= 32,0) < 0) throw std::runtime_error("DD illegal transition");
        if (card/8 == n.play.trump_suit && led != n.play.trump_suit) n.play.trump_broken = 1;
        if (n.play.completed_tricks != old_count) {
            n.won_mask |= static_cast<uint8_t>(1u << team(n.play.current_player));
            if (old_count == 0) n.first_winner = n.play.current_player;
        }
        return n;
    }
    std::pair<int,int> range_bounds(const node& n, int start, int end) const {
        if (start < 0 || end > total || start > end) throw std::runtime_error("DD invalid point interval");
        int lo = inf, hi = -inf;
        for (int first = 0; first < 4; ++first) {
            if (n.first_winner != 4 && first != n.first_winner) continue;
            for (int sweep = 0; sweep < 3; ++sweep) {
                if ((sweep == 1 && (n.won_mask & 2)) || (sweep == 2 && (n.won_mask & 1))) continue;
                const int row = first*3+sweep;
                const auto* values = payoffs + row*(total+1);
                if (monotone[row]) {
                    lo = std::min(lo,values[start]); hi = std::max(hi,values[end]);
                } else for (int raw = start; raw <= end; ++raw) {
                    lo = std::min(lo,values[raw]); hi = std::max(hi,values[raw]);
                }
            }
        }
        return {lo,hi};
    }
    uint32_t masters(const node& n, int seat, uint32_t candidates) const {
        uint32_t outside = 0, result = 0;
        for (int other = 0; other < 4; ++other) if (other != seat) outside |= n.play.hands[other];
        while (candidates) {
            const auto card = static_cast<uint8_t>(std::countr_zero(candidates)); candidates &= candidates-1;
            bool master = true;
            auto scan = outside;
            while (scan) {
                const auto other = static_cast<uint8_t>(std::countr_zero(scan)); scan &= scan-1;
                if ((other/8 == card/8 && strength(other,n) > strength(card,n)) ||
                    (n.play.contract == 1 && card/8 != n.play.trump_suit && other/8 == n.play.trump_suit)) {
                    master = false; break;
                }
            }
            if (master) result |= 1u << card;
        }
        return result;
    }
    int cash_remainder_value(const node& n, int winner) const {
        const int won = n.won_mask | (1 << team(winner));
        const int sweep = won == 1 ? 1 : won == 2 ? 2 : 0;
        const int first = n.first_winner == 4 ? winner : n.first_winner;
        const int raw = team(winner) == 0 ? total-n.play.raw_points[1] : n.play.raw_points[0];
        return payoff(first,sweep,raw);
    }
    std::pair<int,int> bounds(const node& n) const {
        const int start = n.play.raw_points[0], end = total-n.play.raw_points[1];
        auto result = range_bounds(n,start,end);
        if (n.play.completed_tricks == 8) return result;
        if (n.play.trick_size) {
            const auto winner = baloot_partial_trick_winner_v2(&rules,n.play.contract,n.play.trump_suit,
                n.play.trick,n.play.trick_size);
            if (masters(n,winner,n.play.hands[winner]) != n.play.hands[winner]) return result;
            std::array<baloot_play_v2,4> trick{};
            std::copy_n(n.play.trick,n.play.trick_size,trick.begin());
            for (int step = n.play.trick_size; step < 4; ++step) {
                const auto seat = static_cast<uint8_t>((n.play.current_player+step-n.play.trick_size)%4);
                auto cards = n.play.hands[seat];
                while (cards) {
                    const auto card = static_cast<uint8_t>(std::countr_zero(cards)); cards &= cards-1;
                    trick[n.play.trick_size] = {seat,card,0,0};
                    if (baloot_partial_trick_winner_v2(&rules,n.play.contract,n.play.trump_suit,
                            trick.data(),static_cast<uint8_t>(n.play.trick_size+1)) != winner) return result;
                }
            }
            // Secure this trick, then every card in its winner's hand is a
            // master: no remaining choice can change the terminal allocation.
            const int value = cash_remainder_value(n,winner);
            return {value,value};
        }
        auto cash = masters(n,n.play.current_player,baloot_legal_moves_v2(&rules,&n.play));
        const auto count = std::popcount(cash);
        if (!count) return result;
        int guaranteed = 0;
        while (cash) {
            const auto card = static_cast<uint8_t>(std::countr_zero(cash)); cash &= cash-1;
            guaranteed += point(card,n);
        }
        for (int seat = 0; seat < 4; ++seat) if (seat != n.play.current_player) {
            baloot::dd::small_list<int,8> points;
            auto cards = n.play.hands[seat];
            while (cards) {
                const auto card = static_cast<uint8_t>(std::countr_zero(cards)); cards &= cards-1;
                points.push_back(point(card,n));
            }
            points.sort();
            for (int i = 0; i < count; ++i) guaranteed += points[i];
        }
        // This is one enforceable cashing strategy, not a restriction on all
        // legal lines: tighten only its team's own minimax bound.
        auto strategy = n;
        strategy.won_mask |= static_cast<uint8_t>(1 << team(n.play.current_player));
        if (strategy.first_winner == 4) strategy.first_winner = n.play.current_player;
        const bool all = count == std::popcount(n.play.hands[n.play.current_player]);
        const int value = all ? cash_remainder_value(n,n.play.current_player) : 0;
        if (team(n.play.current_player) == 0)
            result.first = std::max(result.first,all ? value : range_bounds(strategy,std::min(end,start+guaranteed),end).first);
        else result.second = std::min(result.second,all ? value : range_bounds(strategy,start,std::max(start,end-guaranteed)).second);
        return result;
    }
    void check() {
        if (++nodes > limits.maximum_nodes || ((nodes & 63u) == 0u &&
            ((cancelled && *cancelled) || std::chrono::steady_clock::now() >= deadline))) throw stopped{};
        // Long-running worlds must retain a fair share of the bounded table
        // while other workers finish and start newer worlds.
        if (concurrent && (nodes & 65535u) == 0u) epoch = shared.next_epoch();
    }
    void finish(baloot_dd_result_v1& result) {
        check();
        if ((cancelled && *cancelled) || std::chrono::steady_clock::now() >= deadline) throw stopped{};
        result.status = BALOOT_NATIVE_SEARCH_EXACT;
    }
    int visit(node n, int alpha, int beta) {
        check();
        // Forced cards need no allocation, hash lookup or minimax branch.
        while (n.play.completed_tricks != 8) {
            const auto legal = baloot_legal_moves_v2(&rules,&n.play);
            if (!std::has_single_bit(legal)) break;
            const auto card = static_cast<uint8_t>(std::countr_zero(legal));
            if (ekka(n,card)) break;
            n = advance(n,card);
            check();
        }
        if (n.play.completed_tricks == 8) {
            const int sweep = n.won_mask == 1 ? 1 : n.won_mask == 2 ? 2 : 0;
            return payoff(n.first_winner,sweep,n.play.raw_points[0]);
        }
        const int old_alpha = alpha, old_beta = beta;
        const auto k = position(n,context);
        uint8_t preferred = 255;
        ++lookups;
        if (auto saved = shared.find(k,concurrent,epoch)) {
            ++hits;
            preferred = saved->action;
            if (saved->lower >= beta) return saved->lower;
            if (saved->upper <= alpha) return saved->upper;
            alpha = std::max(alpha,saved->lower); beta = std::min(beta,saved->upper);
        }
        auto [lower,upper] = bounds(n);
        if (lower >= beta) return lower;
        if (upper <= alpha) return upper;
        alpha = std::max(alpha,lower); beta = std::min(beta,upper);
        if (lower == upper) return lower;
        const bool maximize = team(n.play.current_player) == 0;
        int best = maximize ? -inf : inf;
        uint8_t best_action = 255;
        for (auto action : actions(n,preferred,true)) {
            const int value = visit(advance(n,action),alpha,beta);
            if (best_action == 255 || (maximize ? value > best : value < best)) { best = value; best_action = action; }
            if (maximize) alpha = std::max(alpha,best); else beta = std::min(beta,best);
            if (alpha >= beta) {
                auto& killer = killers[n.play.completed_tricks*4+n.play.trick_size];
                if (killer[0] != action) { killer[1] = killer[0]; killer[0] = action; }
                break;
            }
        }
        int remaining = 0;
        for (auto hand : n.play.hands) remaining += std::popcount(hand);
        entry value;
        value.action = best_action;
        if (best <= old_alpha) value.upper = best;
        else if (best >= old_beta) value.lower = best;
        else value.lower = value.upper = best;
        if (shared.store(k,remaining,value,concurrent,epoch)) ++new_entries;
        return best;
    }
public:
    uint64_t nodes = 0, hits = 0, lookups = 0, new_entries = 0;
    solver(const baloot_rules_v2& r, const baloot_search_limits_v3& l,
           const int32_t* p, int t, const volatile uint32_t* c, workspace& work, uint8_t context_id,
           bool parallel = false)
        : rules(r), limits(l), payoffs(p), cancelled(c), total(t),
          deadline(l.maximum_milliseconds ? std::chrono::steady_clock::now()+std::chrono::milliseconds(l.maximum_milliseconds)
                                          : std::chrono::steady_clock::time_point::max()),
          shared(work), context(context_id), concurrent(parallel), epoch(work.next_epoch()) {
        for (auto& killer : killers) killer.fill(255);
        for (int row = 0; row < 12; ++row)
            monotone[row] = std::is_sorted(payoffs+row*(total+1),payoffs+(row+1)*(total+1));
    }
    int converge(const node& root) {
        check();
        auto [lower,upper] = bounds(root);
        int guess = std::clamp(0,lower,upper);
        while (lower < upper) {
            const int beta = guess == lower ? guess+1 : guess;
            guess = visit(root,beta-1,beta);
            if (guess < beta) upper = guess; else lower = guess;
        }
        return guess;
    }
    void solve(const node& root, baloot_dd_result_v1& result, bool all_roots) {
        auto roots = actions(root,255,false);
        roots.sort();
        if (!all_roots) {
            const int value = converge(root);
            for (auto action : roots) {
                const auto child = advance(root,action);
                const int bound = team(root.play.current_player) == 0 ?
                    visit(child,value-1,value) : visit(child,value,value+1);
                if (team(root.play.current_player) == 0 ? bound >= value : bound <= value) {
                    result.actions[action] = 1;
                    result.values[action] = value;
                    result.action_count = 1;
                    result.best_action = action;
                    finish(result);
                    return;
                }
            }
            throw std::runtime_error("DD optimal root was not recovered");
        }
        result.action_count = static_cast<uint8_t>(roots.size());
        int best = team(root.play.current_player) == 0 ? -inf : inf;
        for (auto action : roots) {
            auto child = advance(root,action);
            const int guess = converge(child);
            result.actions[action] = 1;
            result.values[action] = guess;
            if (result.best_action == 255 || (team(root.play.current_player) == 0 ? guess > best : guess < best)) {
                best = guess; result.best_action = action;
            }
        }
        finish(result);
    }
};
}
extern "C" uint32_t baloot_sizeof_dd_result_v1(void) { return sizeof(baloot_dd_result_v1); }
extern "C" uint32_t baloot_double_dummy_abi_version(void) { return 1u; }
static int32_t validate_dd(workspace* provided,const baloot_rules_v2* rules,const baloot_state_v2* state,
    const int32_t* payoffs,uint32_t payoff_count,uint16_t raw_total,uint8_t first_winner,uint8_t won_mask,uint8_t all_roots,
    const baloot_search_limits_v3* limits,const volatile uint32_t*,baloot_dd_result_v1* result) {
    if (!rules || !state || !payoffs || !limits || !result || rules->abi_version != 2 || state->abi_version != 2 ||
        payoff_count != 12u*(raw_total+1u) || raw_total > 1000 || first_winner > 4 || won_mask > 3 ||
        state->current_player > 3 || state->completed_tricks >= 8 || state->trick_size > 3 ||
        state->contract > 1 || state->locked > 1 || state->trump_broken > 1 ||
        (state->contract == 0 ? state->trump_suit != -1 : state->trump_suit < 0 || state->trump_suit > 3) ||
        state->raw_points[0] < 0 || state->raw_points[1] < 0 ||
        state->raw_points[0] > raw_total || state->raw_points[1] > raw_total ||
        state->raw_points[0]+state->raw_points[1] > raw_total ||
        (state->completed_tricks == 0 ? first_winner != 4 : first_winner > 3) ||
        all_roots > 1 || !limits->maximum_nodes) return -1;
    if (provided && provided->capacity != limits->maximum_cache_entries) return -1;
    uint32_t seen = 0;
    std::array<int,4> expected;
    expected.fill(8-state->completed_tricks);
    int unawarded = rules->last_trick_bonus_raw;
    for (int i = 0; i < state->trick_size; ++i) {
        const auto& play = state->trick[i];
        if (play.card > 31 || play.seat > 3 || play.ekka > 1 ||
                play.seat != (state->current_player+4-state->trick_size+i)%4 ||
                (seen & (1u << play.card))) return -1;
        seen |= 1u << play.card;
        --expected[play.seat];
        unawarded += baloot_card_points_v2(rules,play.card,state->contract,state->trump_suit);
    }
    for (int seat = 0; seat < 4; ++seat) {
        auto hand = state->hands[seat];
        if (std::popcount(hand) != expected[seat] || (seen & hand)) return -1;
        seen |= hand;
        while (hand) {
            const auto card = static_cast<uint8_t>(std::countr_zero(hand)); hand &= hand-1;
            unawarded += baloot_card_points_v2(rules,card,state->contract,state->trump_suit);
        }
    }
    if (unawarded+state->raw_points[0]+state->raw_points[1] != raw_total) return -1;
    return 0;
}
static void search_dd(workspace& work,uint8_t context,const baloot_rules_v2& rules,
    const baloot_state_v2& state,const int32_t* payoffs,uint16_t raw_total,
    uint8_t first_winner,uint8_t won_mask,bool all_roots,const baloot_search_limits_v3& limits,
    const volatile uint32_t* cancelled,baloot_dd_result_v1* result,bool concurrent = false) {
    *result = {}; result->best_action = 255; result->status = BALOOT_NATIVE_SEARCH_BUDGET_EXHAUSTED;
    try {
        solver search(rules,limits,payoffs,raw_total,cancelled,work,context,concurrent);
        try { search.solve(node{state,first_winner,won_mask},*result,all_roots); }
        catch (const stopped&) { if (cancelled && *cancelled) result->status = BALOOT_NATIVE_SEARCH_CANCELLED; }
        catch (...) { result->status = BALOOT_NATIVE_SEARCH_FAILED; }
        result->nodes = search.nodes;
        work.record(search.new_entries,search.hits,search.lookups);
    } catch (...) { result->status = BALOOT_NATIVE_SEARCH_FAILED; }
    if (result->status != BALOOT_NATIVE_SEARCH_EXACT) {
        result->best_action = 255;
        result->action_count = 0;
        std::fill(std::begin(result->values),std::end(result->values),0);
        std::fill(std::begin(result->actions),std::end(result->actions),uint8_t{0});
    }
}
static int32_t solve_dd(workspace* provided,const baloot_rules_v2* rules,const baloot_state_v2* state,
    const int32_t* payoffs,uint32_t payoff_count,uint16_t raw_total,uint8_t first_winner,uint8_t won_mask,uint8_t all_roots,
    const baloot_search_limits_v3* limits,const volatile uint32_t* cancelled,baloot_dd_result_v1* result) {
    if (validate_dd(provided,rules,state,payoffs,payoff_count,raw_total,first_winner,won_mask,
            all_roots,limits,cancelled,result)) return -1;
    try {
        std::unique_ptr<workspace> owned;
        if (!provided) owned = std::make_unique<workspace>(limits->maximum_cache_entries);
        auto& work = provided ? *provided : *owned;
        const auto context = work.bind(*rules,*state,payoffs,payoff_count,raw_total);
        search_dd(work,context,*rules,*state,payoffs,raw_total,first_winner,won_mask,all_roots != 0,*limits,cancelled,result);
    } catch (...) {
        *result = {}; result->status = BALOOT_NATIVE_SEARCH_FAILED; result->best_action = 255;
    }
    return 0;
}
extern "C" int32_t baloot_double_dummy_v1(const baloot_rules_v2* rules,const baloot_state_v2* state,
    const int32_t* payoffs,uint32_t payoff_count,uint16_t raw_total,uint8_t first_winner,uint8_t won_mask,uint8_t all_roots,
    const baloot_search_limits_v3* limits,const volatile uint32_t* cancelled,baloot_dd_result_v1* result) {
    return solve_dd(nullptr,rules,state,payoffs,payoff_count,raw_total,first_winner,won_mask,all_roots,limits,cancelled,result);
}
extern "C" void* baloot_dd_cache_create_v1(uint32_t capacity) {
    if (!capacity || capacity > 1000000u) return nullptr;
    try { return new workspace(capacity); } catch (...) { return nullptr; }
}
extern "C" void baloot_dd_cache_free_v1(void* cache) { delete static_cast<workspace*>(cache); }
extern "C" int32_t baloot_dd_cache_stats_v1(void* cache,uint64_t output[4]) {
    if (!cache || !output) return -1;
    const auto& work = *static_cast<workspace*>(cache);
    output[0]=work.used;output[1]=work.hits;output[2]=work.lookups;output[3]=work.resets;
    return 0;
}
extern "C" int32_t baloot_double_dummy_cached_v1(void* cache,const baloot_rules_v2* rules,const baloot_state_v2* state,
    const int32_t* payoffs,uint32_t payoff_count,uint16_t raw_total,uint8_t first_winner,uint8_t won_mask,uint8_t all_roots,
    const baloot_search_limits_v3* limits,const volatile uint32_t* cancelled,baloot_dd_result_v1* result) {
    if (!cache) return -1;
    return solve_dd(static_cast<workspace*>(cache),rules,state,payoffs,payoff_count,raw_total,first_winner,won_mask,all_roots,limits,cancelled,result);
}
extern "C" uint32_t baloot_sizeof_dd_task_v1(void) { return sizeof(baloot_dd_task_v1); }
extern "C" int32_t baloot_double_dummy_batch_v1(void* cache,const baloot_rules_v2* rules,
    const baloot_dd_task_v1* tasks,uint32_t count,uint8_t threads,const baloot_search_limits_v3* limits,
    const volatile uint32_t* cancelled,baloot_dd_result_v1* results) {
    if (!cache || !tasks || !results || !limits || !limits->maximum_milliseconds ||
        !count || count > 256 || !threads || threads > 4) return -1;
    auto& work = *static_cast<workspace*>(cache);
    for (uint32_t i = 0; i < count; ++i) {
        const auto& t = tasks[i];
        if (validate_dd(&work,rules,&t.state,t.payoffs,t.payoff_count,t.raw_total,t.first_winner,t.won_mask,
                1,limits,cancelled,&results[i])) return -1;
    }
    try {
        const auto deadline = std::chrono::steady_clock::now()+std::chrono::milliseconds(limits->maximum_milliseconds);
        // Bind all contexts before workers start; no registry eviction may
        // invalidate an earlier task's context number in this cohort.
        if (work.contexts.size()+count > 256) { work.reset(); work.contexts.clear(); }
        std::vector<uint8_t> contexts;
        for (uint32_t i = 0; i < count; ++i) {
            const auto& t = tasks[i];
            contexts.push_back(work.bind(*rules,t.state,t.payoffs,t.payoff_count,t.raw_total));
            results[i] = {}; results[i].best_action = 255;
            results[i].status = BALOOT_NATIVE_SEARCH_BUDGET_EXHAUSTED;
        }
        std::atomic<uint32_t> next{0};
        std::vector<std::jthread> workers;
        const auto worker_count = std::min<uint32_t>(threads,count);
        for (uint32_t worker = 0; worker < worker_count; ++worker) workers.emplace_back([&] {
            for (auto i = next.fetch_add(1); i < count; i = next.fetch_add(1)) {
                const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline-std::chrono::steady_clock::now()).count();
                if (left <= 0 || (cancelled && *cancelled)) continue;
                auto bound = *limits;
                bound.maximum_milliseconds = static_cast<uint32_t>(left);
                const auto& t = tasks[i];
                search_dd(work,contexts[i],*rules,t.state,t.payoffs,t.raw_total,t.first_winner,t.won_mask,
                    true,bound,cancelled,&results[i],worker_count > 1);
            }
        });
    } catch (...) { return -2; }
    return static_cast<int32_t>(count);
}
