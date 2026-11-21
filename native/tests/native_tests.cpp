#include "baloot/native.h"
#include "../src/dd_small_list.h"
#include <cassert>
#include <cstdint>
#include <iostream>
static void dd_small_list_edges() {
    baloot::dd::small_list<uint8_t,16> actions;
    actions.sort();
    for (int i = 15; i >= 0; --i) actions.push_back(static_cast<uint8_t>(i));
    actions.sort();
    assert(actions.size() == 16);
    for (std::size_t i = 0; i < actions.size(); ++i) assert(actions[i] == i);
    bool overflow = false;
    try { actions.push_back(16); } catch (const std::length_error&) { overflow = true; }
    assert(overflow && actions.size() == 16); // A seventeenth action cannot be stored.
    actions.sort(std::greater<uint8_t>{});
    for (std::size_t i = 0; i < actions.size(); ++i) assert(actions[i] == 15-i);
    baloot::dd::small_list<int,8> points;
    points.push_back(7);
    points.sort();
    assert(points.size() == 1 && points[0] == 7);
    for (int i = 6; i >= 0; --i) points.push_back(i);
    points.sort();
    assert(points.size() == 8);
    for (std::size_t i = 0; i < points.size(); ++i) assert(points[i] == static_cast<int>(i));
    overflow = false;
    try { points.push_back(8); } catch (const std::length_error&) { overflow = true; }
    assert(overflow && points.size() == 8);
    bool outside = false;
    try { (void)points[8]; } catch (const std::out_of_range&) { outside = true; }
    assert(outside);
}
static baloot_rules_v1 r1(){baloot_rules_v1 r{};r.abi_version=1;int sun[8]={0,1,2,6,3,4,5,7},tr[8]={0,1,6,4,7,2,3,5};for(int i=0;i<8;++i){r.sun_strength[i]=sun[i];r.hokum_non_trump_strength[i]=sun[i];r.hokum_trump_strength[i]=tr[i];}r.allow_discard_when_cannot_overtrump=1;r.partner_winning_exception=1;r.locked_hokum_blocks_trump_lead=1;return r;}
static baloot_rules_v2 r2(){baloot_rules_v2 r{};r.abi_version=2;int32_t sun[8]={0,1,2,6,3,4,5,7},tr[8]={0,1,6,4,7,2,3,5};int16_t sp[8]={0,0,0,10,2,3,4,11},tp[8]={0,0,14,10,20,3,4,11};for(int i=0;i<8;++i){r.strengths[0][i]=sun[i];r.strengths[1][i]=sun[i];r.strengths[2][i]=tr[i];r.points[0][i]=sp[i];r.points[1][i]=sp[i];r.points[2][i]=tp[i];}r.must_follow_suit=1;r.must_trump_when_void=1;r.must_overtrump_when_possible=1;r.allow_discard_when_cannot_overtrump=1;r.partner_winning_exception=1;r.locked_hokum_blocks_trump_lead=1;r.ekka_enabled=1;r.last_trick_bonus_raw=10;r.match_target=152;return r;}
int main(){dd_small_list_edges();auto a=r1();assert(baloot_native_abi_version()==1u);assert(baloot_native_abi_version_v2()==2u);assert(baloot_sizeof_rules_v2()==sizeof(baloot_rules_v2));assert(baloot_sizeof_state_v2()==sizeof(baloot_state_v2));assert(baloot_sizeof_play_v2()==sizeof(baloot_play_v2));assert((baloot_native_capabilities_v2()&BALOOT_NATIVE_CAP_APPLY_PLAY)!=0u);baloot_state_v1 s{};s.abi_version=1;s.contract=0;s.trump_suit=-1;s.current_player=1;s.trick_size=1;s.trick[0]={0,0,0,0};s.hands[1]=(1u<<1)|(1u<<8);assert(baloot_legal_moves_v1(&a,&s)==(1u<<1));auto b=r2();baloot_state_v2 v{};v.abi_version=2;v.contract=1;v.trump_suit=1;v.current_player=1;v.trick_size=1;v.trick[0]={0,0,0,0};v.hands[1]=(1u<<8)|(1u<<24);assert(baloot_legal_moves_v2(&b,&v)==(1u<<8));uint32_t out[2]{};baloot_state_v2 batch[2]={v,v};assert(baloot_legal_moves_batch_v2(&b,batch,2,out)==2&&out[0]==(1u<<8));baloot_play_v2 p[4]={{0,0,0,0},{1,12,0,0},{2,7,0,0},{3,15,0,0}};baloot_trick_result_v2 tr{};assert(baloot_evaluate_trick_v2(&b,1,1,p,&tr)==0&&tr.winner==1);auto before_hash=baloot_state_hash_v2(&v);assert(before_hash!=0);baloot_state_v2 ek{};ek.abi_version=2;ek.contract=1;ek.trump_suit=1;ek.current_player=0;ek.hands[0]=(1u<<6)|(1u<<7);assert(baloot_apply_play_v2(&b,&ek,6,1,0)==-4);assert(ek.hands[0]==((1u<<6)|(1u<<7)));assert(baloot_apply_play_v2(&b,&ek,7,1,0)==0);
baloot_undo_v2 undo{};assert(baloot_sizeof_undo_v2()==sizeof(baloot_undo_v2));assert((baloot_native_capabilities_v2()&BALOOT_NATIVE_CAP_APPLY_UNDO)!=0u);assert(baloot_apply_play_with_undo_v2(&b,&v,8,0,0,&undo)==0);assert(baloot_state_hash_v2(&v)!=before_hash);assert(baloot_undo_play_v2(&v,&undo)==0);assert(baloot_state_hash_v2(&v)==before_hash);
assert(baloot_native_abi_version_v3()==3u);assert((baloot_native_capabilities_v2()&BALOOT_NATIVE_CAP_EXACT_SEARCH)!=0u);assert(baloot_sizeof_rules_v3()==sizeof(baloot_rules_v3));assert(baloot_sizeof_state_v3()==sizeof(baloot_state_v3));
baloot_rules_v3 c{};c.base=b;c.base.baloot_game_points=2;c.base.kabout_sun=44;c.base.kabout_hokum=25;c.card_game_point_totals[0]=26;c.card_game_point_totals[1]=16;c.project_multiplier_cap=2;c.baloot_multiplier_cap=1;c.kabout_enabled=1;
baloot_state_v3 q{};q.base.abi_version=2;q.base.contract=0;q.base.trump_suit=-1;q.base.current_player=0;q.base.completed_tricks=7;q.base.escalation_multiplier=1;q.base.hands[0]=1u<<7;q.base.hands[1]=1u<<15;q.base.hands[2]=1u<<23;q.base.hands[3]=1u<<31;q.base.raw_points[0]=40;q.base.raw_points[1]=36;q.base.initial_hands[0]=q.base.hands[0];q.base.initial_hands[1]=q.base.hands[1];q.base.initial_hands[2]=q.base.hands[2];q.base.initial_hands[3]=q.base.hands[3];q.declarer=2;q.counting_team=1;q.tricks_won[0]=4;q.tricks_won[1]=3;
baloot_search_limits_v3 limits{};limits.maximum_nodes=10000;limits.maximum_milliseconds=1000;limits.maximum_cache_entries=1000;limits.maximum_depth=4;limits.team=0;uint32_t cancelled=0;baloot_search_result_v3 solved{};assert(baloot_exact_search_v3(&c,&q,&limits,&cancelled,&solved)==0);assert(solved.status==BALOOT_NATIVE_SEARCH_EXACT);assert(solved.exact==1u);assert(solved.principal_variation_length==4u);cancelled=1u;baloot_search_result_v3 cancelled_exact{};assert(baloot_exact_search_v3(&c,&q,&limits,&cancelled,&cancelled_exact)==0);assert(cancelled_exact.status==BALOOT_NATIVE_SEARCH_CANCELLED);assert(cancelled_exact.exact==0u);assert(cancelled_exact.nodes==0u);assert(cancelled_exact.principal_variation_length==0u);cancelled=0u;
baloot_rollout_task_v3 rollout[2]{};rollout[0].state=q;rollout[1].state=q;rollout[0].initial_card=BALOOT_NATIVE_ROLLOUT_NO_INITIAL_CARD;rollout[1].initial_card=BALOOT_NATIVE_ROLLOUT_NO_INITIAL_CARD;baloot_search_result_v3 rolled[2]{};assert(baloot_heuristic_rollout_batch_v3(&c,rollout,2,0,&cancelled,rolled)==2);assert(rolled[0].status==BALOOT_NATIVE_SEARCH_HEURISTIC_COMPLETE);assert(rolled[1].status==BALOOT_NATIVE_SEARCH_HEURISTIC_COMPLETE);assert(rolled[0].exact==0u);assert(rolled[1].exact==0u);assert(rolled[0].value==rolled[1].value);assert(rolled[0].own_points==rolled[1].own_points);assert(rolled[0].opponent_points==rolled[1].opponent_points);assert(rolled[0].principal_variation_length==4u);assert(rolled[1].principal_variation_length==4u);for(uint8_t i=0;i<rolled[0].principal_variation_length;++i)assert(rolled[0].principal_variation[i]==rolled[1].principal_variation[i]);baloot_rollout_result_v3 compact[2]{};assert(baloot_heuristic_rollout_batch_compact_v3(&c,rollout,2,0,&cancelled,compact)==2);for(uint8_t i=0;i<2u;++i){assert(compact[i].status==rolled[i].status);assert(compact[i].exact==rolled[i].exact);assert(compact[i].nodes==rolled[i].nodes);assert(compact[i].value==rolled[i].value);assert(compact[i].own_points==rolled[i].own_points);assert(compact[i].opponent_points==rolled[i].opponent_points);assert(compact[i].score_after_round[0]==rolled[i].score_after_round[0]);assert(compact[i].score_after_round[1]==rolled[i].score_after_round[1]);assert(compact[i].match_winner==rolled[i].match_winner);assert(compact[i].contract_succeeded==rolled[i].contract_succeeded);assert(compact[i].round_lead==rolled[i].round_lead);}rollout[0].initial_card=254u;assert(baloot_heuristic_rollout_batch_v3(&c,rollout,1,0,&cancelled,rolled)==-3);assert(baloot_heuristic_rollout_batch_compact_v3(&c,rollout,1,0,&cancelled,compact)==-3);rollout[0].initial_card=BALOOT_NATIVE_ROLLOUT_NO_INITIAL_CARD;cancelled=1u;assert(baloot_heuristic_rollout_batch_v3(&c,rollout,1,0,&cancelled,rolled)==-2);assert(baloot_heuristic_rollout_batch_compact_v3(&c,rollout,1,0,&cancelled,compact)==-2);cancelled=0u;
assert(baloot_sizeof_dd_result_v1()==sizeof(baloot_dd_result_v1));
assert(baloot_partial_trick_winner_v2(&b,1,1,p,1)==0);
assert(baloot_partial_trick_winner_v2(&b,1,1,p,2)==1);
assert(baloot_partial_trick_winner_v2(&b,1,1,p,3)==1);
assert(baloot_partial_trick_winner_v2(&b,1,1,p,4)==1);
assert(baloot_partial_trick_winner_v2(&b,1,1,p,0)==-1);
assert(baloot_double_dummy_abi_version()==1u);
int32_t payoffs[12*131]{};
for(int row=0;row<12;++row)for(int raw=0;raw<=130;++raw)payoffs[row*131+raw]=raw*2-130;
baloot_dd_result_v1 dd{};
assert(baloot_double_dummy_v1(&b,&q.base,payoffs,12*131,130,0,3,1,&limits,&cancelled,&dd)==0);
assert(dd.status==BALOOT_NATIVE_SEARCH_EXACT&&dd.best_action==7&&dd.values[7]==58&&dd.action_count==1);
assert(baloot_double_dummy_v1(&b,&q.base,payoffs,12*131,130,0,3,0,&limits,&cancelled,&dd)==0);
assert(dd.status==BALOOT_NATIVE_SEARCH_EXACT&&dd.best_action==7&&dd.values[7]==58);
cancelled=1u;
assert(baloot_double_dummy_v1(&b,&q.base,payoffs,12*131,130,0,3,1,&limits,&cancelled,&dd)==0);
assert(dd.status==BALOOT_NATIVE_SEARCH_CANCELLED&&dd.action_count==0&&dd.best_action==255);
cancelled=0u;
limits.maximum_milliseconds=0;
assert(baloot_double_dummy_v1(&b,&q.base,payoffs,12*131,130,0,3,1,&limits,&cancelled,&dd)==0);
assert(dd.status==BALOOT_NATIVE_SEARCH_EXACT&&dd.best_action==7&&dd.values[7]==58);
limits.maximum_nodes=1;
assert(baloot_double_dummy_v1(&b,&q.base,payoffs,12*131,130,0,3,1,&limits,&cancelled,&dd)==0);
assert(dd.status==BALOOT_NATIVE_SEARCH_BUDGET_EXHAUSTED&&dd.action_count==0&&dd.best_action==255);
limits.maximum_nodes=10000;
limits.maximum_milliseconds=1000;
assert(baloot_double_dummy_v1(&b,&q.base,payoffs,12*131-1,130,0,3,1,&limits,&cancelled,&dd)==-1);
baloot_state_v2 full{};full.abi_version=2;full.contract=0;full.trump_suit=-1;
for(int seat=0;seat<4;++seat)full.hands[seat]=0xffu<<(seat*8);
assert(baloot_double_dummy_v1(&b,&full,payoffs,12*131,130,4,0,1,&limits,&cancelled,&dd)==0);
assert(dd.status==BALOOT_NATIVE_SEARCH_EXACT&&dd.best_action==0&&dd.action_count==8);
for(int card=0;card<8;++card)assert(dd.actions[card]&&dd.values[card]==130);
assert(baloot_double_dummy_v1(&b,&full,payoffs,12*131,130,4,0,0,&limits,&cancelled,&dd)==0);
assert(dd.status==BALOOT_NATIVE_SEARCH_EXACT&&dd.best_action==0&&dd.values[0]==130);
int32_t hokum_payoffs[12*163]{};
for(int row=0;row<12;++row)for(int raw=0;raw<=162;++raw)hokum_payoffs[row*163+raw]=raw*2-162;
full.contract=1;full.trump_suit=1;
assert(baloot_double_dummy_v1(&b,&full,hokum_payoffs,12*163,162,4,0,1,&limits,&cancelled,&dd)==0);
assert(dd.status==BALOOT_NATIVE_SEARCH_EXACT&&dd.best_action==0);
for(int card=0;card<64;++card)if(dd.actions[card])assert(dd.values[card]==-162);
full.hands[0]|=1u<<8;
assert(baloot_double_dummy_v1(&b,&full,hokum_payoffs,12*163,162,4,0,1,&limits,&cancelled,&dd)==-1);
void* cache=baloot_dd_cache_create_v1(limits.maximum_cache_entries);assert(cache);
assert(!baloot_dd_cache_create_v1(1000001));
baloot_state_v2 late=q.base;late.completed_tricks=6;late.raw_points[0]=46;late.raw_points[1]=46;
late.hands[0]=(1u<<0)|(1u<<7);late.hands[1]=(1u<<8)|(1u<<9);
late.hands[2]=(1u<<3)|(1u<<10);late.hands[3]=(1u<<6)|(1u<<13);
assert(baloot_double_dummy_cached_v1(cache,&b,&late,payoffs,12*131,130,0,3,1,&limits,&cancelled,&dd)==0);
assert(dd.status==BALOOT_NATIVE_SEARCH_EXACT&&dd.best_action==7&&dd.values[7]==38&&dd.values[0]==-10);
uint64_t stats[4]{};assert(baloot_dd_cache_stats_v1(cache,stats)==0&&stats[0]>0);
const auto first_nodes=dd.nodes;
assert(baloot_double_dummy_cached_v1(cache,&b,&late,payoffs,12*131,130,0,3,1,&limits,&cancelled,&dd)==0);
assert(dd.status==BALOOT_NATIVE_SEARCH_EXACT&&dd.values[7]==38&&dd.nodes<=first_nodes);
assert(baloot_dd_cache_stats_v1(cache,stats)==0&&stats[1]>0);
int32_t changed_payoffs[12*131]{};
for(int offset=1;offset<=258;++offset){
 for(int row=0;row<12;++row)for(int raw=0;raw<=130;++raw)changed_payoffs[row*131+raw]=raw*2-130+offset;
 assert(baloot_double_dummy_cached_v1(cache,&b,&late,changed_payoffs,12*131,130,0,3,1,&limits,&cancelled,&dd)==0);
 assert(dd.status==BALOOT_NATIVE_SEARCH_EXACT&&dd.values[7]==38+offset&&dd.values[0]==-10+offset);
}
assert(baloot_dd_cache_stats_v1(cache,stats)==0&&stats[3]>0);
auto invalid=late;invalid.trump_broken=2;
assert(baloot_double_dummy_cached_v1(cache,&b,&invalid,payoffs,12*131,130,0,3,1,&limits,&cancelled,&dd)==-1);
invalid=late;invalid.hands[1]=invalid.hands[0];
assert(baloot_double_dummy_cached_v1(cache,&b,&invalid,payoffs,12*131,130,0,3,1,&limits,&cancelled,&dd)==-1);
invalid=late;invalid.hands[1]&=invalid.hands[1]-1;
assert(baloot_double_dummy_cached_v1(cache,&b,&invalid,payoffs,12*131,130,0,3,1,&limits,&cancelled,&dd)==-1);
invalid=late;invalid.raw_points[0]=100;
assert(baloot_double_dummy_cached_v1(cache,&b,&invalid,payoffs,12*131,130,0,3,1,&limits,&cancelled,&dd)==-1);
assert(baloot_sizeof_dd_task_v1()==sizeof(baloot_dd_task_v1));
baloot_dd_task_v1 tasks[8]{};
baloot_dd_result_v1 cohort[8]{};
for(int i=0;i<8;++i){tasks[i].state=late;tasks[i].payoffs=i%2?changed_payoffs:payoffs;
 tasks[i].payoff_count=12*131;tasks[i].raw_total=130;tasks[i].first_winner=0;tasks[i].won_mask=3;}
assert(baloot_double_dummy_batch_v1(cache,&b,tasks,8,4,&limits,&cancelled,cohort)==8);
for(int i=0;i<8;++i){assert(cohort[i].status==BALOOT_NATIVE_SEARCH_EXACT);
 assert(cohort[i].values[7]==38+(i%2?258:0));assert(cohort[i].values[0]==-10+(i%2?258:0));}
assert(baloot_double_dummy_batch_v1(cache,&b,tasks,8,0,&limits,&cancelled,cohort)==-1);
assert(baloot_double_dummy_batch_v1(cache,&b,tasks,8,5,&limits,&cancelled,cohort)==-1);
cancelled=1;
assert(baloot_double_dummy_batch_v1(cache,&b,tasks,8,4,&limits,&cancelled,cohort)==8);
for(const auto& row:cohort)assert(row.status!=BALOOT_NATIVE_SEARCH_EXACT&&row.action_count==0&&row.best_action==255);
cancelled=0;
tasks[7].state.hands[1]=tasks[7].state.hands[0];
assert(baloot_double_dummy_batch_v1(cache,&b,tasks,8,4,&limits,&cancelled,cohort)==-1);
baloot_dd_cache_free_v1(cache);
std::cout<<"native v1/v2 transitions, v3 exact and DD search passed\n";}
