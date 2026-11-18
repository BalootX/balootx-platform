# Adjudication research boundary

The public authority is `configs/adjudication/research_events_v1.yaml`. Its retained source verification covers the 2023 regulation; current external ratification is still required.

Referee events are restricted evidence alongside a legal trace. They do not authorize generated illegal actions, hard scenario predicates, hidden-information disclosure, or automatic referee decisions.

## Incorrect highest card naming

The exact row records naming a card as the highest when it was not, with opponent choice to register with projects or continue; the neutral event does not claim a false-Ekka interpretation.

Source clause: `section_9.registration_table.incorrect_highest_card_naming`. Domain primitive: `adjudication.external_card_naming`.

Ordinary trace: `not_an_action`. Exact input: `audit_sidecar_only`. Referee required: `true`.

Available remedies: `opponent_register_with_projects`, `continue_play`.

## Out of turn

A play before turn does not mutate the legal trace and is preserved as a referee event with opponent remedy choice.

Source clause: `section_9.first_registration_table.play_before_turn`. Domain primitive: `rules.play.current_actor`.

Ordinary trace: `rejected`. Exact input: `legal_trace_rejects_attempt_audit_sidecar_supported`. Referee required: `true`.

Available remedies: `opponent_register_with_projects`, `continue_play`.

## Invalid bid

Purchase before turn never enters legal auction history and is preserved with the opponent takweesh-or-continue choice; other invalid-bid subtypes are not collapsed into this remedy.

Source clause: `section_9.takweesh_table.purchase_before_turn`. Domain primitive: `rules.auction.legal_actions`.

Ordinary trace: `rejected`. Exact input: `legal_trace_rejects_attempt_audit_sidecar_supported`. Referee required: `true`.

Available remedies: `opponent_takweesh`, `continue_play`.

## Exposed card

A dealt card exposed and named by the opponent is referee evidence with opponent choice to takweesh or continue, not a generated play action.

Source clause: `section_9.takweesh_table.exposed_dealt_card_named_by_opponent`. Domain primitive: `adjudication.external_card_exposure`.

Ordinary trace: `not_an_action`. Exact input: `audit_sidecar_only`. Referee required: `true`.

Available remedies: `opponent_takweesh`, `continue_play`.

## Signalling

Alleged signalling is referee evidence separate from play inference; first and second incidents warn, while a third or referee-evaluated incident may be registered against with projects.

Source clause: `section_2.2_and_section_9.signalling`. Domain primitive: `adjudication.external_communication`.

Ordinary trace: `not_an_action`. Exact input: `audit_sidecar_only`. Referee required: `true`.

Available remedies: `warning`, `register_against_with_projects`.

## Verbal baloot inside sequential 100

Baloot inside a sequential 100 scores automatically, but verbal declaration is forbidden and preserved as a separate referee event with opponent remedy choice.

Source clause: `section_5.clauses_5_6_and_5_8`. Domain primitive: `rules.projects.baloot_verbal_declaration`.

Ordinary trace: `not_an_action`. Exact input: `audit_sidecar_only`. Referee required: `true`.

Available remedies: `opponent_register_with_projects`, `continue_play`.
