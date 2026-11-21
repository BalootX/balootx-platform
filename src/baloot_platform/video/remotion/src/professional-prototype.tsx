import React from "react";
import {QRCodeSVG} from "qrcode.react";
import {AbsoluteFill, Artifact, cancelRender, continueRender, delayRender, Easing, Img, interpolate, staticFile, useCurrentFrame} from "remotion";
import {
  derivePlaybackState,
  collectionOpacity,
  playbackMotionTiming,
  type PlaybackPlay as PublicPlay,
  type PlaybackTrick as PublicTrick,
  type Seat,
} from "./playback-state";
import type {CardValue, PrototypeProps, SceneDocument, SceneLayer} from "./types";
import "./style.css";

const seatRing: Seat[] = ["north", "west", "south", "east"];
type PublicProject = {
  cards: CardValue[] | null;
  kind: string;
  owner: Seat;
  revealed: boolean;
  scoring: boolean;
  value: number;
};
type CardFace = Pick<CardValue, "rank" | "suit">;

const rankCode: Record<string, string> = {
  seven: "7",
  eight: "8",
  nine: "9",
  ten: "10",
  jack: "J",
  queen: "Q",
  king: "K",
  ace: "A",
};
const suitSymbol: Record<string, string> = {
  spades: "♠",
  hearts: "♥",
  clubs: "♣",
  diamonds: "♦",
};
const suitCode: Record<string, string> = {
  spades: "S",
  hearts: "H",
  clubs: "C",
  diamonds: "D",
};
const redSuit = (suit: string): boolean => suit === "hearts" || suit === "diamonds";

const sceneLayer = (scene: SceneDocument | undefined, id: string): SceneLayer | undefined =>
  scene?.layers.find((item) => item.id === id);
const asRecord = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
const asCards = (value: unknown): CardValue[] => (Array.isArray(value) ? (value as CardValue[]) : []);
const asNumbers = (value: unknown): number[] =>
  Array.isArray(value) ? value.map((item) => Number(item ?? 0)) : [0, 0];
const clamp = (value: number, minimum: number, maximum: number): number =>
  Math.min(maximum, Math.max(minimum, value));
const canonicalCardCode = (card: CardFace): string =>
  `${suitCode[card.suit] ?? ""}${rankCode[card.rank] ?? card.rank}`;
const cardLabel = (card: CardFace): string =>
  `${rankCode[card.rank] ?? card.rank}${suitSymbol[card.suit] ?? card.suit}`;
const isExactQualifiedPublishableSource = (source: PrototypeProps["bundle"]["source"]): boolean =>
  source?.qualificationStatus === "qualified" &&
  source.renderPurpose === "content_candidate" &&
  source.auditOnly === false &&
  source.publicationEligible === true &&
  Array.isArray(source.qualificationFailures) &&
  source.qualificationFailures.length === 0 &&
  Array.isArray(source.publicationBlockers) &&
  source.publicationBlockers.length === 0;

const cardPath = (card: CardFace): string =>
  staticFile(`cards/packs/goodall_modern_baloot_32/png/${rankCode[card.rank] ?? card.rank}_${card.suit}.png`);

// Every colour the Majlis look paints. Values come only from presentation.majlis.palette (premium.yaml).
export const majlisPaletteTokens = [
  "w1", "w2", "w3", "rug", "rug_line", "rug_deep", "accent", "m1", "m2", "c1", "c2", "wd1", "wd2",
  "sand", "brass", "brass_light", "card_red", "research_red", "research_tint", "white",
  "team_us", "team_us_text", "team_them", "team_them_text", "ink", "ink_warm", "cream", "parchment",
  "muted", "brown", "edge", "pill_edge", "tile", "meter", "hud_inner", "pass_bg", "pass_text",
  "escalation_bg", "escalation_red", "escalation_ink", "baloot_light", "baloot_edge", "ledger_line",
  "ledger_plus", "timer_track", "decision_red", "escalation_badge",
  "ceiling", "frond", "frond_light", "frond_dark", "mud", "vent", "lantern_body", "lantern_metal",
  "lantern_core", "lantern_glow", "floor_top", "floor_bottom", "smoke", "burner", "tray_rim", "tray",
  "tray_shadow", "coffee", "date_dark", "date", "date_light",
] as const;
type MajlisToken = (typeof majlisPaletteTokens)[number];
const hexColour = /^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$/;
const cssToken = (token: MajlisToken): string => `--mj-${token.replace(/_/g, "-")}`;
export const majlisPaletteStyle = (palette: Record<string, unknown>): React.CSSProperties => {
  const style: Record<string, string> = {};
  for (const token of majlisPaletteTokens) {
    const colour = palette[token];
    if (typeof colour !== "string" || !hexColour.test(colour)) {
      throw new Error(`Missing or invalid governed majlis colour: ${token}`);
    }
    style[cssToken(token)] = colour;
  }
  return style as React.CSSProperties;
};
const paint = (token: MajlisToken): string => `var(${cssToken(token)})`;
const fill = (token: MajlisToken): React.CSSProperties => ({fill: paint(token)});
const stroke = (token: MajlisToken): React.CSSProperties => ({stroke: paint(token)});
const governedChoice = <T extends string>(value: unknown, options: readonly T[], name: string): T => {
  if (!options.includes(value as T)) throw new Error(`Missing or invalid governed majlis ${name}`);
  return value as T;
};

// size="small" is for tags, bubbles, trays and the medallion: it never carries a hand card id.
const Card: React.FC<{card: CardFace & {id?: number}; size?: "small"; className?: string; style?: React.CSSProperties}> = ({card, size, className, style}) => (
  size === "small"
    ? <Img className={`card card-small ${className ?? ""}`} data-small-card-code={canonicalCardCode(card)} style={style} src={cardPath(card)} alt="" />
    : <Img className={`card ${className ?? ""}`} data-card-id={card.id} data-card-code={canonicalCardCode(card)} style={style} src={cardPath(card)} alt="" />
);

const PlaybackDomArtifact: React.FC<{frame: number}> = ({frame}) => {
  const [snapshot, setSnapshot] = React.useState<string | null>(null);
  React.useLayoutEffect(() => {
    const table = document.querySelector<HTMLElement>(".trick-zone");
    if (!table) throw new Error("Playback checkpoint table is missing");
    const surface = table.closest<HTMLElement>(".play-table");
    if (!surface) throw new Error("Playback checkpoint clipping surface is missing");
    const images = Array.from(table.querySelectorAll<HTMLImageElement>(".played.card"));
    const handle = delayRender("capture decoded playback cards");
    const loaded = (image: HTMLImageElement): Promise<void> => new Promise((resolve, reject) => {
      if (image.complete && image.currentSrc) {
        return image.naturalWidth > 0
          ? resolve()
          : reject(new Error(`Playback card image failed: ${image.dataset.cardCode}`));
      }
      image.addEventListener("load", () => resolve(), {once: true});
      image.addEventListener("error", () => reject(new Error(`Playback card image failed: ${image.dataset.cardCode}`)), {once: true});
    });
    void Promise.all(images.map(loaded)).then(() => {
      const clip = surface.getBoundingClientRect();
      const cards = images.map(card => {
        const style = getComputedStyle(card);
        const bounds = card.getBoundingClientRect();
        return {cardId: Number(card.dataset.cardId), cardCode: card.dataset.cardCode,
          imageLoaded: card.complete && card.naturalWidth > 0,
          opacity: Number(style.opacity),
          visible: Number(style.opacity) > 0 && style.visibility !== "hidden" && style.display !== "none"
            && bounds.width > 0 && bounds.height > 0 && bounds.right > clip.left && bounds.bottom > clip.top
            && bounds.left < clip.right && bounds.top < clip.bottom};
      });
      const hands = Array.from(document.querySelectorAll<HTMLElement>(".seat-dock, .observer-hand")).map(hand => ({
        seat: hand.dataset.seat,
        count: Number(hand.dataset.handCount),
        faceCardIds: Array.from(hand.querySelectorAll<HTMLElement>("[data-card-id]")).map(card => Number(card.dataset.cardId)),
      }));
      const hud = document.querySelector<HTMLElement>(".broadcast-hud");
      const displayedScore = [Number(hud?.dataset.scoreUs), Number(hud?.dataset.scoreThem)];
      setSnapshot(JSON.stringify({frame, completedPlayCount: Number(table.dataset.completedPlayCount), cards, hands, displayedScore}));
      continueRender(handle);
    }).catch(cancelRender);
  }, [frame]);
  return snapshot === null ? null : <Artifact filename="playback-dom.json" content={snapshot} />;
};

// Shared SVG patterns and the Sadu card back; all fills follow the configured palette.
const MajlisDefs: React.FC<{watermark: string}> = ({watermark}) => (
  <svg width="0" height="0" style={{position: "absolute"}} aria-hidden="true">
    <defs>
      <pattern id="mj-tri" width="40" height="40" patternUnits="userSpaceOnUse"><rect width="40" height="40" style={fill("ink")} /><polygon points="0,40 20,8 40,40" style={fill("accent")} /><polygon points="10,40 20,24 30,40" style={fill("cream")} /></pattern>
      <pattern id="mj-tris" width="16" height="16" patternUnits="userSpaceOnUse"><rect width="16" height="16" style={fill("ink")} /><polygon points="0,16 8,3 16,16" style={fill("rug")} /><polygon points="5,16 8,10 11,16" style={fill("cream")} /></pattern>
      <pattern id="mj-lat" width="56" height="56" patternUnits="userSpaceOnUse"><path d="M28 6 L50 28 L28 50 L6 28Z" fill="none" strokeWidth="2" style={stroke("rug_line")} /><circle cx="28" cy="28" r="3" style={fill("rug_deep")} /></pattern>
      <pattern id="mj-lats" width="14" height="14" patternUnits="userSpaceOnUse"><path d="M7 1 L13 7 L7 13 L1 7Z" fill="none" strokeWidth="1" style={stroke("rug_line")} /></pattern>
      <pattern id="mj-fringe" width="8" height="22" patternUnits="userSpaceOnUse"><rect x="3" width="2" height="22" style={fill("cream")} /></pattern>
      <pattern id="mj-najd" width="44" height="40" patternUnits="userSpaceOnUse"><rect width="44" height="40" style={fill("mud")} /><polygon points="8,34 22,10 36,34" style={fill("vent")} /></pattern>
      <pattern id="mj-merlon" width="44" height="16" patternUnits="userSpaceOnUse"><path d="M0 16 V10 H6 V4 H14 V0 H30 V4 H38 V10 H44 V16Z" style={fill("mud")} /></pattern>
      <pattern id="mj-frond" width="20" height="14" patternUnits="userSpaceOnUse"><rect width="20" height="14" style={fill("frond")} /><path d="M0 14 L10 0 M10 14 L20 0" strokeWidth="4" style={stroke("frond_light")} /><path d="M0 7 L5 0 M10 7 L15 0" strokeWidth="1" style={stroke("frond_dark")} /></pattern>
      <pattern id="mj-wm" width="460" height="170" patternUnits="userSpaceOnUse" patternTransform="rotate(-24)"><text x="230" y="95" textAnchor="middle" direction="rtl" opacity=".16" fontSize="34" fontWeight="700" style={fill("white")}>{watermark}</text></pattern>
      <radialGradient id="mj-lglow"><stop offset="0" style={{stopColor: paint("lantern_glow"), stopOpacity: 0.35}} /><stop offset="1" style={{stopColor: paint("lantern_glow"), stopOpacity: 0}} /></radialGradient>
      <radialGradient id="mj-vig" cx="50%" cy="50%" r="60%"><stop offset="55%" stopColor="black" stopOpacity="0" /><stop offset="100%" stopColor="black" stopOpacity=".38" /></radialGradient>
      <symbol id="mj-cb" viewBox="0 0 84 118">
        <rect width="84" height="118" rx="8" style={fill("sand")} />
        <rect x="5" y="5" width="74" height="108" rx="5" style={fill("ink")} />
        <rect x="8" y="8" width="68" height="102" rx="3" style={fill("accent")} />
        <rect x="11" y="11" width="62" height="96" rx="2" style={fill("rug")} />
        <rect x="11" y="11" width="62" height="96" fill="url(#mj-lats)" />
        <path d="M42 30 L62 59 L42 88 L22 59Z" style={fill("sand")} />
        <path d="M42 40 L55 59 L42 78 L29 59Z" style={fill("ink")} />
        <path d="M42 50 L48 59 L42 68 L36 59Z" style={fill("team_us")} />
      </symbol>
    </defs>
  </svg>
);

const Lantern: React.FC<{x: number}> = ({x}) => (
  <g transform={`translate(${x} 0)`}>
    <circle cx="0" cy="104" r="70" fill="url(#mj-lglow)" />
    <line x1="0" y1="50" x2="0" y2="72" strokeWidth="2" style={stroke("brass")} />
    <path d="M-12 82 L0 70 L12 82Z" style={fill("brass")} />
    <rect x="-15" y="82" width="30" height="6" rx="2" style={fill("lantern_metal")} />
    <path d="M-13 88 H13 V124 H-13Z" style={fill("lantern_body")} />
    <path d="M-9 124 V100 Q0 88 9 100 V124Z" style={fill("lantern_glow")} />
    <path d="M-5 124 V104 Q0 97 5 104 V124Z" style={fill("lantern_core")} />
    <rect x="-16" y="124" width="32" height="6" rx="2" style={fill("lantern_metal")} />
    <path d="M-8 130 L0 138 L8 130Z" style={fill("brass")} />
  </g>
);

// Top 0-176: palm-frond ceiling, beam ends, stepped mud wall with vents, two lanterns.
const MajlisCeiling: React.FC = () => (
  <svg className="decor-ceiling" width="1080" height="176" viewBox="0 0 1080 176" aria-hidden="true">
    <rect width="1080" height="176" opacity=".55" style={fill("ceiling")} />
    <rect width="1080" height="30" fill="url(#mj-frond)" />
    <rect y="30" width="1080" height="8" style={fill("hud_inner")} />
    {Array.from({length: 11}, (_, index) => <rect key={index} x={28 + index * 96} y="36" width="34" height="14" rx="4" style={fill("mud")} />)}
    <rect y="96" width="1080" height="16" fill="url(#mj-merlon)" opacity=".8" />
    <rect y="112" width="1080" height="40" fill="url(#mj-najd)" opacity=".8" />
    <rect y="154" width="1080" height="3" opacity=".35" style={fill("brass")} />
    <Lantern x={150} />
    <Lantern x={930} />
    <rect y="162" width="1080" height="1" opacity=".2" style={fill("brass")} />
  </svg>
);

const Finjan: React.FC<{x: number; y: number}> = ({x, y}) => (
  <g transform={`translate(${x} ${y})`}>
    <path d="M-17 0 H17 L12 22 H-12Z" style={fill("sand")} />
    <rect x="-17" y="0" width="34" height="4" style={fill("brass")} />
    <ellipse cx="0" cy="2" rx="13" ry="2" style={fill("coffee")} />
  </g>
);

// Floor 1630-1920: Sadu runner, woven mat, incense burner, dallah tray with finjans, dates.
const MajlisFloor: React.FC = () => (
  <>
    <div className="floor" />
    <svg className="decor-runner" width="1080" height="16" aria-hidden="true"><rect width="1080" height="16" fill="url(#mj-tris)" /></svg>
    <svg className="decor-mat" width="1080" height="274" viewBox="0 0 1080 274" aria-hidden="true"><rect width="1080" height="274" fill="url(#mj-lat)" opacity=".1" /><rect y="256" width="1080" height="18" fill="url(#mj-tris)" opacity=".5" /></svg>
    <svg className="decor-coffee" width="1000" height="250" viewBox="0 0 1000 250" aria-hidden="true">
      <g transform="translate(-50 0)">
        <ellipse cx="230" cy="204" rx="70" ry="12" fill="black" opacity=".35" />
        <path d="M226 104 C210 84 240 70 224 50 C212 36 232 24 226 10" fill="none" strokeWidth="4" strokeLinecap="round" opacity=".28" style={stroke("smoke")} />
        <path d="M238 104 C252 86 228 72 244 54 C254 42 238 30 246 18" fill="none" strokeWidth="3" strokeLinecap="round" opacity=".2" style={stroke("smoke")} />
        <rect x="196" y="172" width="68" height="26" rx="3" style={fill("mud")} /><rect x="196" y="172" width="68" height="4" style={fill("brass")} />
        <path d="M204 172 L210 128 H250 L256 172Z" style={fill("burner")} /><rect x="207" y="148" width="46" height="4" style={fill("brass")} />
        <path d="M194 128 H266 L254 110 H206Z" style={fill("brass")} /><ellipse cx="230" cy="110" rx="24" ry="5" style={fill("team_us")} />
      </g>
      <g transform="translate(200 0)">
        <ellipse cx="300" cy="210" rx="250" ry="34" fill="black" opacity=".35" />
        <ellipse cx="300" cy="202" rx="250" ry="32" style={fill("tray_rim")} />
        <ellipse cx="300" cy="198" rx="236" ry="26" style={fill("tray")} />
        <ellipse cx="230" cy="200" rx="44" ry="9" style={fill("tray_shadow")} />
        <path d="M248 100 C290 104 292 156 262 182 L256 175 C280 152 276 114 246 111Z" style={fill("lantern_metal")} />
        <path d="M216 116 C190 106 180 86 168 70 L158 64 L165 79 C176 98 188 118 210 130Z" style={fill("brass")} />
        <path d="M196 198 C186 170 192 142 214 132 L246 132 C268 142 274 170 264 198Z" style={fill("brass")} />
        <path d="M205 184 C200 168 204 152 214 144" fill="none" strokeWidth="3" opacity=".4" style={stroke("parchment")} />
        <rect x="210" y="125" width="40" height="8" rx="2" style={fill("tray_rim")} />
        <path d="M214 126 L220 96 L240 96 L246 126Z" style={fill("brass")} />
        <path d="M208 97 C212 74 248 74 252 97Z" style={fill("brass_light")} />
        <rect x="228" y="66" width="4" height="10" style={fill("brass_light")} /><circle cx="230" cy="64" r="6" style={fill("brass_light")} />
        <Finjan x={330} y={172} /><Finjan x={376} y={178} /><Finjan x={422} y={172} /><Finjan x={468} y={178} />
      </g>
      <g transform="translate(440 0)">
        <ellipse cx="380" cy="204" rx="96" ry="14" fill="black" opacity=".35" />
        <path d="M300 168 Q380 236 460 168Z" style={fill("brass")} /><ellipse cx="380" cy="168" rx="80" ry="14" style={fill("lantern_metal")} />
        <ellipse cx="350" cy="162" rx="14" ry="8" style={fill("date_dark")} /><ellipse cx="378" cy="158" rx="14" ry="8" style={fill("date")} /><ellipse cx="406" cy="163" rx="14" ry="8" style={fill("date_dark")} /><ellipse cx="364" cy="152" rx="13" ry="7" style={fill("date_light")} /><ellipse cx="392" cy="150" rx="13" ry="7" style={fill("date_dark")} />
      </g>
    </svg>
  </>
);

// Rug x 150-930, y 540-1230: fringe, triangle band, stripes, lattice field, vignette.
const MajlisRug: React.FC = () => (
  <svg className="rug" width="780" height="690" viewBox="0 0 780 690" aria-hidden="true">
    <rect x="24" y="0" width="732" height="22" fill="url(#mj-fringe)" /><rect x="24" y="668" width="732" height="22" fill="url(#mj-fringe)" />
    <rect x="0" y="20" width="780" height="650" rx="6" style={fill("ink")} />
    <rect x="10" y="30" width="760" height="630" style={fill("cream")} />
    <rect x="16" y="36" width="748" height="618" fill="url(#mj-tri)" />
    <rect x="56" y="76" width="668" height="538" style={fill("cream")} />
    <rect x="62" y="82" width="656" height="526" style={fill("ink")} />
    <rect x="70" y="90" width="640" height="510" style={fill("accent")} />
    <rect x="76" y="96" width="628" height="498" style={fill("ink")} />
    <rect x="82" y="102" width="616" height="486" style={fill("rug")} />
    <rect x="82" y="102" width="616" height="486" fill="url(#mj-lat)" />
    <path d="M390 230 L540 345 L390 460 L240 345Z" fill="none" strokeWidth="3" opacity=".55" style={stroke("rug_line")} />
    <path d="M390 270 L488 345 L390 420 L292 345Z" opacity=".6" style={fill("rug_deep")} />
    <rect x="0" y="20" width="780" height="650" fill="url(#mj-vig)" />
  </svg>
);

const BackStack: React.FC<{count: number; seat: Exclude<Seat, "south">}> = ({count, seat}) => {
  const visible = count <= 0 ? 0 : Math.min(6, count);
  return (
    <div className={`back-stack fan backs-${seat}`} aria-hidden="true">
      {Array.from({length: visible}, (_, index) => {
        const middle = (visible - 1) / 2;
        const offset = index - middle;
        return (
          <svg key={index} className="back" width="84" height="118" viewBox="0 0 84 118"
            style={{transform: `translateX(${offset * 18}px) rotate(${offset * 2.5}deg)`, zIndex: index + 1}}>
            <use href="#mj-cb" />
          </svg>
        );
      })}
    </div>
  );
};

const SeatPill: React.FC<{count: number; name: string; active: boolean}> = ({count, name, active}) => (
  <div className={`seat ${active ? "active" : ""}`}><span className="count">{count}</span><span className="name">{name}</span></div>
);

const SeatDock: React.FC<{
  actualSeat: Seat;
  seat: Exclude<Seat, "south">;
  count: number;
  active: boolean;
  name: string;
}> = ({actualSeat, seat, count, active, name}) => (
  <section
    className={`seat-dock dock-${seat} ${active ? "is-active" : ""}`}
    data-seat={actualSeat}
    data-hand-count={count}
    data-card-back-count={Math.min(6, Math.max(0, count))}
  >
    <BackStack count={count} seat={seat} />
    <div className={`seat-row row-${seat}`}><SeatPill count={count} name={name} active={active} /></div>
  </section>
);

const SeatTags: React.FC<{
  seat: Seat;
  dealer: boolean;
  buyer: boolean;
  dealerLabel: string;
  buyerText: string;
  purchaseLabel: string;
  purchaseCard?: CardValue;
}> = ({seat, dealer, buyer, dealerLabel, buyerText, purchaseLabel, purchaseCard}) => (dealer || buyer ? (
  <div className={`seat-tags tags-${seat}`}>
    {buyer ? <span className="tag buyer" aria-label={purchaseLabel}><span>{buyerText}</span>{purchaseCard ? <Card size="small" card={purchaseCard} className="pcard" /> : null}</span> : null}
    {dealer ? <span className="tag dealer"><span>◆</span><span>{dealerLabel}</span></span> : null}
  </div>
) : null);

const TeamScore: React.FC<{team: "us" | "them"; label: string; score: number; target: number}> = ({team, label, score, target}) => (
  <div className={`team team-${team}`}>
    <div className="team-head"><span className="team-label"><span className="dot" />{label}</span><span className="team-score">{score}</span></div>
    <div className="meter"><i style={{width: `${clamp(score / target, 0, 1) * 100}%`}} /></div>
  </div>
);

const CallBubble: React.FC<{seat: Seat; kind: string; placement: "setup" | "play"; opacity: number; children: React.ReactNode}> = ({seat, kind, placement, opacity, children}) => (
  <div className={`call call-${kind} call-at-${seat} call-${placement}`} style={{opacity}}>{children}</div>
);

const SunIcon: React.FC<{label: string}> = ({label}) => (
  <svg width="28" height="28" viewBox="0 0 40 40" role="img" aria-label={label}>
    <circle cx="20" cy="20" r="8" style={fill("accent")} />
    <g strokeWidth="3" strokeLinecap="round" style={stroke("accent")}><path d="M20 3v6M20 31v6M3 20h6M31 20h6M8 8l4 4M28 28l4 4M8 32l4-4M28 12l4-4" /></g>
  </svg>
);

const CupIcon: React.FC = () => (
  <svg width="30" height="30" viewBox="0 0 30 30" aria-hidden="true"><path d="M6 9 H24 L21 24 H9Z" style={fill("escalation_red")} /><rect x="6" y="8" width="18" height="3" style={fill("team_us")} /></svg>
);

const destinationTransform: Record<Seat, string> = {
  north: "translate(-50%, -92%) rotate(-2deg)",
  west: "translate(-119%, -49%) rotate(-7deg)",
  east: "translate(19%, -49%) rotate(7deg)",
  south: "translate(-50%, -7%) rotate(1deg)",
};
const entryOffset: Record<Seat, [number, number]> = {
  north: [0, -430],
  west: [-440, 0],
  east: [440, 0],
  south: [0, 560],
};
const collectionOffset: Record<Seat, [number, number]> = {
  north: [0, -650],
  west: [-650, 0],
  east: [650, 0],
  south: [0, 720],
};

export const ProfessionalPrototypeFrame: React.FC<PrototypeProps & {compositionFrame: number}> = ({
  bundle,
  compositionFrame,
  stage,
  reducedMotion = false,
  storyFrame,
  rewinding = false,
  capturePlaybackDom = false,
}) => {
  const frame = storyFrame ?? compositionFrame;
  const localized = (key: string): string => {
    const value = bundle.localizations[key];
    if (typeof value !== "string" || !value.trim()) throw new Error(`Missing governed localization: ${key}`);
    return value;
  };
  const formatted = (key: string, values: Record<string, string | number>): string =>
    localized(key).replace(/\{([a-zA-Z0-9_]+)\}/g, (match, name) =>
      Object.prototype.hasOwnProperty.call(values, name) ? String(values[name]) : match,
    );
  const questionScene = bundle.scenes.find((item) => item.stage === "question") ?? bundle.scenes[0];
  const revealScene = bundle.scenes.find((item) => item.stage === "reveal") ?? questionScene;
  const contextScene = bundle.scenes.find((item) => item.stage === "context");
  const continuationScene = bundle.scenes.find((item) => item.stage === "continuation");
  const scene = stage === "reveal" ? revealScene : questionScene;
  const observer = (seatRing.includes(scene.observer as Seat) ? scene.observer : "south") as Seat;
  const observerIndex = seatRing.indexOf(observer);
  const actualAtVisual = (visual: Seat): Seat =>
    seatRing[(seatRing.indexOf(visual) + observerIndex - 2 + seatRing.length) % seatRing.length];
  const visualOfActual = (actual: Seat): Seat =>
    seatRing[(seatRing.indexOf(actual) - observerIndex + 2 + seatRing.length) % seatRing.length];
  const observerTeamIndex = observer === "north" || observer === "south" ? 0 : 1;
  const relativePair = (values: number[]): number[] =>
    observerTeamIndex === 0 ? values : [values[1] ?? 0, values[0] ?? 0];

  const decisionTable = asRecord(sceneLayer(scene, "table")?.content);
  const decisionHand = asRecord(sceneLayer(scene, "observer-hand")?.content);
  const seatLayer = asRecord(sceneLayer(scene, "seats")?.content);
  const handCounts = asRecord(seatLayer.handCounts);
  const prompt = asRecord(sceneLayer(questionScene, "decision-prompt")?.content);
  const analysis = asRecord(sceneLayer(revealScene, "analysis")?.content);
  const legalChoices = Array.isArray(prompt.legalChoices)
    ? prompt.legalChoices.map(asRecord)
    : [];
  const decisionKind = String(analysis.decisionKind ?? "play");
  const terminalOutcomeAvailable = decisionKind === "play" || analysis.continuationComplete === true;
  const promptActingValue = String(prompt.actingSeat ?? decisionTable.currentPlayer ?? observer);
  const actingSeat = (seatRing.includes(promptActingValue as Seat) ? promptActingValue : observer) as Seat;
  const promptContract = String(prompt.promptContract ?? "choose_acting_action");
  const isActorChallenge = promptContract === "choose_acting_action";
  const playbackLifecycle = asRecord(sceneLayer(continuationScene, "public-lifecycle")?.content);
  const postReveal = bundle.timeline.segments.some((segment) => ["continuation", "outcome", "hold"].includes(segment.stage) && frame >= segment.startFrame);
  const lifecycleScene = postReveal ? continuationScene : contextScene;
  const publicLifecycle = asRecord(sceneLayer(lifecycleScene, "public-lifecycle")?.content);
  const observerLifecycle = asRecord(sceneLayer(lifecycleScene, "observer-lifecycle")?.content);
  const initialHand = asCards(observerLifecycle.initialHandCards);
  const initialFive = asCards(observerLifecycle.initialFiveCards);
  const bidHistory = Array.isArray(publicLifecycle.bidHistory) ? publicLifecycle.bidHistory : [];
  const escalationHistory = Array.isArray(publicLifecycle.escalationHistory) ? publicLifecycle.escalationHistory : [];
  // Decision clips show the dealer's round call before each recorded buying round
  // (the first-round call, then the second-round call only when the trace has a second round).
  const dealerRoundCalls = bundle.exportPreset === "decision_clip";
  const bidActions = bidHistory.flatMap((action, index): Record<string, unknown>[] => {
    const bid: Record<string, unknown> = {...asRecord(action), kind: "bid"};
    const round = Number(bid.round ?? 1);
    const opensRound = dealerRoundCalls && (index === 0 || Number(asRecord(bidHistory[index - 1]).round ?? 1) !== round);
    return opensRound ? [{kind: "round", seat: publicLifecycle.dealer, round}, bid] : [bid];
  });
  const purchaseActionIndex = bidActions.length;
  const setupActions = initialFive.length === 5 ? [
    ...bidActions,
    {kind: "purchase", seat: publicLifecycle.purchaseRecipient},
    ...escalationHistory.map((action) => ({...asRecord(action), kind: "escalation"})),
  ] : [];
  const allTricks = (Array.isArray(playbackLifecycle.completedTricks)
    ? playbackLifecycle.completedTricks
    : []) as PublicTrick[];
  const allPlays = Array.isArray(playbackLifecycle.plays)
    ? (playbackLifecycle.plays as PublicPlay[])
    : allTricks.flatMap((trick) => trick.plays);
  const projects = (Array.isArray(publicLifecycle.projects) ? publicLifecycle.projects : []) as PublicProject[];
  const roundScore = asRecord(publicLifecycle.roundScore);
  const cardGamePoints = relativePair(asNumbers(roundScore.cardGamePoints));
  const projectPoints = relativePair(asNumbers(roundScore.projectPoints));
  const balootPoints = relativePair(asNumbers(roundScore.balootPoints));
  const awardedPoints = relativePair(asNumbers(roundScore.awardedPoints));
  const timelineSegments = bundle.timeline.segments;
  const segmentByStage = (name: string) => timelineSegments.find((item) => item.stage === name);
  const activeTimelineSegment = timelineSegments.find(
    (item) => frame >= item.startFrame && frame < item.startFrame + item.durationFrames,
  );
  const replaySegment = segmentByStage("replay");
  const setupSegment = segmentByStage("setup");
  const continuationSegment = segmentByStage("continuation");

  const isFlagship = stage === "flagship";
  const isHookedFlagshipExport = bundle.exportPreset === "hooked_flagship";
  const isEnvironment = stage === "normal";
  let hand = isEnvironment && initialHand.length ? initialHand : asCards(decisionHand.cards);
  let plays = isEnvironment ? [] : ((asRecord(decisionTable.currentTrick).plays ?? []) as PublicPlay[]);
  let currentTrickCount = plays.length;
  let completedTrickIndex: number | null = null;
  let counts: Record<Seat, number> = {
    north: isEnvironment ? 8 : Number(handCounts.north ?? 0),
    west: isEnvironment ? 8 : Number(handCounts.west ?? 0),
    east: isEnvironment ? 8 : Number(handCounts.east ?? 0),
    south: isEnvironment ? 8 : Number(handCounts.south ?? 0),
  };
  counts[observer] = hand.length;
  let currentPlayer: Seat | null = String(isEnvironment ? publicLifecycle.firstLeader ?? "north" : decisionTable.currentPlayer ?? observer) as Seat;
  const dealer = String(publicLifecycle.dealer ?? decisionTable.dealer ?? "east") as Seat;
  let buyer: Seat | null = String(publicLifecycle.buyer ?? decisionTable.buyer ?? "west") as Seat;
  const contractName = String(publicLifecycle.contract ?? decisionTable.contract ?? "hokum");
  const trumpName = String(publicLifecycle.trumpSuit ?? decisionTable.trumpSuit ?? "spades");
  const preScore = relativePair(Array.isArray(publicLifecycle.preDecisionMatchScore)
    ? asNumbers(publicLifecycle.preDecisionMatchScore)
    : [0, 0]);
  const finalScore = Array.isArray(publicLifecycle.matchScore) ? relativePair(asNumbers(publicLifecycle.matchScore)) : preScore;
  let score = isEnvironment ? preScore : (Array.isArray(decisionTable.matchScore) ? relativePair(asNumbers(decisionTable.matchScore)) : preScore);
  let showPrompt = stage === "hook";
  let showReveal = stage === "reveal";
  let showOutcome = false;
  let suppressTerminalLifecycle = false;
  let sourcePlayCount = allPlays.length;
  let completedPlayCount = 0;
  let playProgress = 1;
  let lastPlayAge = 0;
  let playInterval = 0;
  let winner: Seat | null = null;
  let promptTimer = String(Math.max(1, Math.round(Number(asRecord(bundle.presentation.timeline).pause_seconds))));
  let promptFraction = 1;
  let setupAction: Record<string, unknown> | undefined;
  let setupActionIndex = -1;

  const pausePlayIndex = clamp(Number(publicLifecycle.pauseBeforePlayIndex ?? 0), 0, allPlays.length);
  const hasContextPlayback = timelineSegments.some((segment) => segment.stage === "context" && segment.durationFrames > 0);
  const replayTarget = Math.max(0, pausePlayIndex - Number(hasContextPlayback));
  const contextTarget = pausePlayIndex;
  const continuationCount = Math.max(0, allPlays.length - pausePlayIndex);
  // Decision clips that replay the round from the deal (ending on a hold, or on the round's final score).
  const replaysFromDeal = bundle.exportPreset === "decision_clip" && Boolean(replaySegment);
  const playsFullRound = isHookedFlagshipExport || replaysFromDeal;

  if (isFlagship) {
    const timelineStage = activeTimelineSegment?.stage ?? timelineSegments.at(-1)?.stage;
    const segmentStart = activeTimelineSegment?.startFrame ?? 0;
    const segmentDuration = activeTimelineSegment?.durationFrames ?? 1;
    const segmentFrame = Math.max(0, frame - segmentStart);
    const useDecisionProjection = decisionKind !== "play" && (
      bundle.exportPreset === "decision_clip" || [
        "hook", "rewind", "context", "question", "reveal",
      ].includes(String(timelineStage))
    );
    if (useDecisionProjection) {
      hand = asCards(decisionHand.cards);
      plays = ((asRecord(decisionTable.currentTrick).plays ?? []) as PublicPlay[]);
      currentTrickCount = plays.length;
      counts = {
        north: Number(handCounts.north ?? 0),
        west: Number(handCounts.west ?? 0),
        east: Number(handCounts.east ?? 0),
        south: Number(handCounts.south ?? 0),
      };
      counts[observer] = hand.length;
      currentPlayer = actingSeat;
      completedPlayCount = 0;
      sourcePlayCount = 0;
      suppressTerminalLifecycle = true;
      showPrompt = timelineStage === "question";
      showReveal = timelineStage === "reveal";
      score = Array.isArray(decisionTable.matchScore)
        ? relativePair(asNumbers(decisionTable.matchScore))
        : preScore;
      buyer = seatRing.includes(String(decisionTable.buyer) as Seat)
        ? String(decisionTable.buyer) as Seat
        : null;
    } else if (timelineStage === "hook" || timelineStage === "rewind") {
      const showOpeningState = isHookedFlagshipExport && initialHand.length > 0;
      hand = showOpeningState && setupSegment ? initialFive : showOpeningState ? initialHand : asCards(decisionHand.cards);
      plays = showOpeningState ? [] : ((asRecord(decisionTable.currentTrick).plays ?? []) as PublicPlay[]);
      counts = {
        north: showOpeningState ? hand.length : Number(handCounts.north ?? 2),
        west: showOpeningState ? hand.length : Number(handCounts.west ?? 2),
        east: showOpeningState ? hand.length : Number(handCounts.east ?? 2),
        south: showOpeningState ? hand.length : Number(handCounts.south ?? 2),
      };
      counts[observer] = hand.length;
      currentPlayer = showOpeningState && setupSegment
        ? String(asRecord(bidHistory[0]).seat ?? publicLifecycle.firstLeader ?? "north") as Seat
        : showOpeningState ? String(publicLifecycle.firstLeader ?? "north") as Seat : actingSeat;
      if (showOpeningState && setupSegment) buyer = null;
      showPrompt = !showOpeningState;
      if (!showOpeningState) completedPlayCount = pausePlayIndex;
      promptTimer = localized("video.decision.timer_open");
      score = preScore;
    } else {
      if (timelineStage === "setup") {
        setupActionIndex = Math.min(
          setupActions.length - 1,
          Math.floor(segmentFrame * setupActions.length / segmentDuration),
        );
        setupAction = setupActions[setupActionIndex];
        completedPlayCount = 0;
        suppressTerminalLifecycle = true;
        score = preScore;
      } else if (timelineStage === "replay") {
        playInterval = segmentDuration / Math.max(1, replayTarget);
        completedPlayCount = Math.min(replayTarget, Math.floor((segmentFrame * Math.max(1, replayTarget)) / segmentDuration) + 1);
        const lastStart = segmentStart + (completedPlayCount - 1) * playInterval;
        lastPlayAge = frame - lastStart;
      } else if (timelineStage === "context") {
        const contextCount = Math.max(1, contextTarget - replayTarget);
        playInterval = segmentDuration / contextCount;
        completedPlayCount = Math.min(contextTarget, replayTarget + Math.floor((segmentFrame * contextCount) / segmentDuration) + 1);
        const lastStart = segmentStart + Math.max(0, completedPlayCount - replayTarget - 1) * playInterval;
        lastPlayAge = frame - lastStart;
      } else if (timelineStage === "question") {
        completedPlayCount = pausePlayIndex;
        showPrompt = true;
        currentPlayer = actingSeat;
        promptTimer = String(Math.max(1, Math.ceil((segmentDuration - segmentFrame) / bundle.render.fps)));
        promptFraction = clamp((segmentDuration - segmentFrame) / segmentDuration, 0, 1);
      } else if (timelineStage === "reveal") {
        completedPlayCount = pausePlayIndex;
        showReveal = true;
        currentPlayer = actingSeat;
      } else if (timelineStage === "continuation") {
        playInterval = segmentDuration / Math.max(1, continuationCount);
        completedPlayCount = Math.min(allPlays.length, pausePlayIndex + Math.floor((segmentFrame * Math.max(1, continuationCount)) / segmentDuration) + 1);
        const lastStart = segmentStart + Math.max(0, completedPlayCount - pausePlayIndex - 1) * playInterval;
        lastPlayAge = frame - lastStart;
      } else if (timelineStage === "hold" && replaysFromDeal) {
        // Replay clips end on the recorded answer card landing; nothing after it is shown.
        completedPlayCount = Math.min(allPlays.length, pausePlayIndex + 1);
        playInterval = segmentDuration;
        lastPlayAge = segmentFrame;
      } else if (timelineStage === "hold") {
        completedPlayCount = allPlays.length;
        if (timelineSegments.some(segment => segment.stage === "outcome")) {
          showOutcome = true;
          score = finalScore;
          currentPlayer = null;
        }
      } else {
        completedPlayCount = allPlays.length;
        showOutcome = true;
        score = finalScore;
        currentPlayer = null;
      }

      const playback = derivePlaybackState({
        initialHand,
        allPlays,
        allTricks,
        observer,
        completedPlayCount,
      });
      plays = showPrompt || showReveal ? playback.currentTrick : playback.tablePlays;
      currentTrickCount = playback.currentTrick.length;
      completedTrickIndex = playback.completedTrickIndex;
      hand = playback.hand;
      counts = playback.handCounts;
      currentPlayer = playback.currentPlayer;
      winner = playback.completedTrickWinner;
      if (showPrompt || showReveal) currentPlayer = actingSeat;
      if (showOutcome) currentPlayer = null;
      if (replaysFromDeal && timelineStage === "hold") {
        winner = null;
        currentPlayer = null;
      }
      if (showOutcome) plays = [];
      if (setupAction) {
        const purchased = setupActionIndex >= purchaseActionIndex;
        hand = purchased ? initialHand : initialFive;
        counts = {north: purchased ? 8 : 5, west: purchased ? 8 : 5, south: purchased ? 8 : 5, east: purchased ? 8 : 5};
        counts[observer] = hand.length;
        plays = [];
        currentTrickCount = 0;
        currentPlayer = String(setupAction.seat ?? publicLifecycle.firstLeader ?? "north") as Seat;
        buyer = purchased ? String(publicLifecycle.buyer ?? "north") as Seat : null;
      }
    }
  }

  const trump = trumpName ? suitSymbol[trumpName] ?? trumpName : "";
  const questionKey = String(prompt.textKey ?? "video.decision.choose_best_move");
  const selectedActionCode = typeof analysis.selectedAction === "string" ? analysis.selectedAction : "";
  const recommendedFromAnalysis = asRecord(analysis.recommendedAction).card as CardValue | undefined;
  const recommended = recommendedFromAnalysis ?? [...asCards(decisionHand.cards), ...initialHand]
    .find((card) => canonicalCardCode(card) === selectedActionCode);
  const isUniqueExactWinner = isActorChallenge &&
    (analysis.utility === "round_win" || analysis.utility === "contract_success") &&
    analysis.achievedMode === "exact" &&
    analysis.completionStatus === "exact_complete" &&
    analysis.claimSatisfied === true &&
    Number(analysis.winningRootActionCount) === 1;
  const analysisReasonKey = String(analysis.reasonKey ?? (isActorChallenge ? "video.analysis.expert_reason" : "video.observer.insight"));
  const reasonKey = isUniqueExactWinner && analysisReasonKey === "video.analysis.one_winning_action"
    ? "video.analysis.only_winning_move"
    : analysisReasonKey;
  const reason = reasonKey === "video.analysis.only_winning_move" && recommended
    ? formatted(reasonKey, {best: cardLabel(recommended)})
    : localized(reasonKey === "video.analysis.only_winning_move"
      ? "video.analysis.one_winning_action"
      : reasonKey);
  const actionLabel = (action: Record<string, unknown>): string => {
    const actionName = String(action.action ?? "");
    if (decisionKind === "bid") return localized(`video.bid.${actionName}`);
    if (decisionKind === "project") {
      return localized(`video.project.${String(action.kind ?? "")}`);
    }
    if (decisionKind === "escalation") {
      const label = localized(`video.escalation.${actionName}`);
      const mode = String(action.mode ?? "");
      return mode
        ? `${label} · ${localized(`video.escalation.mode.${mode}`)}`
        : label;
    }
    throw new Error("unsupported governed decision action");
  };
  const proofScopeKey = typeof analysis.proofScopeKey === "string" ? analysis.proofScopeKey : null;
  const sunTieKey = typeof analysis.sunTieOutcome === "string" ? `video.scoring.sun_tie.${analysis.sunTieOutcome}` : null;
  const playerNames = bundle.identity.players;
  const purchaseBid = asRecord([...bidHistory].reverse().find((bid) => asRecord(bid).action !== "pass"));
  const bidAction = String(purchaseBid.action ?? contractName);
  const bidRound = Number(purchaseBid.round ?? 1);
  const bidType = localized(`video.bid.${bidAction}`);
  const roundLabel = localized(`video.bid.round.${bidRound}`);
  const purchaseLabel = formatted(
    bidAction === "hokum" ? "video.purchase.hokum_summary" : "video.purchase.plain_summary",
    {contract: bidType, round: roundLabel, trump},
  );
  const presentation = asRecord(bundle.presentation.presentation);
  // Production can hide reason and proof text independently of the reveal.
  const showJustification = presentation.show_justification_text !== false;
  const layout = asRecord(bundle.presentation.layout);
  const timelineProfile = asRecord(bundle.presentation.timeline);
  const motion = asRecord(bundle.presentation.motion);
  const auditMedia = bundle.source !== undefined && !isExactQualifiedPublishableSource(bundle.source);
  const showIdentity = Boolean(presentation.show_identity_pill) || auditMedia;
  const qrActionable = bundle.source === undefined || bundle.source.qrActionable === true;
  const showQr = Boolean(presentation.show_enrollment_qr) && !auditMedia && qrActionable;
  const publicDecisionReference = bundle.source?.sourceSchemaVersion === 3
    ? `${bundle.source.decisionReference.resultCode}:${bundle.source.decisionReference.decisionOrdinal}`
    : undefined;
  const legacyExactStateId = bundle.source?.sourceSchemaVersion === 3
    ? undefined
    : bundle.source?.exactStateId ?? bundle.identity.sceneId;
  const qrSize = Number(presentation.qr_size_px ?? 72);
  const qrMargin = Number(presentation.qr_quiet_zone_modules ?? 4);
  const {flightFrames, collectionFrames, collectionStart} = playbackMotionTiming(
    playInterval, Number(motion.flight_steps ?? 21), Number(motion.collection_steps ?? 22),
    Number(motion.transition_steps ?? 4),
  );
  if (playInterval > 0) playProgress = reducedMotion || flightFrames === 0 ? 1 : clamp(lastPlayAge / flightFrames, 0, 1);
  const collectionProgress = winner && !showOutcome
    ? collectionFrames === 0
      ? Number(lastPlayAge >= collectionStart)
      : clamp((lastPlayAge - collectionStart) / collectionFrames, 0, 1)
    : 0;
  const collectionEase = Easing.bezier(0.4, 0, 0.2, 1)(collectionProgress);
  const tableOpacity = collectionOpacity(collectionProgress);
  const showProjects = Boolean(layout.show_projects ?? true);
  const projectAnnouncementTrick = Number(layout.project_announcement_trick ?? 1);
  const projectRevealTrick = Number(layout.project_reveal_trick ?? 2);
  const persistRevealedProjects = Boolean(layout.persist_revealed_projects ?? true);
  const showScoreBreakdown = Boolean(layout.show_score_breakdown ?? true);
  const openingPlaySegment = replaySegment ?? (decisionKind !== "play" ? continuationSegment : undefined);
  const openingPlayCount = decisionKind === "play" ? replayTarget : allPlays.length;
  const openingPlayInterval = (openingPlaySegment?.durationFrames ?? 2772) / Math.max(1, openingPlayCount);
  const openingPlayStart = openingPlaySegment?.startFrame ?? 108;
  const announcementFrames = Number(timelineProfile.project_call_seconds ?? 1.6) * bundle.render.fps;
  const projectRevealFrames = Number(timelineProfile.project_reveal_seconds ?? 2.3) * bundle.render.fps;
  const projectTurnStart = (project: PublicProject, trickNumber: number): number => {
    const playIndex = allPlays.findIndex((play) =>
      Number(play.trickIndex ?? 0) === Math.max(0, trickNumber - 1) && play.seat === project.owner
    );
    return playIndex < 0 ? Number.POSITIVE_INFINITY : openingPlayStart + playIndex * openingPlayInterval;
  };
  const showProjectAnnouncement = (project: PublicProject): boolean => {
    const turnStart = projectTurnStart(project, projectAnnouncementTrick);
    return showProjects && !suppressTerminalLifecycle && isFlagship && playsFullRound && frame >= turnStart && frame < turnStart + announcementFrames;
  };
  const showProjectReveal = (project: PublicProject): boolean => {
    const turnStart = projectTurnStart(project, projectRevealTrick);
    return showProjects && !suppressTerminalLifecycle && project.revealed && asCards(project.cards).length > 0 && isFlagship && playsFullRound && frame >= turnStart && frame < turnStart + projectRevealFrames;
  };
  const showProjectBadge = (project: PublicProject): boolean => {
    const revealEnd = projectTurnStart(project, projectRevealTrick) + projectRevealFrames;
    return showProjects && !suppressTerminalLifecycle && project.revealed && asCards(project.cards).length > 0 && persistRevealedProjects && !showOutcome && (
      (isFlagship && playsFullRound && (frame < openingPlayStart || frame >= revealEnd)) ||
      (!isFlagship && !isEnvironment)
    );
  };
  const projectLabel = (project: PublicProject): string =>
    localized(`video.project.${project.kind}`);
  const demoMotionFrame = frame % 120;
  const demoMotionProgress = reducedMotion ? 1 : interpolate(demoMotionFrame, [0, 21], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.22, 1, 0.36, 1),
  });
  const demoMotionCard = !isFlagship ? hand[0] : undefined;


  const majlis = asRecord(bundle.presentation.majlis);
  const palette = asRecord(majlis.palette);
  const paletteStyle = majlisPaletteStyle(palette);
  const hudStyle = governedChoice(majlis.hud, ["plaque", "slim", "cushions"] as const, "hud");
  const idBox = governedChoice(majlis.id_box, ["qr", "number"] as const, "id_box");
  // Publishable research notice: the design's red header band and a red code box, without audit marks.
  const researchNotice = majlis.research_notice === true && !auditMedia;
  const scoreTarget = Number(majlis.score_target);
  if (!(scoreTarget > 0)) throw new Error("Missing or invalid governed majlis score_target");
  const social = asRecord(majlis.social);
  const callouts = asRecord(bundle.presentation.callouts);
  const fps = bundle.render.fps;
  const fadeInFrames = Math.max(1, (Number(callouts.fade_in_ms ?? 110) / 1000) * fps);
  const fadeOutFrames = Math.max(1, (Number(callouts.fade_out_ms ?? 130) / 1000) * fps);
  const fade = (age: number, remaining: number): number =>
    Math.min(clamp(age / fadeInFrames, 0, 1), clamp(remaining / fadeOutFrames, 0, 1));

  // The answer hold keeps the decided trick on the table, so count that trick, not the next.
  const trickNumber = replaysFromDeal && activeTimelineSegment?.stage === "hold"
    ? clamp(Math.ceil(completedPlayCount / 4), 1, 8)
    : clamp(9 - Math.max(...seatRing.map((seat) => counts[seat])), 1, 8);
  const visibleSetupActions: Record<string, unknown>[] = setupAction ? setupActions.slice(0, setupActionIndex + 1) : [];
  const biddingOpen = Boolean(setupAction) && setupActionIndex < purchaseActionIndex;
  const biddingRound = Number(asRecord([...visibleSetupActions].reverse().find((action) => action.kind === "bid")).round ?? 1);
  const contractPending = buyer === null;
  const purchaseCard = publicLifecycle.purchaseCard ? publicLifecycle.purchaseCard as CardValue : undefined;
  const escalationMultiplier: Record<string, number> = {double: 2, triple: 3, four: 4};
  const shownEscalations = contractPending
    ? []
    : setupAction
      ? visibleSetupActions.filter((action) => action.kind === "escalation")
      : escalationHistory.map(asRecord);
  const multiplier = Math.max(1, ...shownEscalations.map((action) => escalationMultiplier[String(action.action)] ?? 1));
  const idCode = bundle.identity.gameSerial.split("-").at(-1) || bundle.identity.gameSerial;
  // Slim box is 76px: fill it with a 2-module margin (~2.3 px/module) instead of a 60px QR.
  const qrPixels = hudStyle === "slim" ? 76 : qrSize;
  const qrMarginModules = hudStyle === "slim" ? Math.min(qrMargin, 2) : qrMargin;

  const setupStart = setupSegment?.startFrame ?? 0;
  const setupDuration = setupSegment?.durationFrames ?? 1;
  const actionFrames = setupDuration / Math.max(1, setupActions.length);
  const currentActionAge = frame - setupStart - setupActionIndex * actionFrames;
  const setupRemaining = setupStart + setupDuration - frame;
  const latestCalls = new Map<Seat, {action: Record<string, unknown>; current: boolean}>();
  visibleSetupActions.forEach((action, index) => {
    const seat = String(action.seat) as Seat;
    if ((action.kind === "bid" || action.kind === "escalation" || action.kind === "round") && seatRing.includes(seat)) {
      latestCalls.set(seat, {action, current: index === setupActionIndex});
    }
  });
  const callFor = (action: Record<string, unknown>, current: boolean): {kind: string; body: React.ReactNode} => {
    const name = String(action.action ?? "");
    if (action.kind === "round") {
      return {kind: current ? "hot" : "bid", body: <b>{localized(`video.bid.round.${Number(action.round)}`)}</b>};
    }
    if (action.kind === "escalation") {
      if (name === "decline") return {kind: "pass", body: localized("video.escalation.decline")};
      const mode = String(action.mode ?? "");
      return {kind: "dbl", body: <>
        <b>{localized(`video.escalation.${name}`)}</b>
        {mode
          ? <small>{localized(`video.escalation.mode.${mode}`)}</small>
          : escalationMultiplier[name]
            ? <span className="x" dir="ltr">{`×${escalationMultiplier[name]}`}</span>
            : name === "gahwa" ? <CupIcon /> : null}
      </>};
    }
    if (name === "pass") return {kind: "pass", body: localized("video.bid.pass")};
    const suit = String(action.trumpSuit ?? action.suit ?? "");
    return {kind: current ? "hot" : "bid", body: <>
      <b>{localized(`video.bid.${name}`)}</b>
      {name === "sun"
        ? <span className="chip"><SunIcon label={localized("video.bid.sun")} /></span>
        : suitSymbol[suit] ? <span className={`chip ${redSuit(suit) ? "red" : ""}`}>{suitSymbol[suit]}</span> : null}
      {Number(action.round) === 2 ? <small>{localized("video.bid.round.2")}</small> : null}
    </>};
  };

  const lastTablePlay = plays.at(-1) as (PublicPlay & {baloot?: boolean}) | undefined;
  const balootFrames = Number(timelineProfile.public_callout_seconds ?? 1.65) * fps;
  const showBaloot = isFlagship && !showOutcome && playInterval > 0 && lastTablePlay?.baloot === true &&
    lastPlayAge < balootFrames && Boolean(suitSymbol[trumpName]);
  const balootSuit = trumpName as CardValue["suit"];
  const researchShort = localized("video.governance.research_short");
  const notPublishable = localized("video.governance.not_publishable");
  const positive = (value: number): string | undefined => (value > 0 ? "is-plus" : undefined);
  type Placed = {visual: Seat; node: React.ReactNode};
  const northActual = actualAtVisual("north");
  const setupCalls: Placed[] = [...latestCalls.entries()].map(([seat, call]) => {
    const {kind, body} = callFor(call.action, call.current);
    const visual = visualOfActual(seat);
    return {visual, node: <CallBubble key={`setup-${seat}`} seat={visual} kind={kind} placement="setup" opacity={fade(call.current ? currentActionAge : fadeInFrames, setupRemaining)}>{body}</CallBubble>};
  });
  const playCallouts: Placed[] = projects.flatMap((project, index) => {
    const visual = visualOfActual(project.owner);
    const key = `${project.owner}-${project.kind}-${index}`;
    const announcementAge = frame - projectTurnStart(project, projectAnnouncementTrick);
    const revealAge = frame - projectTurnStart(project, projectRevealTrick);
    const cards = asCards(project.cards);
    const placed: Placed[] = [];
    if (showProjectAnnouncement(project)) placed.push({visual, node: (
      <CallBubble key={`call-${key}`} seat={visual} kind="proj" placement="play" opacity={fade(announcementAge, announcementFrames - announcementAge)}>
        <small>{localized("video.project.call_label")}</small><b>{projectLabel(project)}</b>
      </CallBubble>
    )});
    if (showProjectReveal(project)) placed.push({visual, node: (
      <div key={`tray-${key}`} className={`ptray tray-at-${visual} ${cards.length > 3 ? "is-dense" : ""}`} style={{opacity: fade(revealAge, projectRevealFrames - revealAge)}}>
        <span className="ptray-label"><small>{localized("video.project.call_label")}</small><b>{projectLabel(project)}</b></span>
        <span className="ptray-cards">{cards.map((card) => <Card key={card.id} size="small" card={card} className="tray-card" />)}</span>
      </div>
    )});
    if (showProjectBadge(project)) placed.push({visual, node: (
      <div key={`badge-${key}`} className={`project-badge tray-at-${visual}`}>
        <span>{projectLabel(project)}</span>
        <b className={cards[0] && redSuit(cards[0].suit) ? "red" : undefined}>{cards[0] ? suitSymbol[cards[0].suit] : ""}</b>
      </div>
    )});
    return placed;
  });
  if (showBaloot && lastTablePlay) {
    const visual = visualOfActual(lastTablePlay.seat);
    playCallouts.push({visual, node: (
      <CallBubble key="baloot" seat={visual} kind="baloot" placement="play" opacity={fade(lastPlayAge, balootFrames - lastPlayAge)}>
        <b>{localized("video.call.baloot")}</b>
        <span className="minis"><Card size="small" card={{rank: "king", suit: balootSuit}} className="pcard" /><Card size="small" card={{rank: "queen", suit: balootSuit}} className="pcard" /></span>
      </CallBubble>
    )});
  }
  const tagsFor = (visualSeat: Seat): React.ReactNode => {
    const actualSeat = actualAtVisual(visualSeat);
    return <SeatTags seat={visualSeat} dealer={dealer === actualSeat} buyer={buyer === actualSeat} dealerLabel={localized("video.role.dealer")} buyerText={`${localized("video.role.buyer")} · ${bidType}`} purchaseLabel={purchaseLabel} purchaseCard={purchaseCard} />;
  };
  if (bundle.source !== undefined && isExactQualifiedPublishableSource(bundle.source) &&
    [social.tiktok, social.instagram].some((handle) => /^\[.*\]$/.test(String(handle ?? "").trim().replace(/^@+/, "").trim()))) {
    throw new Error("Publishable video still has placeholder social handles");
  }
  // Answer-only reveal: the badge joins the south lane beside the name pill, clear of the raised card.
  const answerOnlyReveal = showReveal && !showJustification;
  const revealPanel = showReveal ? (
    <section
      className="reveal-reason"
      data-selected-action={selectedActionCode || undefined}
      data-winning-root-action-count={String(analysis.winningRootActionCount ?? "")}
    >
      <div className={showJustification ? "reason" : "reason is-answer-only"} role="status">
        <svg className="reason-band" width="780" height="14" aria-hidden="true"><rect width="780" height="14" fill="url(#mj-tris)" /></svg>
        <div className="reason-body">
          {selectedActionCode ? <div className="pick">
            <small>{localized("video.decision.best_move")}</small>
            <strong className="winning-move-code" dir="auto" aria-label={decisionKind === "play" && recommended ? cardLabel(recommended) : undefined}>{
              decisionKind === "play"
                ? (recommended ? <Card size="small" card={recommended} className="pick-card" /> : selectedActionCode)
                : actionLabel(legalChoices.find((action) => action.code === selectedActionCode) ?? {code: selectedActionCode})
            }</strong>
          </div> : null}
          {showJustification ? <div className="reason-text">
            <p>{reason}</p>
            {proofScopeKey ? <p className="scope" data-proof-visibility={String(analysis.achievedVisibility ?? "none")}>
              {localized(proofScopeKey)}
            </p> : null}
            {sunTieKey && bundle.localizations[sunTieKey] ? <p className="scope" data-sun-tie-outcome={String(analysis.sunTieOutcome)}>
              {localized(sunTieKey)}
            </p> : null}
          </div> : null}
        </div>
      </div>
    </section>
  ) : null;

  return (
    <AbsoluteFill className={`canvas majlis ${rewinding || activeTimelineSegment?.stage === "rewind" ? "is-rewinding" : ""}`} dir="rtl" style={paletteStyle}>
      <style>{`
        @font-face{font-family:NotoSansArabic;src:url('${staticFile("fonts/NotoSansArabic-Regular.ttf")}')}
        @font-face{font-family:NotoSansArabic;src:url('${staticFile("fonts/NotoSansArabic-Bold.ttf")}');font-weight:700}
        @font-face{font-family:NotoKufiArabic;src:url('${staticFile("fonts/NotoKufiArabic-Bold.ttf")}');font-weight:700}
      `}</style>
      <MajlisDefs watermark={researchShort} />
      <MajlisCeiling />
      <div className="jalsa jalsa-left" />
      <div className="jalsa jalsa-right" />
      <MajlisFloor />
      {rewinding || activeTimelineSegment?.stage === "rewind" ? <div className="rewind-mark" aria-hidden="true">↶</div> : null}

      <header className={`broadcast-hud hud hud-${hudStyle}`} data-score-us={score[0] ?? 0} data-score-them={score[1] ?? 0} aria-label={localized("video.accessibility.score_and_game")}>
        <div className="cell cell-us">
          <div className="tile">
            <small>{biddingOpen ? localized("video.hud.bidding_round") : localized("video.hud.trick")}</small>
            <b dir="ltr">{biddingOpen ? biddingRound : trickNumber}<span>/{biddingOpen ? 2 : 8}</span></b>
          </div>
          <TeamScore team="us" label={localized("video.result.us")} score={score[0] ?? 0} target={scoreTarget} />
        </div>
        <div className="crest">
          <div className="medal">
            {contractPending ? <>
              {purchaseCard ? <Card size="small" card={purchaseCard} className="medal-card" /> : null}
              <span className="medal-word">{localized("video.hud.purchase")}</span>
            </> : <>
              {contractName === "hokum" && suitSymbol[trumpName] ? <span className={`suit-chip ${redSuit(trumpName) ? "red" : ""}`}>{trump}</span> : null}
              <span className="medal-word">{localized(`video.bid.${contractName}`)}</span>
            </>}
            {multiplier > 1 ? <span className="xbadge" dir="ltr">{`×${multiplier}`}</span> : null}
          </div>
        </div>
        <div className="cell cell-them">
          <TeamScore team="them" label={localized("video.result.them")} score={score[1] ?? 0} target={scoreTarget} />
          {showIdentity ? (
            <div
              className="header-identity"
              data-decision-reference={publicDecisionReference}
              data-exact-state-id={legacyExactStateId}
            >
              {auditMedia ? (
                <div className="audit-media-badge rstamp" role="status" aria-label={localized("video.governance.research_audit_badge")}>
                  <b>{researchShort}</b><small>{notPublishable}</small>
                </div>
              ) : showQr && idBox === "qr" ? (
                <div className="qr qr-shell"><QRCodeSVG value={bundle.identity.enrollmentUri} size={qrPixels} level="M" marginSize={qrMarginModules} bgColor={String(palette.sand)} fgColor={String(palette.ink)} title={localized("video.accessibility.qr")} /></div>
              ) : (
                <div className="idnum game-number is-notice"><small>{localized("video.hud.session")}</small><b dir="ltr" style={{"--id-cols": Math.ceil(idCode.length / (idCode.length > 5 ? 2 : 1))} as React.CSSProperties}>{idCode}</b></div>
              )}
            </div>
          ) : null}
        </div>
      </header>

      {(["north", "west", "east"] as const).map((visualSeat) => {
        const actualSeat = actualAtVisual(visualSeat);
        return <SeatDock key={visualSeat} actualSeat={actualSeat} seat={visualSeat} count={counts[actualSeat]} active={!showOutcome && currentPlayer === actualSeat} name={playerNames[actualSeat]} />;
      })}
      {(["west", "east"] as const).map((visualSeat) => <React.Fragment key={visualSeat}>{tagsFor(visualSeat)}</React.Fragment>)}

      <main className="play-table">
        {capturePlaybackDom ? <PlaybackDomArtifact frame={frame} /> : null}
        <MajlisRug />
        {auditMedia ? <svg className="watermark" width="616" height="486" aria-hidden="true"><rect width="616" height="486" fill="url(#mj-wm)" /></svg> : null}
        {biddingOpen ? <div className="setup-deck" aria-hidden="true">
          {[0, 1, 2].map((index) => (
            <svg key={index} className="deck" width="140" height="196" viewBox="0 0 84 118" style={{left: 246 + index * 4, top: 256 - index * 4}}><use href="#mj-cb" /></svg>
          ))}
          {purchaseCard ? <Card size="small" card={purchaseCard} className="purchase-card" /> : null}
        </div> : null}
        <div
          className="trick-zone"
          data-completed-play-count={completedPlayCount}
          data-current-trick-count={currentTrickCount}
          data-table-play-count={plays.length}
          data-collection-progress={collectionProgress}
          data-table-opacity={tableOpacity}
          data-visible-table-card-ids={JSON.stringify(tableOpacity > 0 ? plays.map(play => play.card.id) : [])}
          data-completed-trick-index={completedTrickIndex ?? undefined}
          data-source-play-count={sourcePlayCount}
        >
          {plays.map((play, index) => {
            const entering = isFlagship && index === plays.length - 1 && playProgress < 1;
            const visualSeat = visualOfActual(play.seat);
            const offset = entryOffset[visualSeat];
            const collected = collectionOffset[visualOfActual(winner ?? play.seat)];
            const baseTransform = entering
              ? `${destinationTransform[visualSeat]} translate(${offset[0] * (1 - playProgress)}px, ${offset[1] * (1 - playProgress)}px) scale(${0.82 + 0.18 * playProgress})`
              : destinationTransform[visualSeat];
            const transform = collectionProgress > 0
              ? `${baseTransform} translate(${reducedMotion ? 0 : collected[0] * collectionEase}px, ${reducedMotion ? 0 : collected[1] * collectionEase}px) scale(${1 - 0.24 * collectionEase})`
              : baseTransform;
            const winning = winner === play.seat && completedPlayCount % 4 === 0 && collectionProgress < 0.12;
            const answer = replaysFromDeal && ["hold", "continuation"].includes(String(activeTimelineSegment?.stage)) && play.card.id === recommended?.id;
            return <Card key={`${play.seat}-${play.card.id}`} card={play.card} className={`played ${winning ? "is-trick-winner" : ""}${answer ? " is-answer" : ""}`} style={{zIndex: index + 1, transform, opacity: tableOpacity}} />;
          })}
          {demoMotionCard && frame > 0 ? (
            <Card card={demoMotionCard} className="played motion-card" style={{
              opacity: reducedMotion ? interpolate(demoMotionFrame, [0, 8], [0, 1], {extrapolateRight: "clamp"}) : 1,
              transform: reducedMotion
                ? "translate(-50%, -50%)"
                : `translate(calc(-50% + ${interpolate(demoMotionProgress, [0, 1], [0, -105])}px), calc(-50% + ${interpolate(demoMotionProgress, [0, 1], [610, 0])}px)) scale(${interpolate(demoMotionProgress, [0, 1], [1.06, 1])})`,
            }} />
          ) : null}
        </div>
      </main>

      {setupAction ? <section className="call-layer" data-setup-action-index={setupActionIndex} data-setup-action-kind={String(setupAction.kind)} data-setup-actor={String(setupAction.seat)}>
        {setupCalls.filter((item) => item.visual !== "north").map((item) => item.node)}
      </section> : null}
      {playCallouts.filter((item) => item.visual !== "north").map((item) => item.node)}

      <div className="seat-lane lane-north" dir="ltr">
        <div className="lane-side lane-start" dir="rtl">{tagsFor("north")}</div>
        <div className="lane-spacer" aria-hidden="true"><SeatPill count={counts[northActual]} name={playerNames[northActual]} active={false} /></div>
        <div className="lane-side lane-end" dir="rtl">{[...setupCalls, ...playCallouts].filter((item) => item.visual === "north").map((item) => item.node)}</div>
      </div>
      <div className={`seat-lane lane-south${answerOnlyReveal ? " is-reveal" : ""}`} dir="ltr">
        <div className="lane-side lane-start" dir="rtl">{tagsFor("south")}</div>
        <div className="lane-spacer" aria-hidden="true"><SeatPill count={counts[observer]} name={playerNames[observer]} active={false} /></div>
        <div className="lane-side lane-end">{answerOnlyReveal ? revealPanel : null}</div>
      </div>

      {showPrompt ? <>
        <div className="handglow" aria-hidden="true" />
        <section className="decision-prompt">
          <div className="ask" role="status">
            <span className="timer" style={{background: `conic-gradient(${paint("decision_red")} 0 ${promptFraction * 100}%, ${paint("timer_track")} 0)`}}><span>{promptTimer}</span></span>
            <span><small>{formatted("video.decision.your_turn", {name: playerNames[actingSeat]})}</small><strong>{localized(questionKey)}</strong></span>
          </div>
          {decisionKind !== "play" ? <ul className="decision-choices" aria-label={localized("video.accessibility.decision_choices")}>
            {legalChoices.map((action) => <li key={String(action.code)} dir="auto">{actionLabel(action)}</li>)}
          </ul> : null}
        </section>
      </> : null}
      {answerOnlyReveal ? null : revealPanel}
      {showOutcome && terminalOutcomeAvailable ? (
        <section className={`outcome-card ledger ${showScoreBreakdown ? "has-breakdown" : ""}`}>
          <svg width="500" height="14" aria-hidden="true" style={{display: "block"}}><rect width="500" height="14" fill="url(#mj-tris)" /></svg>
          <h2>{localized("video.result.title")}</h2>
          {showScoreBreakdown ? <div className="result-ledger">
            <div className="lrow lhead result-heading"><span /><b>{localized("video.result.us")}</b><b>{localized("video.result.them")}</b></div>
            <div className="lrow result-row"><span>{localized("video.result.round")}</span><b dir="ltr">{cardGamePoints[0] ?? 0}</b><b dir="ltr">{cardGamePoints[1] ?? 0}</b></div>
            <div className="lrow result-row project-points"><span>{localized("video.result.projects")}</span><b dir="ltr" className={positive(projectPoints[0] ?? 0)}>+{projectPoints[0] ?? 0}</b><b dir="ltr" className={positive(projectPoints[1] ?? 0)}>+{projectPoints[1] ?? 0}</b></div>
            {balootPoints.some((value) => value > 0) ? <div className="lrow result-row"><span>{localized("video.result.baloot")}</span><b dir="ltr" className={positive(balootPoints[0] ?? 0)}>+{balootPoints[0] ?? 0}</b><b dir="ltr" className={positive(balootPoints[1] ?? 0)}>+{balootPoints[1] ?? 0}</b></div> : null}
            <div className="lrow result-row awarded-points"><span>{localized("video.result.awarded")}</span><b dir="ltr">{awardedPoints[0] ?? 0}</b><b dir="ltr">{awardedPoints[1] ?? 0}</b></div>
            <div className="lrow ltotal result-total"><span>{localized("video.result.total")}</span><b dir="ltr" className="total-us">{String(finalScore[0] ?? 0)}</b><b dir="ltr" className="total-them">{String(finalScore[1] ?? 0)}</b></div>
          </div> : null}
        </section>
      ) : null}
      {showOutcome && !terminalOutcomeAvailable && showJustification ? (
        <section className="reveal-reason proof-boundary-only">
          <div className="reason" role="status"><div className="reason-body"><div className="reason-text"><p>{reason}</p></div></div></div>
        </section>
      ) : null}

      {!showPrompt ? <div className={`viewer-identity seat-row row-south${answerOnlyReveal ? " is-reveal" : ""}`}>
        <SeatPill count={counts[observer]} name={playerNames[observer]} active={!showOutcome && currentPlayer === observer} />
      </div> : null}

      {String(social.tiktok ?? "") ? <div className="social social-tiktok" aria-label="TikTok">
        <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={stroke("team_us_text")}><path d="M14 3v11.5a3.5 3.5 0 1 1-3.5-3.5" /><path d="M14 3c.6 2.6 2.4 4.3 5 4.6" /></svg>
        <span dir="ltr">{`@${String(social.tiktok ?? "")}`}</span>
      </div> : null}
      {String(social.instagram ?? "") ? <div className="social social-instagram" aria-label="Instagram">
        <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true" fill="none" strokeWidth="2" style={stroke("team_us_text")}><rect x="3" y="3" width="18" height="18" rx="5" /><circle cx="12" cy="12" r="4" /><circle cx="17.3" cy="6.7" r="1" stroke="none" style={fill("team_us_text")} /></svg>
        <span dir="ltr">{`@${String(social.instagram ?? "")}`}</span>
      </div> : null}

      <section className={`observer-hand ${!showOutcome && currentPlayer === observer ? "is-active" : ""}`} aria-label={localized("video.accessibility.hand")} data-seat={observer} data-hand-count={hand.length}>
        {hand.map((card, index) => {
          if (!isFlagship && frame > 0 && index === 0) return null;
          const middle = (hand.length - 1) / 2;
          const offset = index - middle;
          const spacing = Math.min(102, 720 / Math.max(1, hand.length - 1));
          const isRecommended = isActorChallenge && recommended?.id === card.id && showReveal;
          return <Card key={card.id} card={card} className={isRecommended ? "is-recommended" : ""} style={{
            transform: `translateX(${offset * spacing}px) rotate(${offset * 1.8}deg) translateY(${isRecommended ? -34 : Math.abs(offset) * 4}px)`,
            zIndex: index + 1,
          }} />;
        })}
      </section>

      {auditMedia ? <>
        <div className="rhead" role="note">
          <span className="rhead-icon" aria-hidden="true">!</span>
          <span className="rhead-text"><b>{localized("video.governance.research_title")}</b><span>{notPublishable} · <span className="mono" dir="ltr">{bundle.identity.gameSerial}</span></span></span>
        </div>
        <div className="tape tape-top" /><div className="tape tape-bottom" /><div className="tape tape-left" /><div className="tape tape-right" />
      </> : researchNotice ? (
        <div className="rhead research-notice" role="note">
          <span className="rhead-icon" aria-hidden="true">!</span>
          <span className="rhead-text"><b>{localized("video.notice.title")}</b><span className="notice-sentence" dir="rtl">{localized("video.notice.sentence")}</span></span>
        </div>
      ) : null}
    </AbsoluteFill>
  );
};

export const ProfessionalPrototype: React.FC<PrototypeProps> = (props) => (
  <ProfessionalPrototypeFrame {...props} compositionFrame={useCurrentFrame()} />
);
