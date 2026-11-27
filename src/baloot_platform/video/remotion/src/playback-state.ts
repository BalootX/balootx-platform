import type {CardValue} from "./types";

export type Seat = "north" | "west" | "east" | "south";
export type PlaybackPlay = {
  seat: Seat;
  card: CardValue;
  trickIndex?: number;
  position?: number;
  baloot?: boolean;
};
export type PlaybackTrick = {
  index: number;
  plays: PlaybackPlay[];
  winner: Seat;
};

export type PlaybackState = {
  completedPlayCount: number;
  visibleHistory: PlaybackPlay[];
  hand: CardValue[];
  handCounts: Record<Seat, number>;
  currentTrick: PlaybackPlay[];
  tablePlays: PlaybackPlay[];
  completedTrickIndex: number | null;
  completedTrickWinner: Seat | null;
  currentPlayer: Seat;
};

const seats: Seat[] = ["north", "west", "south", "east"];

// Zero opacity is not visible even though the collected cards remain in the DOM.
export const collectionOpacity = (progress: number): number =>
  progress <= 0.72 ? 1 : Math.max(0, 1 - (progress - 0.72) / 0.28);

export const derivePlaybackState = ({
  initialHand,
  allPlays,
  allTricks,
  observer,
  completedPlayCount,
}: {
  initialHand: CardValue[];
  allPlays: PlaybackPlay[];
  allTricks: PlaybackTrick[];
  observer: Seat;
  completedPlayCount: number;
}): PlaybackState => {
  const count = Math.min(allPlays.length, Math.max(0, Math.trunc(completedPlayCount)));
  const visibleHistory = allPlays.slice(0, count);
  const hand = initialHand.filter((card) => !visibleHistory.some(
    (play) => play.seat === observer && play.card.id === card.id,
  ));
  const handCounts = Object.fromEntries(seats.map((seat) => [
    seat,
    8 - visibleHistory.filter((play) => play.seat === seat).length,
  ])) as Record<Seat, number>;
  handCounts[observer] = hand.length;

  const positionInTrick = count % 4;
  const currentTrick = positionInTrick === 0
    ? []
    : visibleHistory.slice(count - positionInTrick);
  const completedTrick = count > 0 && positionInTrick === 0
    ? allTricks[Math.floor(count / 4) - 1]
    : undefined;

  return {
    completedPlayCount: count,
    visibleHistory,
    hand,
    handCounts,
    currentTrick,
    tablePlays: completedTrick?.plays ?? currentTrick,
    completedTrickIndex: completedTrick?.index ?? null,
    completedTrickWinner: completedTrick?.winner ?? null,
    currentPlayer: count < allPlays.length
      ? allPlays[count].seat
      : allTricks.at(-1)?.winner ?? observer,
  };
};
