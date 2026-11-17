export type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | {[key: string]: JsonValue};

export type JsonObject = Record<string, JsonValue>;

export type SeatName = 'north' | 'west' | 'south' | 'east';

export function isSeatName(value: unknown): value is SeatName {
  return value === 'north' || value === 'west' || value === 'south' || value === 'east';
}

export function copyJson<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}
