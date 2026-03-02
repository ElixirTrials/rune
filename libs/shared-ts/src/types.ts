/**
 * Shared type definitions.
 */

/** Common result wrapper for API responses. */
export interface Result<T, E = Error> {
  ok: true;
  value: T;
}

export interface ResultErr<E = Error> {
  ok: false;
  error: E;
}

export type ResultType<T, E = Error> = Result<T, E> | ResultErr<E>;
