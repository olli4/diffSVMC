# ESLint: warn when a borrowed parameter escapes via return

## Component

`@hamk-uas/jax-js-nonconsuming/eslint-plugin`

## Problem

When a function receives an `np.Array` parameter and returns it directly
(pass-through), neither the caller's `using` binding nor the receiver's
`using` binding knows the other exists. Both will call `dispose()`,
causing a double-dispose.

The correct fix is to return `param.ref` (zero-copy refcount increment),
but nothing in the current `recommended` rule set catches the omission.

### Example — buggy

```ts
function foo(x: np.Array): { x: np.Array } {
  return { x };  // ← x escapes without .ref; double-dispose if caller uses `using`
}
```

### Example — correct

```ts
function foo(x: np.Array): { x: np.Array } {
  return { x: x.ref };  // ← refcount incremented; safe
}
```

### Workaround we used before discovering `.ref`

```ts
return { x: x.add(0.0) };  // creates a whole new GPU buffer — wasteful
```

## Requested behaviour

Add a lint rule (candidate for the `strict` tier mentioned in the repo's
`eslint.config.js`) that warns when:

1. A function parameter typed as `np.Array` (or `Tracer`) appears in a
   return expression without passing through an operation or `.ref`.
2. Ideally also catches the `.add(0.0)` workaround and suggests `.ref`.

## Context

Discovered while porting SVMC allocation submodels to `svmc-js`. The
`no-nested-array-leak` rule in `strict` may partially overlap — worth
checking whether it already covers this case before adding a new rule.

## Resolution

**Fixed in v0.12.6** — New ESLint rule `no-borrowed-param-return`
(`@hamk-uas/eslint-plugin-jax-js` v0.1.3) flags functions that return a
borrowed `np.Array` or `Tracer` parameter without retaining via `.ref`.
