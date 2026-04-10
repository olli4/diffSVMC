/**
 * Differentiable SVMC submodels — jax-js-nonconsuming port.
 *
 * Submodels are ported bottom-up from the Fortran SVMC source.
 * See DEPENDENCY-TREE.md for the full dependency graph.
 */
export {
	_setCalibrationState,
	defaultDevice,
	getBackend,
	init,
	jit,
	tree,
} from "@hamk-uas/jax-js-nonconsuming";
export * from "./precision.js";
export * from "./integration.js";
export * from "./allocation/index.js";
export * from "./phydro/index.js";
export * from "./water/index.js";
export * from "./yasso/index.js";
