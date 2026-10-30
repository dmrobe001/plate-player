// Code version, injected by build.mjs so that records can say which build wrote them.
/* global __BUILD_INFO__ */
export const BUILD_INFO = typeof __BUILD_INFO__ !== 'undefined' ? __BUILD_INFO__ : { commit: 'unbuilt', dirty: true, built: null };
