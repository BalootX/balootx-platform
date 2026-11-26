// Classic page scripts reach the request helper through this loader. A failed load
// is not kept: the next call imports again. The retry uses a new URL because the
// browser's module map keeps a failed fetch for the life of the document.
(function (root) {
  function createRequestHelperLoader(importModule) {
    let pending = null;
    let attempts = 0;
    return function loadRequestHelper() {
      if (!pending) {
        const specifier = attempts
          ? `/shared-assets/request.js?retry=${attempts}`
          : '/shared-assets/request.js';
        attempts += 1;
        pending = importModule(specifier).catch((error) => {
          pending = null;
          throw error;
        });
      }
      return pending;
    };
  }
  root.createRequestHelperLoader = createRequestHelperLoader;
  root.loadRequestHelper = createRequestHelperLoader((specifier) => import(specifier));
})(globalThis);
