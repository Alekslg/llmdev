(function () {
  const PREFIX = "scrollPosition_";
  const SAVE_INTERVAL = 200;
  let saveTimeout = null;

  function getKey() {
    return PREFIX + window.location.pathname;
  }

  function saveScroll() {
    localStorage.setItem(getKey(), String(window.scrollY || 0));
  }

  function debouncedSave() {
    clearTimeout(saveTimeout);
    saveTimeout = setTimeout(saveScroll, SAVE_INTERVAL);
  }

  function restoreScroll() {
    const savedY = parseInt(localStorage.getItem(getKey()) || "0", 10);
    if (!isNaN(savedY)) {
      requestAnimationFrame(() => {
        window.scrollTo({ top: savedY, behavior: "auto" });
      });
    }
  }

  // Сохраняем позицию при прокрутке
  window.addEventListener("scroll", debouncedSave);

  // Ловим переходы между страницами MkDocs
  if (window.document$) {
    // MkDocs Material instant navigation
    window.document$.subscribe(() => {
      console.debug(
        "[scroll-restore] page changed → restoring scroll for",
        window.location.pathname
      );
      restoreScroll();
    });
  } else {
    // fallback для обычной навигации
    window.addEventListener("DOMContentLoaded", restoreScroll);
    window.addEventListener("beforeunload", saveScroll);
  }
})();
