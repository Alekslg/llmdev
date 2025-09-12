/* Автосохранение выбранной темы + авто Book mode для мобильных */
(function () {
  const key = "data-md-color-scheme";
  const stored = localStorage.getItem(key);

  // Проверка: мобильное устройство или маленький экран
  const isMobile = window.matchMedia("(max-width: 768px)").matches;

  if (stored) {
    // Если пользователь уже выбирал — применяем сохранённый режим
    document.body.setAttribute(key, stored);
  } else {
    // Если первый раз — выбираем по устройству
    document.body.setAttribute(key, isMobile ? "book" : "default");
  }

  // Следим за изменением темы и сохраняем
  const observer = new MutationObserver(() => {
    const current = document.body.getAttribute(key);
    if (current) {
      localStorage.setItem(key, current);
    }
  });

  observer.observe(document.body, {
    attributes: true,
    attributeFilter: [key],
  });
})();
